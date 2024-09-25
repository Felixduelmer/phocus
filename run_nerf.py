import os
import numpy as np
import imageio
import torch
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from radam import RAdam

from load_us import load_us_data
from ray_utils import get_rays_us_linear
import wandb

from monai.losses import SSIMLoss
from ultrasound_rendering import Ultrasound_Rendering

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded, keep_mask = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    if keep_mask is not None:
        outputs_flat[~keep_mask, -1] = 0 # set sigma to 0 for invalid points
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, us_renderer=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
            ret = render_rays_us(rays_flat[i:i+chunk], us_renderer, **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_rays_us(ray_batch, us_renderer, network_fn, network_query_fn, perturb=0., raw_noise_std=0.,**kwargs):
    """Volumetric rendering.

    Args:
    ray_batch: Tensor of shape [batch_size, ...]. We define rays and do not sample.

    Returns:
    Rendered outputs.
    """

    def raw2outputs(raw, random_vals, raw_noise_std):
        """Transforms model's predictions to semantically meaningful values."""
        ret = us_renderer.convolutional_rendering(raw, random_vals, raw_noise_std)
        return ret

    ###############################
    # Batch size
    N_rays = ray_batch.shape[0]
    viewdirs = None

    # Extract ray origin, direction
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance
    bounds = ray_batch[..., 6:8].reshape(-1, 1, 2)
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray
    z_vals = torch.linspace(0., 1., us_renderer.N_samples_axial) * far

    z_vals_rand = 0.0
    x_vals_rand = 0.0
    
    if perturb > 0.:
        z_vals_rand = torch.rand((1))
        z_vals = z_vals + (z_vals[1] - z_vals[0]) * z_vals_rand
        x_vals_rand = torch.rand((1))
        rays_o = rays_o + (rays_o[1] - rays_o[0]) * x_vals_rand
    
    z_vals = z_vals.expand(N_rays, us_renderer.N_samples_axial)
    origin = rays_o.unsqueeze(-2)
    step = rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)

    pts = step + origin
    raw = network_query_fn(pts, viewdirs, network_fn)

    ret = raw2outputs(raw, (z_vals_rand, x_vals_rand), raw_noise_std)
    return ret


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args, i=args.i_embed)
    if args.i_embed==1:
        # hashed embedding table
        embedding_params = list(embed_fn.parameters())

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args, i=args.i_embed_views)

    output_ch = args.output_ch
    skips = [4]

    if args.i_embed==1 and not args.only_inr:
        model = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views, output_ch=output_ch).to(device)
    else:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None

    if args.N_importance > 0:
        if args.i_embed==1:
            model_fine = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views).to(device)
        else:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    if args.i_embed==1:
        optimizer = RAdam([
                            {'params': grad_vars, 'weight_decay': 1e-6},
                            {'params': embedding_params, 'eps': 1e-15}
                        ], lr=args.lrate, betas=(0.9, 0.99))
        print('Using RAdam with weight decay for embedding table')
        print(f"Number of parameters: {sum(p.numel() for p in grad_vars if p.requires_grad)}")
        print(f"Number of embedding parameters: {sum(p.numel() for p in embedding_params if p.requires_grad)}")
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])

    ##########################
    # pdb.set_trace()

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn': embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer



def render_us(us_renderer, chunk=1024 * 32, bounding_box = None, rays=None, c2w=None, near=0., far=55. * 0.001, use_viewdirs=False,  **kwargs):
    """Render rays."""

    # assert rays is not None or c2w is not None
    assert rays is not None or c2w is not None

    if c2w is not None:
        # Special case to render full image
        rays_o, rays_d = get_rays_us_linear(us_renderer.N_samples_lateral, us_renderer.sw, c2w)
    else:
        # Use provided ray batch
        rays_o, rays_d = rays
    
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = rays_o.reshape(-1, 3).float()
    rays_d = rays_d.reshape(-1, 3).float()
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk=chunk,us_renderer=us_renderer, **kwargs)
    # for k in all_ret:
    #     k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
    #     all_ret[k] = all_ret[k].reshape(k_sh)
    return all_ret

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser(default_config_files=['configs/generic_config_us.txt'])
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=10,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=1,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=0,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='us',
                        help='options: us')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1000,
                        help='frequency of wandb image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=100,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000,
                        help='frequency of render_poses video saving')

    parser.add_argument("--finest_res",   type=int, default=1024,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size",   type=int, default=19,
                        help='log2 of hashmap size')
    parser.add_argument("--sparse-loss-weight", type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6,
                        help='learning rate')
    
    ## additional arguments
    parser.add_argument("--random_seed", type=int, default=0, 
                        help="set this to 1 for random seed")
    parser.add_argument("--loss", type=str, default='l2')
    parser.add_argument("--output_ch", type=int, default=5),
    parser.add_argument('--n_iters', type=int, default=10000)

    ## loss
    parser.add_argument("--ssim_filter_size", type=int, default=7)
    parser.add_argument("--ssim_lambda", type=float, default=0.5)

    ## ultrasound parameters
    parser.add_argument('--probe_depth', type=float, default=55)
    parser.add_argument('--probe_width', type=float, default=37)
    parser.add_argument('--signal_frequency', type=int, default=5000000)
    parser.add_argument('--num_cycles', type=int, default=3)
    parser.add_argument('--f_number', type=float, default=0.5)

    parser.add_argument('--sample_factor', type=float, default=1, help='Factor how much the image should be over or undersampled. E.g. if set to two, the sampling will be twice as large as the pixel size.')
    parser.add_argument('--only_inr', action='store_true')
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    use_wandb = False

    if args.random_seed == 0:
        print('Setting deterministic behaviour')
        random_seed = 10
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if use_wandb:
        wandb.init(project="phocus", name=args.expname)
        wandb.config.update(args)


    # Load data
    K = None
    
    if args.dataset_type == "us":

        # The poses are not normalized. We scale down the space.
        # It is possible to normalize poses and remove scaling.
        scaling = 0.001
        near = 0.
        probe_depth = args.probe_depth * scaling
        probe_width = args.probe_width * scaling 
        far = probe_depth

        images, poses, i_test, bounding_box = load_us_data(args.datadir, probe_width, near, far)
        args.bounding_box = bounding_box

        H, W = images.shape[1], images.shape[2]

        ultrasound_renderer = Ultrasound_Rendering(signal_frequency=args.signal_frequency, 
                                                   num_cycles=args.num_cycles, 
                                                   f_number=args.f_number, 
                                                   probe_width=probe_width, 
                                                   probe_depth=probe_depth, 
                                                   image_dimensions=[H, W], 
                                                   visualization=True, 
                                                   sampling_factor=args.sample_factor,
                                                   device=device)

        if not isinstance(i_test, list):
            i_test = [i_test]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0]))]) 
        print("Test {}, train {}".format(len(i_test), len(i_train)))
    
    



    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    
    if args.dataset_type == "us":
        # i_train = i_test

        # Losses
        ssim_weight = args.ssim_lambda
        l2_weight = 1. - ssim_weight
        ssim_loss = SSIMLoss(spatial_dims=2, data_range=1.0, kernel_type='gaussian', win_size=args.ssim_filter_size, k1=0.01, k2=0.1)
        tv_loss = TotalVariationLoss()

    else: 
        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if K is None:
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    if args.i_embed==1:
        args.expname += "_hashXYZ"
    elif args.i_embed==0:
        args.expname += "_posXYZ"
    if args.i_embed_views==2:
        args.expname += "_sphereVIEW"
    elif args.i_embed_views==0:
        args.expname += "_posVIEW"
    args.expname += "_fine"+str(args.finest_res) + "_log2T"+str(args.log2_hashmap_size)
    args.expname += "_lr"+str(args.lrate) + "_decay"+str(args.lrate_decay)
    args.expname += "_RAdam"
    args.expname += "_F_number" + str(args.f_number)
    args.expname += "_num_cycles" + str(args.num_cycles)
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    if args.render_test:
        render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                # images = None
                visualize_all = True
                if visualize_all:
                    poses = torch.Tensor(np.array(poses)).to(device)
                    testsavedir = os.path.join(basedir, expname, 'render_all')
                else:
                    poses = torch.Tensor(np.array(poses[i_test])).to(device)
                    testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses.shape)

            if args.dataset_type == "us":
                bbox = args.bounding_box

                total_params = []
                all_images = []
                for pose_i in tqdm(range(poses.shape[0])):
                    pose = poses[pose_i, :3,:4]


                    ## TODO: render_kwargs_test should be updated with the correct bounding box
                
                
                    ## TODO: render_kwargs_test should be updated with the correct bounding box
                
                    rendering_output = render_us(ultrasound_renderer, bounding_box= bounding_box, c2w=pose, chunk=args.chunk,
                        retraw=True, **render_kwargs_test)
                    
                    if visualize_all:
                        array = rendering_output['b_mode'][0, 0].cpu().numpy()
                        all_images.append(array)
                    
                    else:
                        #save the rendering output to a file
                        for _, (key, tensor) in enumerate(rendering_output.items()):
                            array = tensor[0, 0].cpu().numpy()
                            # Create a new figure for each image
                            fig, ax = plt.subplots(figsize=(20, 20))  # Adjust size as needed
                            grayscale = "b_mode" in key
                            cmap = "gray" if grayscale else "viridis"
                            im = ax.imshow(array, cmap=cmap)  # Change the colormap as appropriate
                            plt.colorbar(im, ax=ax)
                            ax.axis("off")

                            # Save the figure
                            plt.savefig(f'{testsavedir}/{key}_{pose_i}.png', bbox_inches='tight')
                            plt.close(fig)  # Close the figure to free memory
                            np.save(f'{testsavedir}/{key}_{pose_i}.npy', array)
                        
                            target = images[pose_i]
                            np.save(f'{testsavedir}/target_{pose_i}.npy', target)

                            array = target
                            fig, ax = plt.subplots(figsize=(20, 20))  # Adjust size as needed
                            im = ax.imshow(array, cmap="gray")  # Change the colormap as appropriate
                            plt.colorbar(im, ax=ax)
                            ax.axis("off")
                            plt.savefig("{}/target.png".format(testsavedir), bbox_inches='tight')
                            plt.close(fig)
                            params = []
                            for i in ["b_mode"]: # "envelope_image",
                                params.append(rendering_output[i][0, 0])
                            params = torch.stack(params, 0)
                            total_params.append(params)
                            plt.imshow(rendering_output["b_mode"][0, 0].cpu().numpy(), cmap='gray')
                            plt.colorbar()
                            plt.show()
                        total_params = torch.stack(total_params, 0)
                        np.save(os.path.join(testsavedir, 'rendering_output.npy'), total_params.cpu().numpy())
                if visualize_all:
                    # normalize to 0 and 255
                    all_images = np.array(all_images)
                    np.save(os.path.join(testsavedir, 'all_output.npy'), all_images)
                    images_dir = os.path.join(testsavedir, "images")
                    os.makedirs(os.path.join(images_dir,  "images"), exist_ok=True)
                    log_compressed_images = os.path.join(testsavedir, "log_compressed_images")
                    os.makedirs(log_compressed_images, exist_ok=True)
                    
                    all_images = (all_images - np.min(all_images)) / (np.max(all_images) - np.min(all_images)) * 255
                    all_images = all_images.astype(np.uint8)
                    for idx in range(all_images.shape[0]):
                        imageio.imwrite(f'{images_dir}/{idx:04d}.png', all_images[idx])

            print('Done rendering', testsavedir)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if args.dataset_type != "us":
        if use_batching:
            images = torch.Tensor(images).to(device)
        poses = torch.Tensor(poses).to(device)
        if use_batching:
            rays_rgb = torch.Tensor(rays_rgb).to(device)
    else:
        images = torch.Tensor(images).to(device)
        poses = torch.Tensor(poses).to(device)

    N_iters = 10000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)



    start = start + 1
    for i in trange(start, N_iters):
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]
            pose = torch.Tensor(pose).to(device)

        # perform randoming sampling factor for the ultrasound dataset between 1 and sample_factor
        rendering_output = render_us( ultrasound_renderer, c2w=pose, chunk=args.chunk,
                retraw=True, **render_kwargs_train)
        output_image = rendering_output['b_mode']
        scatterers_map = rendering_output['scatterers_map']

        optimizer.zero_grad()

        # loss computation
        loss = dict()
        if args.loss == 'l2':
                l2_intensity_loss = img2mse(output_image, target)
                loss["l2"] = (1., l2_intensity_loss)
        elif args.loss == 'ssim':
                target = target.unsqueeze(0).unsqueeze(0)
                ssim_intensity_loss = ssim_loss(output_image, target)
                loss["ssim"] = (ssim_weight, ssim_intensity_loss)
                l2_intensity_loss = img2mse(output_image, target)
                loss["l2"] = (l2_weight, l2_intensity_loss)
                total_variation_loss = tv_loss(scatterers_map)
                loss["total_variation"] = (1e-4, total_variation_loss)
        
        total_loss = 0.
        for key, loss_value in loss.items():
            # print(key, loss_value)
            total_loss += loss_value[0] * loss_value[1]
        total_loss.backward()
        optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.i_embed==1:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                test_imgs = [np.random.choice(i_val)]

                for img_i in test_imgs:
                    target = images[img_i]
                    target = torch.Tensor(target).to(device)
                    target = target[None, None, ...] 
                    pose = poses[img_i, :3,:4]
                    pose = torch.Tensor(pose).to(device)

                    rendering_output = render_us(ultrasound_renderer, c2w=pose, chunk=args.chunk,
                        retraw=True, **render_kwargs_test)
                    output_image = rendering_output['b_mode']

                    test_loss = dict()

                    if args.loss == 'l2':
                            l2_intensity_loss = img2mse(output_image, target)
                            test_loss["l2"] = (1., l2_intensity_loss)
                    elif args.loss == 'ssim':
                            ssim_intensity_loss = ssim_loss(output_image, target)
                            test_loss["ssim"] = (ssim_weight, ssim_intensity_loss)
                            l2_intensity_loss = img2mse(output_image, target)
                            test_loss["l2"] = (l2_weight, l2_intensity_loss)
                            total_variation_loss = tv_loss(scatterers_map) 
                            test_loss["total_variation"] = (1e-4, total_variation_loss)

                    total_test_loss = 0.
                    for loss_value in test_loss.values():
                        total_test_loss += loss_value[0] * loss_value[1]

                    # Loop through the items again to save each as a separate image
                    for _, (key, tensor) in enumerate(rendering_output.items()):
                        np.save(f'{testsavedir}/{key}.npy', tensor[0, 0].cpu().numpy())
                        array = tensor[0, 0].cpu().numpy()
                        # Create a new figure for each image
                        fig, ax = plt.subplots(figsize=(20, 20))  # Adjust size as needed
                        grayscale = "b_mode" in key
                        cmap = "gray" if grayscale else "viridis"
                        im = ax.imshow(array, cmap=cmap)  # Change the colormap as appropriate
                        plt.colorbar(im, ax=ax)
                        ax.axis("off")

                        # Save the figure
                        plt.savefig(f'{testsavedir}/{key}_{img_i}.png', bbox_inches='tight')
                        if i%args.i_img==0:
                            if use_wandb:
                                wandb.log({key: wandb.Image(fig), "step": i})
                        plt.close(fig)  # Close the figure to free memory

                    #normalized echogenicity map
                    echogenicity_map = rendering_output["scatterers_map"][0, 0].cpu().numpy()
                    echogenicity_map = ((echogenicity_map - np.min(echogenicity_map)) / (np.max(echogenicity_map) - np.min(echogenicity_map))) * 255
                    echogenicity_map = echogenicity_map.astype(np.uint8)
                    imageio.imwrite(f'{testsavedir}/echogenicity_map_{img_i}.png', echogenicity_map)

                    #save array
                    array = target.cpu().numpy()
                    fig, ax = plt.subplots(figsize=(20, 20))  # Adjust size as needed
                    im = ax.imshow(array[0, 0], cmap="gray")  # Change the colormap as appropriate
                    plt.colorbar(im, ax=ax)
                    ax.axis("off")
                    plt.savefig(f"{testsavedir}/target_{img_i}.png", bbox_inches='tight')
                    if i%args.i_img==0:
                        if use_wandb:
                            wandb.log({"target": wandb.Image(fig), "step": i})
                    if use_wandb:
                        wandb.log({"test_loss": total_test_loss.item()})
                    plt.close(fig)  # Close the figure to free memory
                print('Saved test set') 



        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {total_loss.item()}")
            if use_wandb:
                wandb.log({"train_loss": total_loss.item(), "step": i})

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_printoptions(precision=8)

    train()
