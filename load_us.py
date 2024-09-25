import numpy as np
import os, imageio
from utils import get_bbox3d_for_us
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _load_data(basedir):
    # poses are 4x4 [R T] matrices
    poses = np.load(os.path.join(basedir, 'poses.npy'))

    if len(poses.shape) == 2:
        poses = poses[None, ...]
    sfx = ''

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = sorted([os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                       f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')], key=lambda i:
    int(i.split("/")[-1].replace(".png", "")))
    if poses.shape[0] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[0]))
        return

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f)
        else:
            return imageio.imread(f)
    imgs = [imread(f) / 255. for f in imgfiles]
    imgs = np.stack(imgs)
    poses[:, :3, 3] *= 0.001
    print('Loaded image data', imgs.shape, poses.shape)
    return poses, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds


def load_us_data(basedir, probe_width, near=0.0, far=1.0):
    poses, imgs = _load_data(basedir)
    print('Loaded', basedir)
    images = imgs

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, )

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    bounding_box = get_bbox3d_for_us(poses, probe_width, near, far)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    # visualize_poses(poses)

    return images, poses, i_test, bounding_box


def visualize_poses(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    poses = np.array([poses[i, :, :] for i in np.arange(100)]) # np.arange(0, poses.shape[0], 100)])

    for pose in poses:
        # Extract translation (position) from the last column
        x, y, z = pose[:3, 3]

        # Create basis vectors for the pose
        # The columns of the rotation matrix represent the basis vectors
        basis_vectors = pose[:3, :3]  # Scale for visualization purposes

        # Draw the basis vectors
        ax.quiver(x, y, z, *basis_vectors[:, 0], color='r', length=0.001, normalize=True)
        ax.quiver(x, y, z, *basis_vectors[:, 1], color='g', length=0.001, normalize=True)
        ax.quiver(x, y, z, *basis_vectors[:, 2], color='b', length=0.001, normalize=True)

    # Setting the labels for axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax.set_aspect('equal', 'box')
    plt.show()




if __name__ == "__main__":
    # visualize the poses:
    poses = np.load("data/synthetic_speckle_cube/all_train/poses.npy")
    visualize_poses(poses)