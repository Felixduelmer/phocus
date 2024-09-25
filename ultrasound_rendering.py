import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.layers import HilbertTransform
from mpl_toolkits.axes_grid1 import make_axes_locatable


from typing import Tuple

class Ultrasound_Rendering(torch.nn.Module):
    def __init__(self, 
                 signal_frequency: float,
                 num_cycles: float,
                 f_number: float,
                 probe_width: float,
                 probe_depth: float,
                 image_dimensions: Tuple[int, int],
                 visualization: bool,
                 sampling_factor: float = 1.0,
                 device: str = "cuda"):
        super(Ultrasound_Rendering, self).__init__()
        self.signal_frequency = signal_frequency # [Hz]
        self.num_cycles = num_cycles # number of cycles in the tone burst
        self.f_number = f_number # with this we can calculate the focal_depth
        self.probe_depth = probe_depth # [m]
        self.probe_width = probe_width # [m]
        self.device = device
        self.sound_speed = 1540 # [m/s] let us assume this to be constant
        self.D = torch.tensor(self.probe_width/2, device=self.device) # [m] aperture size
        self.H, self.W = image_dimensions
        self.hilbert_transform = HilbertTransform()

        # calculate some distances
        self.wavelength = torch.tensor(self.sound_speed / self.signal_frequency, device=device)
        self.tone_length = self.num_cycles * self.wavelength
        self.half_axial_extent_psf = self.tone_length / 2.0
        self.half_lateral_extent_psf =  self.f_number * self.wavelength *2 # Rayleighs diffraction limit, we take 1 lobe of the sinc function

        # sampling factor
        self.sampling_factor = sampling_factor

        self.sw = self.probe_width / (self.W * self.sampling_factor)
        self.sh = self.probe_depth / (self.H * self.sampling_factor)

        self.N_samples_lateral = int(self.probe_width/self.sw)
        self.N_samples_axial = int(self.probe_depth/self.sh)

        self.sw_image = self.probe_width / self.W
        self.sh_image = self.probe_depth / self.H

        #define the PSF functions
        self.ax_signal = lambda y: torch.exp(-1/2 * (3*(1/(self.tone_length/2))*y) ** 2)
        self.lat_signal = lambda x, f: torch.pow(torch.sinc(x / (f * self.wavelength)), 2)

        if visualization:
            self.visualize_PSF()

    def visualize_PSF(self) -> None:

        ax_points = torch.linspace(0.0, 1.0, int(torch.floor(self.half_axial_extent_psf/self.sh)), device=self.device) * self.half_axial_extent_psf
        ax_points = torch.cat((torch.flip(-ax_points[1:], dims=[0]), ax_points))
        ax_signal = self.ax_signal(ax_points)

        lat_points = torch.linspace(0.0, 1.0, int(torch.floor(self.half_lateral_extent_psf/self.sw)), device=self.device) * self.half_lateral_extent_psf
        lat_points = torch.cat((torch.flip(-lat_points[1:], dims=[0]), lat_points))
        lat_signal = self.lat_signal(lat_points, self.f_number)

        lateral_distribution = lat_points.cpu().numpy()
        axial_distribution = ax_points.cpu().numpy()
        self.psf_kernel = torch.einsum('i,j->ij', ax_signal, lat_signal)


        # Create a figure and a grid of subplots
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the 2D PSF kernel
        im = ax.imshow(self.psf_kernel.cpu().numpy(), extent=[lateral_distribution[0]*1000, lateral_distribution[-1]*1000, axial_distribution[0]*1000, axial_distribution[-1]*1000], vmin=-1, vmax=1)
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)   

        # Create an axis on the right for the axial signal
        divider = make_axes_locatable(ax)
        ax_axial = divider.append_axes("right", size="10%", pad=0.1)
        ax_axial.plot(ax_signal.cpu().numpy(), axial_distribution*1000)
        ax_axial.yaxis.tick_right()
        ax_axial.yaxis.set_label_position("right")
        ax_axial.set_ylabel('Axial Distribution [mm]')

        # Create an axis on the bottom for the lateral signal
        ax_lateral = divider.append_axes("bottom", size="15%", pad=0.1)
        ax_lateral.plot(lateral_distribution*1000, lat_signal.cpu().numpy())
        ax_lateral.set_xlabel('Lateral Distribution [mm]')
        ax_lateral.xaxis.tick_bottom()
        ax_lateral.xaxis.set_label_position("bottom")
        ax_lateral.yaxis.set_visible(True)

        plt.tight_layout()
        plt.colorbar(im, ax=ax)
        plt.show()
                        
        self.psf_kernel = torch.tensor(self.psf_kernel, dtype=torch.float32).to(device=self.device)

    def convolutional_rendering(self, raw, random_vals, raw_noise_std=0):
        # we are expecting a single map of shape (B, C, H, W)
        N_rays, W, Maps = raw.shape 

        scatterer = torch.exp(raw[...,0])
        scatterer = scatterer.transpose(1, 0)[None, None, ...]
        rand_ax, rand_lat = random_vals

        # define the PSF 
        ax_values = torch.linspace(0.0, 1.0, int(torch.floor(self.half_axial_extent_psf/self.sh).item()), device=self.device) * self.half_axial_extent_psf
        self.ax_padding = ax_values[1:].shape[0]
        ax_values = torch.cat((torch.flip(-ax_values[1:], dims=[0]), ax_values))  + rand_ax * self.sh
        lat_values = torch.linspace(0.0, 1.0, int(torch.floor(self.half_lateral_extent_psf/self.sw).item()), device=self.device) * self.half_lateral_extent_psf
        self.lat_padding = lat_values[1:].shape[0]
        lat_values = torch.cat((torch.flip(-lat_values[1:], dims=[0]), lat_values)) + rand_lat * self.sw
        ax_signal = self.ax_signal(ax_values)
        lat_signal = self.lat_signal(lat_values, self.f_number)
        psf = torch.outer(ax_signal, lat_signal)[None, None, ...]
        psf = psf / torch.sum(psf)

        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.relu(torch.normal(0, raw_noise_std,  (self.H, self.W)))[None, None, ...]

        padded_scatterer = F.pad(scatterer, (self.lat_padding, self.lat_padding, self.ax_padding, self.ax_padding), mode="reflect")
        envelope_data = torch.nn.functional.conv2d(padded_scatterer, psf, stride=int(self.sampling_factor), padding="valid") + noise
        

        b_mode = 20 * torch.log10(envelope_data+1.0)
        scatterer = 20 * torch.log10(scatterer+1.0)

        return {"b_mode": b_mode, "scatterers_map": scatterer, "envelope_data": envelope_data} 
