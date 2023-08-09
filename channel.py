import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from matplotlib.patches import Ellipse
from astropy.io import fits
import astropy.constants as const
import astropy.units as u
from dataclasses import dataclass
from typing import Tuple
from ipywidgets import interact

@dataclass
class _ChannelParameters:
    """
    Parameters for a channel instance
    """
    ra_offset: np.ndarray
    dec_offset: np.ndarray
    extent: Tuple
    beam: np.float64
    fwhm: float

class Channel:
    """
    Create channel maps, movies, and images from ALMA data
    """

    def __init__(self, filename: str, rest_freq: float):
        self.filename = filename
        self.data = fits.getdata(filename)
        self.header = fits.getheader(filename)
        self.rest_freq = rest_freq
        self.parameters = self._get_parameters(hd=self.header)

    def plot_slice(self, z, cmap = 'magma', wrt_source = False, source_velocity = None):

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 5))
        Tb = self.get_brightness_temperature(z)
        im = plt.imshow(Tb, cmap=cmap, origin='lower', extent = self.parameters.extent, aspect = 'auto')

        # Add velocity
        nu = self.header['CRVAL3'] + z * self.header['CDELT3']  
        v = const.c.to(u.km/u.s).value * (1 - nu / self.rest_freq)
        if wrt_source:
            ax.text(0.5, 0.05,  f'v = {np.round(v - source_velocity, 2)} km/s', transform=ax.transAxes, ha='center', va='bottom', fontsize=20, color = 'white')
        else:
            ax.text(0.5, 0.05,  f'v = {np.round(v, 2)} km/s', transform=ax.transAxes, ha='center', va='bottom', fontsize=20, color = 'white')

        # Set limits + labels + colorbar
        ax.set_xlim(5.5, -5.5)
        ax.set_ylim(-5.5, 5.5)
        ax.set_xlabel(r"$\Delta$ RA [arcsec]")
        ax.set_ylabel(r"$\Delta$ DEC [arcsec]")
        fig.colorbar(im, orientation='vertical', ticklocation='right', label = r'$\mathrm{T_{b}}$ [K]', extend = 'both')

        # Add beam
        beam_plot = Ellipse((4.0, -4.0),
                        self.header['BMAJ']*3600., self.header['BMIN']*3600., 90.-self.header['BPA'])
        beam_plot.set_facecolor('w')
        ax.add_artist(beam_plot)

        plt.show()

    def plot_interactive(self, cmap = 'magma', wrt_source = False, source_velocity = None):
        """
        Plot channels with slider (DS9 style).
        """
        im = self.data
        interact(lambda z: self.plot_slice(z, cmap = cmap, wrt_source = wrt_source, source_velocity = source_velocity), z=(0, im.shape[0] - 1))


    def plot_channel_map(self, source_velocity = None, wrt_source = False,  xlims = np.array([5.5, -5.5]), ylims = np.array([-5.5, 5.5]), cm = 'magma', center_channels = False, n_channels = 25, nrows = 5, ncols = 5):
        """
        Create a channel map
        """

        # Load the data
        I = self.data
        hd = self.header

        # Get indices
        if center_channels:
            midpoint = I.shape[0] // 2  # Integer division to find the central index
            half_length = 12  # Half the total number of indices we want minus one (because we also include the midpoint)

            start_index = midpoint - half_length
            end_index = midpoint + half_length + 1  # +1 because Python slices are exclusive at the upper bound

            indices = np.arange(start_index, end_index, dtype=int)
        
        else:
            indices = np.linspace(0, I.shape[0] - 1, n_channels, dtype=int)

        # Make the figure
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (nrows * 5, ncols * 5))

        # Flatten the axs array for easier indexing
        axs = axs.flatten()

        plt.subplots_adjust(hspace = 0.02, wspace = 0.02)
        plt.rcParams['font.size'] = 25
        plt.rcParams['axes.labelsize'] = 25   # Axis label font size
        plt.rcParams['xtick.labelsize'] = 25  # X-tick font size
        plt.rcParams['ytick.labelsize'] = 25  # Y-tick font size

        midpoint = indices.shape[0] // 2
        midpoint_index = indices[midpoint]
        Tb_mid = self.get_brightness_temperature(midpoint_index)
        vmin, vmax = np.nanpercentile(Tb_mid, [0.001, 99.999])

        # Go through the array of indices
        for i, index in enumerate(indices):

            ax = axs[i]  # Use the i-th subplot
            
            # Compute frequency
            nu = hd['CRVAL3'] + index * hd['CDELT3']  

            v = const.c.to(u.km/u.s).value * (1 - nu / self.rest_freq)

            Tb = self.get_brightness_temperature(index)
            
            ##### PLOT #####
            
            im = ax.imshow(Tb, origin='lower', extent = self.parameters.extent, cmap = cm, aspect = 'auto', vmin = vmin, vmax = vmax)

            # Add velocity
            if wrt_source:
                ax.text(0.5, 0.05,  f'v = {np.round(v - source_velocity, 2)} km/s', transform=ax.transAxes, ha='center', va='bottom', fontsize=20, color = 'white')

            else:
                ax.text(0.5, 0.05,  f'v = {np.round(v, 2)} km/s', transform=ax.transAxes, ha='center', va='bottom', fontsize=20, color = 'white')

            # Set limits + labels 
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

            # Set the ticks white for all plots
            ax.tick_params(color='white', which='both')
            
            # Hide tick labels for all
            ax.tick_params(labelbottom=False, labelleft=False)

            ##### ADD BEAM #####

            # Calculate bottom left position with some offset
            x_position = xlims[0] + 0.1 * (xlims[1] - xlims[0])
            y_position = ylims[0] + 0.1 * (ylims[1] - ylims[0])

            # Place beam in bottom left corner
            beam_plot = Ellipse((x_position, y_position),
                            hd['BMAJ']*3600., hd['BMIN']*3600., 90.-hd['BPA'])
            beam_plot.set_facecolor('w')
            ax.add_artist(beam_plot)

        # Reference the bottom-left plot dynamically after flattening
        ax_bottom_left = axs[(nrows-1) * ncols]

        # Show tick labels for only the bottom-left subplot
        ax_bottom_left.tick_params(labelbottom=True, labelleft=True)

        # Add x and y labels to the bottom-left plot
        ax_bottom_left.set_xlabel(r"$\Delta$ RA [arcsec]")
        ax_bottom_left.set_ylabel(r"$\Delta$ DEC [arcsec]")

        ##### COLORBAR #####
        # Compute the position for the colorbar axes
        left = 0.92   # adjust this value to move the colorbar left/right
        bottom = 0.1  # adjust this value to move the colorbar up/down
        width = 0.02
        height = 0.32  # this height corresponds to two panels up, considering hspace

        cbar_ax = fig.add_axes([left, bottom, width, height])
        fig.colorbar(im, cax=cbar_ax, orientation='vertical', label = r'$\mathrm{T_{b}}$ [K]', extend = 'both')

        

        plt.show()


    def get_brightness_temperature(self, index):
        """
        Get the brightness temperature 
        """
        I = self.data
        hd = self.header

        # Select the data for this channel
        channel  = I[index,:,:] # Selecting the middle channel
        
        # Compute frequency
        nu = hd['CRVAL3'] + index * hd['CDELT3']

        # Compute temperature
        Intensity = (channel * 1e-26 / self.parameters.beam)

        c = 299792458.0
        k = 1.380649e-23

        # Compute brightness temperature
        Tb = ((Intensity * c**2) / (2 * (nu)**2 * k))

        return Tb
             

    @staticmethod
    def _get_parameters(hd):
            """
            Extract the necessary parameters from the data, such as ra, dec, beam size, and fwhm
            """

            # Compute RA & DEC in arcseconds
            ra_offset = 3600. * hd['CDELT1'] * (np.arange(hd['NAXIS1'])-(hd['CRPIX1']-1))
            dec_offset = 3600. * hd['CDELT2'] * (np.arange(hd['NAXIS2'])-(hd['CRPIX2']-1))

            # Set the plot extent
            extent = (np.max(ra_offset), np.min(ra_offset), np.min(dec_offset), np.max(dec_offset))
            beam = (np.pi/180.)**2 * np.pi * hd['BMAJ'] * hd['BMIN'] / (4.*np.log(2.))
            FWHM = 3600 * hd['BMAJ'] * hd['BMIN']

            return _ChannelParameters(
                ra_offset,
                dec_offset,
                extent,
                beam,
                FWHM
            )

    
