
##############################################################################
''' bunch of miscellaneous functions that I didn't know where to put'''

import numpy as np
from bisect import bisect_left
import time
from scipy.interpolate import RectBivariateSpline

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lightbeam import LPmodes

##############################################################################
def getslices(bounds, arr):
    '''
    given a range, get the idxs corresponding to that range in the sorted 
    array arr 
    '''

    if len(bounds)==0:
        return np.s_[0:0]
    
    elif len(bounds)==1:
        return np.s_[bisect_left(arr, bounds[0])]
    
    elif len(bounds)==2:
        return np.s_[bisect_left(arr, bounds[0]):bisect_left(arr, bounds[1])+1]
    
    else:
        raise Exception("malformed bounds input in getslices(); check savex,savey,savez in config.py")


##############################################################################
def resize2(image, newshape):
    '''
    another resampling function that uses scipy, not cv2
    '''

    xpix = np.arange(image.shape[0])
    ypix = np.arange(image.shape[1])

    xpix_new = np.linspace(xpix[0], xpix[-1], newshape[0])
    ypix_new = np.linspace(ypix[0], ypix[-1], newshape[1])

    return RectBivariateSpline(xpix, ypix, image)(xpix_new, ypix_new)


##############################################################################
def overlap(u1, u2, weight=1,
            c=False):
    
    if not c:
        return weight * np.abs(np.sum(np.conj(u1)*u2))
    
    return weight * np.sum(np.conj(u1)*u2)


##############################################################################
def overlap_nonu(u1, u2, weights):

    return np.abs(np.sum(weights * np.conj(u1) * u2))


##############################################################################
def overlap_nonu_trap(u1, u2, xa,
                      ya, c=False):

    integrand = np.conj(u1)*u2
    integral = np.traps(np.trapz(integrand, ya, axis=-1), xa)

    if c:
        return integral
    
    return np.abs(integral)


##############################################################################
def normalize(u0, weight=1, normval=1):

    norm = np.sqrt(normval / overlap(u0, u0, weight))
    u0 *= norm

    return u0


##############################################################################
def norm_nonu(u0, weights, normval=1):

    norm = np.sqrt(normval / overlap_nonu(u0, u0, weights))
    u0 *= norm

    return u0


##############################################################################
def printProgressBar(iteration, total, prefix='', 
                     suffix = '', decimals=1, length=100, 
                     fill='█', printEnd="\r"):
    """
    Pulled from 
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    
    percent = ("{0:." + str(decimals) + "f}").format(100 \
                                                     * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)

    # Print New Line on Complete
    if iteration == total: 
        print()


##############################################################################
def timeit(method):
    '''
    pulled from someone's github or something. can't find it anymore
    '''

    def timed(*args, **kw):

        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))

        else:
            print('%r  %2.4f s' % \
                  (method.__name__, (te - ts)))
            
        return result
    
    return timed


##############################################################################
def gauss(xg, yg, theta,
          phi, sigu, sigv,
          k0, x0=0, y0=0.):
    '''
    tilted gaussian beam
    '''

    u = np.cos(theta)*np.cos(phi)*(xg - x0) + np.cos(theta)*np.sin(phi)*(yg - y0)
    v = -np.sin(phi)*(xg - x0) + np.cos(phi)*(yg - y0)
    w = np.sin(theta)*np.cos(phi)*(xg - x0)  + np.sin(theta)*np.sin(phi)*(yg - y0)
    out = ( np.exp(1.j*k0*w) \
           * np.exp( -0.5*np.power(u/sigu, 2.) \
                    - 0.5*np.power(v/sigv, 2.) ) ).astype(np.complex128)
    
    return out/np.sqrt(overlap(out,out))


##############################################################################
def unweight_field(u_weighted, weights):
    """
    Convert a weighted field to an unweighted field, presercing the 
    normalised total power (adiabatic transition through waveguide)
    """

    # Unweighted field
    u_unweighted = u_weighted * np.sqrt(weights)

    return u_unweighted


##############################################################################
def read_rsoft(fname):

    arr = np.loadtxt(fname, skiprows=4).T
    reals = arr[::2]
    imags = arr[1::2]
    field = (reals + 1.j*imags).T

    return field.astype(np.complex128)


##############################################################################
def write_rsoft(fname, u0, xw,
                yw):
    '''
    save field to a file format useable by rsoft
    '''

    out = np.empty((u0.shape[0]*2, u0.shape[1]))
    reals = np.real(u0)
    imags = np.imag(u0)

    for j in range(out.shape[0]):

        if j%2==0:
            out[j] = reals[:,int(j/2)]

        else:
            out[j] = imags[:,int(j/2)]
    
    header = "/rn,a,b/nx0/ls1\n/r,qa,qb\n{} {} {} 0 OUTPUT_REAL_IMAG_3D\n{} {} {}".format(u0.shape[0],
                                                                                          -xw/2,
                                                                                          xw/2,
                                                                                          u0.shape[1],
                                                                                          -yw/2,
                                                                                          yw/2)
    
    np.savetxt(fname+".dat", out.T, header = header, 
               fmt = "%f", comments="", newline="\n")

##############################################################################
def isolate_cores(E_complex,
                  core_positions_um,
                  ds=1.0,
                  crop_size=80):
    """
    Uses the known physical core positions to isolate and centre each core.
    *** Change to make crop size a function of core radius? ***
    """
    Ny, Nx = E_complex.shape

    ## Centre positions of PL outputs
    cy_global = (Ny - 1) / 2.0
    cx_global = (Nx - 1) / 2.0

    ## um -> pixels 
    pos_pix = np.asarray(core_positions_um) / ds
    pos_pix[:, 0] += cy_global
    pos_pix[:, 1] += cx_global

    # Radius in pixels
    rad_pix = crop_size / 2

    centered_cores = []

    pos_pix_lo = (pos_pix-rad_pix).astype(int)
    pos_pix_hi = (pos_pix+rad_pix).astype(int)

    for i in range(len(pos_pix)):

        core_crop = E_complex[pos_pix_lo[i,0]:pos_pix_hi[i,0],
                              pos_pix_lo[i,1]:pos_pix_hi[i,1]]

        centered_cores.append(core_crop)

    return centered_cores

###############################################################################
def probe_field(cores, r_core_out, wl, 
                n_core, n_clad, ds):
    
    N = cores[0].shape[0]
    grid = (np.arange(N) - (N - 1)/2) * ds  # µm
    X, Y = np.meshgrid(grid, grid)

    u_lp01_probe = LPmodes.lpfield(X, Y, 
                                  0, 
                                  1, 
                                  r_core_out, 
                                  wl, n_core, 
                                  n_clad)
    
    return u_lp01_probe


##############################################################################
def overlap_weighted(u1, u2, weights):
    """
    Weighted complex inner product (overlap integral)
    """

    c_lm = np.sum(np.conj(u1) * u2 * weights)

    return c_lm


##############################################################################
def normalize_weighted(u, weights, normval=1.0):
    """
    Normalize so that inner_weighted(u,u,w) == normval (1)
    """

    p = np.real(overlap_weighted(u, u, weights))
    if p == 0:
        raise ValueError("Cannot normalize: zero weighted power.")
    
    u_norm = u * np.sqrt(normval / p)
    
    return u_norm

##############################################################################
def plot_6mode_transfer_matrix(P_lm_array, Phi_lm_array, 
                           ds, dz, ref_val, wl, 
                           r_core_ms, n_modes=6, n_cores=7, 
                           save_fig=False, f_path=''):

    mode_labels = ['LP01', 'LP02', 
                   'LP11a', 'LP11b', 
                   'LP21a', 'LP21b']

    fig, (axs1, axs2) = plt.subplots(1, 2, 
                                    figsize=(8.5, 5)) #, 
                                    #  sharey=True)


    im1 = axs1.imshow(np.transpose(P_lm_array),
                vmin=0, vmax=np.amax(P_lm_array),
            cmap='viridis') #,
            #    origin='lower')
    im2 = axs2.imshow(np.transpose(Phi_lm_array),
                # vmin=-np.pi, vmax=np.pi,
                vmin=0, vmax=2*np.pi,
            cmap='twilight_shifted') #,
            #    origin='lower')

    axs1.set_title(f'Amplitude Transfer Matrix', fontsize=13)
    axs2.set_title(f'Phase Transfer Matrix', fontsize=13)

    axs1.set_ylabel(r'$\mathrm{Output \ Core}$', 
                    fontsize=12)
    axs2.set_ylabel(r'$\mathrm{Output \ Core}$', 
                    fontsize=12)
    axs1.set_xlabel(r'$\mathrm{Excited \ Mode}$',
                    fontsize=12)
    axs2.set_xlabel(r'$\mathrm{Excited \ Mode}$',
                    fontsize=12)

    divider = make_axes_locatable(axs1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, 
                orientation='vertical',
                label="Power [W]")

    divider = make_axes_locatable(axs2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, 
                orientation='vertical',
                label="phase [rad]")

    # axs2[0].axis('off')
    # axs2[1].axis('off')

    # fig2.colorbar(axp3, ax=axs2[0]) #, shrink=0.8) #, location='bottom')

    axs1.set_xticks(ticks=np.arange(0,n_modes), 
                        labels=mode_labels[:], 
                        fontsize=11,
                        rotation=90)
    axs2.set_xticks(ticks=np.arange(0,n_modes), 
                        labels=mode_labels[:], 
                        fontsize=11,
                        rotation=90)

    axs1.set_yticks(ticks=np.arange(0,n_cores), 
                        labels=np.arange(1,n_cores+1), 
                        fontsize=11)
    axs2.set_yticks(ticks=np.arange(0,n_cores), 
                        labels=np.arange(1,n_cores+1), 
                        fontsize=11)

    plt.tight_layout()
    plt.suptitle(f'wl={wl}um, r_core_ms={r_core_ms:.3f}um, ds={ds}um, dz={dz}um, ref_val={ref_val}')
    if save_fig:
        plt.savefig(f_path, format='pdf', dpi=600)
    # plt.savefig('Figures/pl19/pl19_transfer_matrix'+f_suffix+'.png')
                # format='pdf', dpi=600)
    plt.show()