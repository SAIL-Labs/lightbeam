#%%###########################################################################
'''
Propagates 6 LP modes through a 6-core hybrid mode-selective photonic lantern
using lightbeam
'''

#%%###########################################################################
### Limit Number of Cores

import os

# limit core usage
os.environ["OMP_NUM_THREADS"] = "2"

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
print("Thread limits set.")

#%%###########################################################################
### Import Libraries and Modules

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')  # non-interactive backend required for multiprocessing
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

import lightbeam
import lightbeam.optics as optics
from lightbeam import LPmodes
from lightbeam.mesh import RectMesh3D
from lightbeam.misc import normalize, unweight_field, isolate_cores, \
    probe_field, overlap_weighted, normalize_weighted, plot_6mode_transfer_matrix
from lightbeam.prop import Prop3D


#%%###########################################################################
### Photonic Lantern Parameters -> USER INPUT

wl = 1.55 # wavelength [um]

## Length of lantern
z_len = 50000 # [um]

## Set the final cross-sectional scale
taper_ratio = 6.25

## Output Radii ##
r_core_wfs_out = 3.25 # [um]

coef_r_ms = 1.219
r_core_ms_out = coef_r_ms * r_core_wfs_out  # [um]

r_clad_out = 62.5 # [um]

core_spacing_out = 35 # [um]


## Refractive Indices ##
n_core_wfs = 1.4467895
n_core_ms = 1.4467895 #- delta_n_ms
n_clad = 1.44
n_cap = 1.4345


#%%###########################################################################
### Input Geometry and Meshing ###

## Input Radii ##
r_core_ms = r_core_ms_out/taper_ratio # central science core radius [um]
r_core_wfs = r_core_wfs_out/taper_ratio # outer AO cores radii [um]
r_clad = r_clad_out/taper_ratio # cladding radius [um]

# spacing between cores
core_spacing = core_spacing_out/taper_ratio # [um]


## Resolution ##
ds = 0.25 # [um]
dz = 2 # [um]

## Mesh ##
xw = 2*r_clad_out + 10 # [um]
yw = 2*r_clad_out + 10 # [um]

num_PML = 10 # perfectly matched layers

## Adaptive meshing refinement ratio
ref_val = 1

## Create Mesh
_mesh = RectMesh3D(xw, yw, z_len,
                   ds, dz, num_PML)

x_grid, y_grid = _mesh.xg[num_PML:-num_PML, num_PML:-num_PML], \
    _mesh.yg[num_PML:-num_PML, num_PML:-num_PML]



#%%###########################################################################
### Worker Function

def propagate_mode(i):
    """Propagate LP mode i, save a plot, and return output arrays."""
    mode = _w_modes[i]

    if mode[0] == 0:
        u0 = normalize(LPmodes.lpfield(_w_xgrid, _w_ygrid,
                                        mode[0], mode[1],
                                        _w_r_clad, _w_wl, _w_n_clad, _w_n_cap))
        ab = ''
    elif i % 2 == 1:
        u0 = normalize(LPmodes.lpfield(_w_xgrid, _w_ygrid,
                                        mode[0], mode[1],
                                        _w_r_clad, _w_wl, _w_n_clad, _w_n_cap,
                                        which=_w_modes_flip[0]))
        ab = 'a'
    else:
        u0 = normalize(LPmodes.lpfield_fixed(_w_xgrid, _w_ygrid,
                                        mode[0], mode[1],
                                        _w_r_clad, _w_wl, _w_n_clad, _w_n_cap,
                                        which=_w_modes_flip[1]))
        ab = 'b'

    print(f"Propagating mode {i}: LP{mode[0],mode[1]}{ab}", flush=True)
    prop = Prop3D(_w_wl, _w_mesh, _w_pl6, _w_n_clad)

    u_out_weighted, _, u_out_weights = prop.prop2end(u0,
                                                     ref_val=_w_ref_val,
                                                     remesh_every=50)

    u_out = unweight_field(u_out_weighted, u_out_weights)
    print(f"Mode {i} output power: {np.sum(np.abs(u_out)**2):.6f}", flush=True)

    # fig2, axs2 = plt.subplots(1, 2, sharex=True, sharey=True,
    #                            tight_layout=True, figsize=(8, 4))
    # axs2[0].set_title(f'Output Intensity: LP{mode[0], mode[1]}{ab}', fontsize=14)
    # axs2[1].set_title(f'Output Phase: LP{mode[0], mode[1]}{ab}', fontsize=14)
    # axs2[0].imshow(np.abs(u_out)**2, cmap='viridis', origin='lower')
    # axs2[1].imshow(np.angle(u_out), cmap='twilight', origin='lower')
    # plt.tight_layout()
    # plt.savefig(f'{_w_f_path}{_w_f_prefix}mode{i}_LP{mode[0]}{mode[1]}{ab}_{_w_f_suffix}.png',
    #             dpi=150, bbox_inches='tight')
    # plt.close(fig2)

    return u_out_weighted, u_out_weights, u_out



#%% Create Lantern Object

pl_6 = optics.lant6_hms(r_core_ms, 
                        r_core_wfs,
                        r_clad, 
                        n_core_ms,
                        n_core_wfs, 
                        n_clad, 
                        n_cap,
                        core_spacing,
                        z_len,
                        z_offset=0,
                        scale_func=None,
                        final_scale=taper_ratio)


#%% Calc Prop Modes for Lantern

calc_modes = False # set to True to calculate modes instead of using hardcoded list

if calc_modes:
    # unnecessary to calculate NA for lantern modes, but included for completeness
    numerical_aperture = LPmodes.get_NA(n_clad, n_cap)
    wave_number = 2*np.pi/wl
    norm_freq = LPmodes.get_V(wave_number, r_clad, n_clad,
                                n_cap)

    modes = LPmodes.get_modes(norm_freq)

#%%###########################################################################
### Initialise

## Set Modes Manually
modes = [[0, 1],
         [0, 2],
         [1, 1],
         [1, 1],
         [2, 1],
         [2, 1]]

n_modes = len(modes)

modes_flip = ['cos', 'sin']

## Number of cores
n_cores = 6

#%% File Names

f_path = '/import/roci1/nlon0790/Results/hms-pl6/'
f_prefix = 'hms-pl6c_'
f_suffix = f'wl={wl}_rms={r_core_ms}_ds={ds}_dz={dz}_rv={ref_val}_xyw={xw}_zlen={z_len}_tr={taper_ratio}'


#%%###########################################################################
### Set Worker Globals and Run Parallelised Loop

# These module-level globals are inherited by worker processes via fork.
_w_mesh      = _mesh
_w_pl6       = pl_6
_w_xgrid     = x_grid
_w_ygrid     = y_grid
_w_modes     = modes
_w_modes_flip = modes_flip
_w_wl        = wl
_w_r_clad    = r_clad
_w_n_clad    = n_clad
_w_n_cap     = n_cap
_w_ref_val   = ref_val
_w_f_path    = f_path
_w_f_prefix  = f_prefix
_w_f_suffix  = f_suffix

n_workers = n_modes  # one worker per mode; reduce if memory-constrained

time_start = time.time()

with Pool(processes=n_workers) as pool:
    results = pool.map(propagate_mode, range(n_modes))

## Time Check
time_end = time.time()
print(f"Total time taken: {time_end - time_start:.2f} seconds")

#%% Convert to Arrays (pool.map preserves order)
U_out_weighted_array = np.array([r[0] for r in results])
U_out_weights_array  = np.array([r[1] for r in results])
U_out_array          = np.array([r[2] for r in results])


#%%###########################################################################
### Save Output Electric Fields

print("Saving PL output electric fields...")

U_out_weighted_data = h5py.File(f_path+f_prefix+'U_out_weighted_array_'+f_suffix+'.h5', 'w')
U_out_weighted_data.create_dataset('U_out_weighted', data=U_out_weighted_array)
U_out_weighted_data.close()

U_out_data = h5py.File(f_path+f_prefix+'U_out_array_'+f_suffix+'.h5', 'w')
U_out_data.create_dataset('U_out', data=U_out_array)
U_out_data.close()

U_out_weights_data = h5py.File(f_path+f_prefix+'U_out_weights_array_'+f_suffix+'.h5', 'w')
U_out_weights_data.create_dataset('U_out_weights', data=U_out_weights_array)
U_out_weights_data.close()


#%%###########################################################################
### Loop Through Modes to Isloate Cores and Calculate Modal Coefficients

C_lm_array = []
P_lm_array = []
Phi_lm_array = []


## Loop Through Propagation Modes
for i in range(n_modes):

    ## Init. temporary lists
    c_lm_cores = []
    p_lm_cores = []
    phi_lm_cores = []

    core_locs_out = optics.lant6_hms.get_6port_positions(core_spacing_out)

    # Isolate core weighted fields 
    cores_weighted = isolate_cores(
        U_out_weighted_array[i, num_PML:-num_PML, num_PML:-num_PML], 
        core_locs_out,
        ds=ds, 
        crop_size=80
    )

    # Isolate core weights
    weights_cores = isolate_cores(
        U_out_weights_array[i, num_PML:-num_PML, num_PML:-num_PML], 
        core_locs_out,
        ds=ds, 
        crop_size=80
    )

    # Build a single wfs probe on the crop grid (same for each core, before normalization)
    u_lp01_probe_wfs = probe_field(cores_weighted, 
                                    r_core_wfs_out, 
                                #    r_core_2_out*(len(U_out_array[0,0,:])/xw), 
                                    wl, 
                                    n_core_wfs, 
                                    n_clad, 
                                    ds)

    # Build a single ms probe on the crop grid (same for each core, before normalization)
    u_lp01_probe_ms = probe_field(cores_weighted, 
                                    r_core_ms_out, 
                                #    r_core_2_out*(len(U_out_array[0,0,:])/xw), 
                                    wl, 
                                    n_core_ms, 
                                    n_clad, 
                                    ds)


    for j in range(n_cores):

        # Normalize probe in the core's weighted metric
        if j == 0:
            probe_j = normalize_weighted(u_lp01_probe_ms, weights_cores[j], 
                                         normval=1.0)
        else:
            probe_j = normalize_weighted(u_lp01_probe_wfs, weights_cores[j], 
                                         normval=1.0)

        # Modal coefficient (complex)
        c_lm = overlap_weighted(probe_j, cores_weighted[j], weights_cores[j])
        c_lm_cores.append(c_lm)
        p_lm_cores.append(np.abs(c_lm)**2)
        phi_lm_cores.append(np.angle(c_lm))

    # print('c_lm shape: ', np.shape(c_lm_cores))


    print("Sum of coupled powers per core (weighted):", sum(p_lm_cores))

    C_lm_array.append(c_lm_cores)
    P_lm_array.append(p_lm_cores)
    Phi_lm_array.append(phi_lm_cores)

    # print('C_lm shape: ', np.shape(C_lm_array))


## Convert to numpy arrays
C_lm_array = np.array(C_lm_array)
P_lm_array = np.array(P_lm_array)
Phi_lm_array = np.array(Phi_lm_array)
Phi_lm_array = (Phi_lm_array + np.pi)


##############################################################################
### Save Outputs

print("Saving transfer matrix elements...")

C_lm_data = h5py.File(f_path+'cores/'+f_prefix+'C_lm_array_'+f_suffix+'.h5', 'w')
C_lm_data.create_dataset('C_lm', data=C_lm_array)
C_lm_data.close()

P_lm_data = h5py.File(f_path+'cores/'+f_prefix+'P_lm_array_'+f_suffix+'.h5', 'w')
P_lm_data.create_dataset('P_lm', data=P_lm_array)
P_lm_data.close()

Phi_lm_data = h5py.File(f_path+'cores/'+f_prefix+'Phi_lm_array_'+f_suffix+'.h5', 'w')
Phi_lm_data.create_dataset('Phi_lm', data=Phi_lm_array)
Phi_lm_data.close()


#%%###########################################################################
### Plot Transfer Matrix

plot_matrix = False
save_matrix_plot = False

if plot_matrix:
        f_fig_path = 'hms-pl7_transfer_matrix'+f_suffix+'.pdf'
        plot_6mode_transfer_matrix(P_lm_array, Phi_lm_array, 
                                ds, dz, ref_val,
                                wl, r_core_ms, 
                                n_modes=n_modes, n_cores=n_cores,
                                save_fig=save_matrix_plot, f_path=f_fig_path)