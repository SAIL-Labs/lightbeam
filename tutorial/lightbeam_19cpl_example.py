#%%###########################################################################
'''
Propagates 17 LP modes through a 19-core a photonic lantern using lightbeam

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
import matplotlib.pyplot as plt

import lightbeam
import lightbeam.optics as optics
from lightbeam import LPmodes
from lightbeam.mesh import RectMesh3D
from lightbeam.misc import normalize, unweight_field
from lightbeam.prop import Prop3D


#%%###########################################################################
### Code example: 19 port lantern ###

## Set Parameters and Variables

wl = 1.5 # wavelength [um]

#### Mesh ####
xw = 349 # [um] 
yw= 349 # [um] 
z_len = 40000 # [um]

num_PML = 10 # perfectly matched layers

ds = 0.25 # [um]
dz = 2 # [um]

## Adaptive meshing refinement ratio
ref_val = 1e-2

## Create Mesh
_mesh = RectMesh3D(xw, yw, z_len,
                   ds, dz, num_PML)

x_grid, y_grid = _mesh.xg[num_PML:-num_PML, num_PML:-num_PML], \
    _mesh.yg[num_PML:-num_PML, num_PML:-num_PML]

## Set the final cross-sectional scale
taper_ratio = 10

## Output Radii ##
r_core_1_out = 3.25 # [um]
r_core_2_out = 3.25 # [um]
r_clad_out = 164 # [um]

core_spacing_out = 60 # [um]

## Input Radii ##
r_core_1 = r_core_1_out/taper_ratio # central science core radius [um]
r_core_2 = r_core_2_out/taper_ratio # outer AO cores radii [um]
r_clad = r_clad_out/taper_ratio # cladding radius [um]

# spacing between cores
core_spacing = core_spacing_out/taper_ratio # [um]

## Refractive Indices ##
n_core = 1.4467895 
n_clad = 1.44 
n_cap = 1.4345


#%% Create Lantern Object

pl_19 = optics.lant19(r_core_2, 
                      r_clad,
                      n_core, 
                      n_clad, 
                      n_cap,
                      core_spacing, 
                      z_len, 
                      z_offset=0,
                      scale_func=None, 
                      final_scale=taper_ratio)



#%%###########################################################################
### Initialise

## Set Modes Manually
modes = [[0, 1],
          [0, 2],
          [0, 3],
          [1, 1],
          [1, 1],
          [1, 2],
          [1, 2],
          [2, 1],
          [2, 1],
          [2, 2],
          [2, 2],
          [3, 1],
          [3, 1],
          [3, 2],
          [3, 2],
          [4, 1],
          [4, 1],
          [5, 1],
          [5, 1]]


n_modes = len(modes)

modes_flip = ['cos', 'sin']

## Initialise Arrays to Store Data
u0 = []
U_out_weighted_array = []
U_out_weights_array = []
U_out_array = []

#%% File Names

f_path = 'Results/'
f_suffix = f'ds={ds}_dz={dz}_rv={ref_val}_xyw={xw}_zlen={z_len}_tr={taper_ratio}'


#%%###########################################################################
### Loop Through Modes

## Loop Through Propagation Modes
for i in range(n_modes):
    if modes[i][0] == 0:
        u0 = normalize(LPmodes.lpfield(x_grid, y_grid, 
                                        modes[i][0],
                                        modes[i][1],
                                        r_clad,
                                        wl,
                                        n_core,
                                        n_clad))
        
        ab = ''
        
    else:
        if i % 2 == 1:
            u0 = normalize(LPmodes.lpfield_fixed(x_grid, y_grid, 
                                            modes[i][0],
                                            modes[i][1],
                                            r_clad,
                                            wl,
                                            n_core,
                                            n_clad,
                                            which=modes_flip[0]))
            
            ab = 'a'

        else:
            u0 = normalize(LPmodes.lpfield_fixed(x_grid, y_grid, 
                                            modes[i][0],
                                            modes[i][1],
                                            r_clad,
                                            wl,
                                            n_core,
                                            n_clad,
                                            which=modes_flip[1]))
            
            ab = 'b'


    # print('u0 shape =', np.shape(u0))
    # print("Input power (uniform):", np.sum(np.abs(u0)**2) * ds * ds)
    
    #### propagation ####
    print("Propagating core:", i)
    prop = Prop3D(wl,       # wavelength
                _mesh,      # mesh    
                pl_19,      # optical system
                n_clad)     # cladding refractive index

    
    
    ## prop2end
    u_out_weighted, u_out_2, u_out_weights = prop.prop2end(u0,
                                                          ref_val=ref_val,
                                                          remesh_every=50)
    
    # print('u_out_shape =', np.shape(u_out_weighted))
    
    ## Calculate Unweighted Output Field
    u_out = unweight_field(u_out_weighted, u_out_weights)

    # print("Output power (uniform):", np.sum(np.abs(u_out)**2) * ds * ds)
    print("Output power (unweighted):", np.sum(np.abs(u_out_weighted)**2 \
                                             * u_out_weights))


    ## Append output to list
    U_out_weighted_array.append(u_out_weighted)
    U_out_weights_array.append(u_out_weights)
    U_out_array.append(u_out)


    ## Plot Output Fields
    [fig2, axs2] = plt.subplots(1, 2, 
                        sharex=True, 
                        sharey=True, 
                        tight_layout=True, 
                        figsize=(8, 4))
    axs2[0].set_title(f'Output Intensity: LP{modes[i][0], modes[i][1]}{ab}', 
                      fontsize=14)
    axs2[1].set_title(f'Output Phase: LP{modes[i][0], modes[i][1]}{ab}', 
                      fontsize=14)
    im2 = axs2[0].imshow(np.abs(u_out)**2,
            cmap='viridis',
            origin='lower',
            label='b)')
    im2 = axs2[1].imshow(np.angle(u_out),
            cmap='twilight',
            origin='lower',
            label='b)')
    plt.tight_layout()
    plt.show()

    plt.close(fig2)


#%% Covert to Arrays
U_out_weighted_array = np.array(U_out_weighted_array)
U_out_array = np.array(U_out_array)
U_out_weights_array = np.array(U_out_weights_array)


#%%###########################################################################
### Save Outputs

print("Saving...")


U_out_weighted_data = h5py.File(f_path+'U_out_weighted_array_'+f_suffix+'.h5', 'w')
U_out_weighted_data.create_dataset('U_out_weighted', data=U_out_weighted_array)
U_out_weighted_data.close()

U_out_data = h5py.File(f_path+'U_out_array_'+f_suffix+'.h5', 'w')
U_out_data.create_dataset('U_out', data=U_out_array)
U_out_data.close()

U_out_weights_data = h5py.File(f_path+'U_out_weights_array_'+f_suffix+'.h5', 'w')
U_out_weights_data.create_dataset('U_out_weights', data=U_out_weights_array)
U_out_weights_data.close()

