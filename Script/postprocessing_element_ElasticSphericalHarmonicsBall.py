import os
import pyvtk
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from netCDF4 import Dataset
import shutil
import glob

# The data structure in element-wise output is too complicated for xarray.open_mfdataset.
# Here we open the files as individual datasets and concatenate them on the variable level.
# This code is compatible with parallel netcdf build (single file output)

# load_wave_data=True:  read wave data and return numpy.ndarray
# load_wave_data=False: do not read wave data and return xarray.DataArray (use False if data is big)

def read_element_output(data_dir, load_wave_data=True):
    ################ open files ################
    # filenames
    nc_fnames = [f for f in os.listdir(data_dir) if 'axisem3d_synthetics.nc' in f]
    # print('files to open: ', nc_fnames)

    # open files
    nc_files = []
    for nc_fname in nc_fnames:
        nc_files.append(xr.open_dataset(data_dir + '/' + nc_fname))
    
    ################ variables that are the same in the datasets ################
    # read Na grid (all azimuthal dimensions)
    na_grid = nc_files[0].data_vars['list_na_grid'].values.astype(int)

    # read time
    data_time = nc_files[0].data_vars['data_time'].values
    
    
    ################ variables to be concatenated over the datasets ################
    # define empty lists of xarray.DataArray objects
    xda_list_element_na = []
    xda_list_element_coords = []
    dict_xda_list_element = {}
    dict_xda_data_wave = {}
    for nag in na_grid:
        dict_xda_list_element[nag] = []
        dict_xda_data_wave[nag] = []
    
    # loop over nc files
    for nc_file in nc_files:
        # append DataArrays
        xda_list_element_na.append(nc_file.data_vars['list_element_na'])
        xda_list_element_coords.append(nc_file.data_vars['list_element_coords'])
        for nag in na_grid:
            dict_xda_list_element[nag].append(nc_file.data_vars['list_element__NaG=%d' % nag])
            dict_xda_data_wave[nag].append(nc_file.data_vars['data_wave__NaG=%d' % nag])
            
    # concat xarray.DataArray
    xda_list_element_na = xr.concat(xda_list_element_na, dim='dim_element')
    xda_list_element_coords = xr.concat(xda_list_element_coords, dim='dim_element')
    for nag in na_grid:
        dict_xda_list_element[nag] = xr.concat(dict_xda_list_element[nag], dim='dim_element__NaG=%d' % nag)
        dict_xda_data_wave[nag] = xr.concat(dict_xda_data_wave[nag], dim='dim_element__NaG=%d' % nag)
        
    # read data to numpy.ndarray
    list_element_na = xda_list_element_na.values.astype(int)
    list_element_coords = xda_list_element_coords.values
    dict_list_element = {}
    dict_data_wave = {}
    for nag in na_grid:
        dict_list_element[nag] = dict_xda_list_element[nag].values.astype(int)
        if load_wave_data:
            dict_data_wave[nag] = dict_xda_data_wave[nag].values
        
    ############### return ################
    if load_wave_data:
        return na_grid, data_time, list_element_na, list_element_coords, dict_list_element, dict_data_wave
    else:
        return na_grid, data_time, list_element_na, list_element_coords, dict_list_element, dict_xda_data_wave

# data dir
# ModelNameList = ['model0021']
ModelNameList = []
for imodel in np.arange(2100,10000):
    ModeName = 'LatinSphericalHarmonicsElasticBall%04d' %imodel
    ModelNameList.append(ModeName)
    
# wave dimension to animation
# output_channel = 'X'
# wave_dim_X = output_channel.index('X')

output_channel = 'RTZ'
wave_dim_1 = output_channel.index('R')
wave_dim_2 = output_channel.index('T')
wave_dim_3 = output_channel.index('Z')

for ModelName in ModelNameList:
    data_dir = '../Runs/%s/output/elements/azimuthal_slices' %ModelName
    
    if not os.path.exists(data_dir):
        continue
    
    try:
        # read
        na_grid, data_time, list_element_na, list_element_coords, \
        dict_list_element, dict_data_wave = read_element_output(data_dir)
    except:
        print(ModelName, "reading error!!!!!")
        continue

    # time steps
    ntime = len(data_time)

    # phi of the slices
    phi_slices = [0.        , 0.39269908, 0.78539816, 1.17809725, 1.57079633, 1.96349541, 2.35619449, 2.74889357, 3.14159265, 3.53429174, 3.92699082, 4.3196899 , 4.71238898, 5.10508806, 5.49778714, 5.89048623]

    nslice = len(phi_slices)

    # GLL coords on elements
    nelem = list_element_coords.shape[0]
    ngll = list_element_coords.shape[1]
    # flattened coords, (s, z)
    element_coords_sz = list_element_coords.reshape((nelem * ngll), 2)

    # loop over slices
    for islice, phi in enumerate(phi_slices):
        
        # vtk mesh
        xyz = np.ndarray((nelem * ngll, 3))
        xyz[:, 0] = element_coords_sz[:, 0] * np.cos(phi)
        xyz[:, 1] = element_coords_sz[:, 0] * np.sin(phi)
        xyz[:, 2] = element_coords_sz[:, 1]

        # loop over elements to read wave data
        # wave_s = np.ndarray((nelem * ngll, ntime))
        # wave_p = np.ndarray((nelem * ngll, ntime))
        # wave_z = np.ndarray((nelem * ngll, ntime))
        wave_disp = np.ndarray((nelem * ngll, 3, ntime))

        # wave_X = np.ndarray((nelem * ngll, ntime))

        # # check singlar value 
        # if wave_X.max() > 100:
        #     print(wave_X.max(), '%s Singular vale!!!!!' %ModelName)
        #     print(ModelName, phi)
        #     continue


        for ielem in np.arange(nelem):
            wave_disp[(ielem * ngll):(ielem * ngll + ngll), 0, :] = dict_data_wave[nslice][ielem, islice, :, 0, :]
            wave_disp[(ielem * ngll):(ielem * ngll + ngll), 1, :] = dict_data_wave[nslice][ielem, islice, :, 1, :]
            wave_disp[(ielem * ngll):(ielem * ngll + ngll), 2, :] = dict_data_wave[nslice][ielem, islice, :, 2, :]
        
        # loop over time to write netcdf
        for itime in np.arange(ntime):

            # if itime<50 or itime>55:
            #     continue

            # make slice for phi
            NETCDFDir = data_dir + '/netcdf_slices/time%d' % itime
            os.makedirs(NETCDFDir, exist_ok=True)

            nc = Dataset(NETCDFDir+'/disp_slice%d.nc' %islice, 'w')
            nc.createDimension('npoint', size=len(xyz))
            nc.createDimension('d3', size=3)
            # nc.createDimension('ntime', size=ntime)

            nc.createVariable('element_coords_cartesian', float, dimensions=('npoint','d3'))
            nc['element_coords_cartesian'][:,:] = xyz[:,:]

            # nc.createVariable('time', float, dimensions=('ntime'))
            # nc['time'][:] = data_time[:]

            nc.createVariable('disp', float, dimensions=('npoint','d3'))
            nc['disp'][:,:] = wave_disp[:,:,itime]

            nc.close()

            # # make slice for disp
            # NETCDFDir = data_dir + '/netcdf/snapshot%d' % itime
            # os.makedirs(NETCDFDir, exist_ok=True)

            # nc = Dataset(NETCDFDir+'/disp_slice%d.nc' %islice, 'w')
            # nc.createDimension('npoint', size=len(xyz))
            # nc.createDimension('3D', size=3)

            # nc.createVariable('x', float, dimensions=('npoint'))
            # nc['x'][:] = xyz[:,0]
            # nc.createVariable('y', float, dimensions=('npoint'))
            # nc['y'][:] = xyz[:,1]
            # nc.createVariable('z', float, dimensions=('npoint'))
            # nc['z'][:] = xyz[:,2]
            # nc.createVariable('time', float, dimensions=('npoint'))
            # nc['time'][:] = np.ones(len(xyz))*data_time[itime]

            # # convert spz to xyz coordinate frame by formula 3.9 Leng thesis page 28
            # nc.createVariable('disp_x', float, dimensions=('npoint'))
            # nc['disp_x'][:] = wave_s[:,itime] * np.cos(phi) - wave_p[:,itime] * np.sin(phi)
            # nc.createVariable('disp_y', float, dimensions=('npoint'))
            # nc['disp_y'][:] = wave_s[:,itime] * np.sin(phi) + wave_p[:,itime] * np.cos(phi)
            # nc.createVariable('disp_z', float, dimensions=('npoint'))
            # nc['disp_z'][:] = wave_z[:,itime]

            # nc.close()

            print('Done time step %d / %d' % (itime + 1, ntime), end='\r')
        print('\nDone slice %d / %d' % (islice + 1, len(phi_slices)))

    # # Check Repeated Dataset
    # if os.path.exists('../DataSet/%s/' %ModelName):
    #     shutil.rmtree('../DataSet/%s/' %ModelName)

    # Initial Model Folder
    os.makedirs('../DataSet/%s/' %ModelName, exist_ok=True)

    # Check 3D model nc file
    NCFilePath = glob.glob('../Runs/%s/input/*.nc' %ModelName)
    if len(NCFilePath) > 0:
        for path in NCFilePath:
            shutil.copy(path, '../DataSet/%s/' %ModelName)
    # Move Spherical Harmonics Paramters
    SphericalHarmonicsPath = glob.glob('../Runs/%s/Spherical_Harmonics.pkl' %ModelName)[0]
    shutil.copy(SphericalHarmonicsPath, '../DataSet/%s/' %ModelName)

    target_dir = '../DataSet/%s/snapshot/' %ModelName
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(data_dir + '/netcdf_slices/', target_dir)

    print(data_dir + '/netcdf_slices/', '../DataSet/%s/snapshot' %ModelName)