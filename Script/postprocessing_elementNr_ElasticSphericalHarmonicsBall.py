import os
import pyvtk
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from netCDF4 import Dataset
import shutil
import glob
from multiprocessing import Pool

nproc = 24

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
for imodel in np.arange(4212,4300):
    ModeName = 'LatinSphericalHarmonicsElasticBall%04d' %imodel
    ModelNameList.append(ModeName)
    
# wave dimension to animation
# output_channel = 'X'
# wave_dim_X = output_channel.index('X')

output_channel = 'RTZ'
wave_dim_1 = output_channel.index('R')
wave_dim_2 = output_channel.index('T')
wave_dim_3 = output_channel.index('Z')


# for ModelName in ModelNameList:
def Process(ModelName):
    data_dir = '../Runs/%s/output/elements/full_coeff' %ModelName
    
    if not os.path.exists(data_dir):
        print('ERROR %s not found' %data_dir)
        return
    
    try:
        # read
        na_grid, data_time, list_element_na, list_element_coords, \
        dict_list_element, dict_data_wave = read_element_output(data_dir)
    except:
        print(ModelName, "reading error!!!!!")
        return

    # time steps
    ntime = len(data_time)

    # # phi of the slices
    nag = na_grid[0]

    # GLL coords on elements
    nelem = list_element_coords.shape[0]
    ngll = list_element_coords.shape[1]
    # flattened coords, (s, z)
    element_coords_sz = list_element_coords.reshape((nelem * ngll), 2)

    # # save element_coords_sz
    # element_coords_sz

    wave_disp = np.ndarray((nag, nelem * ngll, 3))

    for itime in np.arange(ntime):

        for ielem in np.arange(nelem):
            wave_disp[:, (ielem * ngll):(ielem * ngll + ngll),:] = dict_data_wave[nag][ielem, :, :, :, itime]


        # make slice for phi
        NETCDFDir = data_dir + '/netcdf_coeff'
        os.makedirs(NETCDFDir, exist_ok=True)

        nc = Dataset(NETCDFDir+'/disp_coef_time%d.nc' %itime, 'w')
        nc.createDimension('npoint', size=nelem * ngll)
        nc.createDimension('Nr_Dim', size=nag)
        nc.createDimension('d2', size=2)
        nc.createDimension('d3', size=3)
        # nc.createDimension('ntime', size=ntime)

        nc.createVariable('element_coords_sz', float, dimensions=('npoint','d2'))
        nc['element_coords_sz'][:] = element_coords_sz[:]
        # nc.createVariable('time', float, dimensions=('ntime'))
        # nc['time'][:] = data_time[:]

        nc.createVariable('disp_coef', float, dimensions=('Nr_Dim', 'npoint','d3'))
        nc['disp_coef'][:,:] = wave_disp[:,:]

        nc.close()

    # # Check Repeated Dataset
    # if os.path.exists('../DataSet/%s/' %ModelName):
    #     shutil.rmtree('../DataSet/%s/' %ModelName)

    # Initial Model Folder
    os.makedirs('../DataSet/%s/' %ModelName, exist_ok=True)

    # Make time info file
    if os.path.exists('../DataSet/%s/time_info.nc' %ModelName):
        os.remove('../DataSet/%s/time_info.nc' %ModelName)
    nc = Dataset('../DataSet/%s/time_info.nc' %ModelName, 'w')
    nc.createDimension('ntime', size=ntime)
    nc.createVariable('time', float, dimensions=('ntime'))
    nc['time'][:] = data_time[:]

    # Check 3D model nc file
    NCFilePath = glob.glob('../Runs/%s/input/*.nc' %ModelName)
    if len(NCFilePath) > 0:
        for path in NCFilePath:
            shutil.copy(path, '../DataSet/%s/' %ModelName)
    # Move Spherical Harmonics Paramters
    SphericalHarmonicsPath = glob.glob('../Runs/%s/Spherical_Harmonics.pkl' %ModelName)[0]
    shutil.copy(SphericalHarmonicsPath, '../DataSet/%s/' %ModelName)

    target_dir = '../DataSet/%s/snapshot_coeff/' %ModelName
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(data_dir + '/netcdf_coeff/', target_dir)

    print(data_dir + '/netcdf_coeff/', target_dir)

with Pool(nproc) as p:
    p.map(Process,ModelNameList)  # Multiprocessing DownloadEvent