import numpy as np
from netCDF4 import Dataset
import h5py
import time
import pickle
from tqdm import tqdm
# import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from obspy.core import Stream, Trace, UTCDateTime, Stats
import os
import xarray as xr
import sys

def read_seismogram(Run_dir):
    # Run_dir = "../Runs/LatinSphericalHarmonicsPREM0002"
    BallRadius = 1000
    nlat, nlon = 37, 37
    StartTime, EndTime = 0.0, 3.0   # note in Second
    resample_rate = 0.02    # note in Second
    lowpass_freq = 20.    # note in Hz
    PointPerTrace = int((EndTime - StartTime)/resample_rate)

    # initial wave & location array
    wave_disp = np.ndarray((nlat, nlon, 3, PointPerTrace)) #nlat, nlon, ncomponent, PointPerTrace
    station_coords_cartesian = np.ndarray((nlat, nlon, 3))
    station_coords_spherical = np.ndarray((nlat, nlon, 3))

    StationInfo = np.loadtxt(f"{Run_dir}/input/Synthetic_Stations_Ball.txt",dtype=str, skiprows=3)
    stalatlon_dict = {}
    # make station coordinate dict
    for item in StationInfo:
        stkey = item[1]+'.'+item[0]
        x = float(item[2])
        y = float(item[3])
        depth = float(item[5])
        if stkey not in stalatlon_dict.keys():
            stalatlon_dict[stkey] = []
        stalatlon_dict[stkey].append((x, y, depth))

    # read rank-station info
    rank_station_info = np.loadtxt(f"{Run_dir}/output/stations/Synthetic_Stations/rank_station.info", dtype=str, skiprows=1)
    # dict: mpi-rank -> [station keys]
    rank_station_dict = {}
    for item in rank_station_info:
        rank = item[0]
        stkey = item[1]
        inrank_index = item[2]
        # initialize with an empty array if rank does not exists in rank_station_dict
        if rank not in rank_station_dict.keys():
            rank_station_dict[rank] = []
        # append the station
        rank_station_dict[rank].append([stkey, inrank_index])

    # loop over mpi-ranks to read data
    for rank in rank_station_dict.keys():
        f = Dataset(f"{Run_dir}/output/stations/Synthetic_Stations/axisem3d_synthetics.nc.rank%s" %rank, 'r')
        time = np.array(f.variables['data_time'][:])

        for [StationName, inrank_index] in rank_station_dict[rank]:
            lat, lon = stalatlon_dict[StationName][0][0], stalatlon_dict[StationName][0][1]
            ilat, ilon = int((lat+90)/5), int((lon+180)/10)
            colat = 90 - lat
            theta = np.radians(colat)
            phi = np.radians(lon)
            stadepth = stalatlon_dict[StationName][0][2]
            stax = (BallRadius - stadepth)*np.sin(theta)*np.cos(phi)
            stay = (BallRadius - stadepth)*np.sin(theta)*np.sin(phi)
            staz = (BallRadius - stadepth)*np.cos(theta)
            station_coords_cartesian[ilat, ilon,:] = [stax,stay,staz]
            station_coords_spherical[ilat, ilon,:] = [BallRadius - stadepth,theta,phi]
            # trace header
            stats = Stats()
            stats.starttime = UTCDateTime(time[0])
            stats.delta = UTCDateTime(time[1] - time[0])
            stats.npts = len(time)
            # stream
            stream = Stream()
            for ich, ch in enumerate('RTZ'):
                stats.channel = ch  
                # default unit is km
                stream.append(Trace(f.variables['data_wave'][int(inrank_index)][ich], header=stats))
            stream.filter('lowpass', freq=lowpass_freq)
            stream.resample(1/resample_rate)
            # stream = stream.slice(UTCDateTime(int(arrivals[0].time)+StartTime), UTCDateTime(int(arrivals[0].time)+EndTime))
            stream = stream.slice(UTCDateTime(StartTime), UTCDateTime(EndTime))

            wave_disp[ilat, ilon, 0, :] = stream[0].data[0:PointPerTrace]
            wave_disp[ilat, ilon, 1, :] = stream[1].data[0:PointPerTrace]
            wave_disp[ilat, ilon, 2, :] = stream[2].data[0:PointPerTrace]
            # wave_time = np.array(stream[0].times())

        # wave_disp[ilat, ilon, :, :] = np.array(f.variables['data_wave'][int(inrank_index)][:])
        f.close()

    return wave_disp #nlat nlon, ncomponent, npoint

def save_seismogram_hdf5(StartNum,EndNum):
    '''
    Transforming seismogram nc files and harmonics pkl files to a single HDF5 file
    '''
    path = '../FinishedRuns'
    model_name = "LatinSphericalHarmonicsElasticBall"
    hdf5_file_path = f"../DataSet/seismogram_data_{StartNum:0>4d}_{EndNum:0>4d}.h5"
    num_models = EndNum-StartNum  # Total number of models

    disp_dims = (num_models, 37, 37, 3, 150)
    station_coords_cartesian_dims = (37, 37, 3) # nlat, nlon, ncomponent
    station_coords_spherical_dims = (37, 37, 3)
    time_dims = (150) # PointPerTrace
    # source_dims = (num_models, 9)
    harmonics_dims = (num_models, 1215)

    with h5py.File(hdf5_file_path, "w") as data_hdf5:
        # Preallocate datasets for each variable
        disp_data = data_hdf5.create_dataset('disp', disp_dims, dtype='float32')
        station_coords_cartesian = data_hdf5.create_dataset("station_coords_cartesian", station_coords_cartesian_dims, dtype='float32')
        station_coords_spherical = data_hdf5.create_dataset("station_coords_spherical", station_coords_spherical_dims, dtype='float32')
        time_data = data_hdf5.create_dataset("time", time_dims, dtype='float32')
        # source_data = data_hdf5.create_dataset("source", source_dims, dtype='float32')
        harmonics_data = data_hdf5.create_dataset("harmonics", harmonics_dims, dtype='float32')

        # make station coordinates
        lat = np.radians(90 - np.linspace(-90, 90, 37)) # note this is colatitude
        lon = np.radians(np.linspace(-180, 180, 37))
        LON, LAT = np.meshgrid(lon, lat)
        radius = np.ones((37,37))*1000
        station_coords_spherical[:,:,:] = np.stack([LAT,LON,radius], axis=-1)

        station_coords_cartesian[:,:,:] = np.zeros(station_coords_spherical.shape)
        station_coords_cartesian[:,:,0] = station_coords_spherical[:,:,2]*np.sin(station_coords_spherical[:,:,0])*np.cos(station_coords_spherical[:,:,1])
        station_coords_cartesian[:,:,1] = station_coords_spherical[:,:,2]*np.sin(station_coords_spherical[:,:,0])*np.sin(station_coords_spherical[:,:,1])
        station_coords_cartesian[:,:,2] = station_coords_spherical[:,:,2]*np.cos(station_coords_spherical[:,:,0])

        # make time data
        time_data[:] = np.linspace(0,2.98,150)

        for index, model_id in enumerate(range(StartNum, EndNum)):
            print(f"Processing model {model_id}")
            
            # Define paths for seismogram and harmonics
            run_path = f"{path}/{model_name}{model_id:0>4d}"
            harmonics_path = f"{path}/{model_name}{model_id:0>4d}/Spherical_Harmonics.pkl"

            # Attempt to read files
            try:
                wave_disp = read_seismogram(run_path)
                harmonics_pkl = pickle.load(open(harmonics_path, "rb"))

            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
                continue

            disp_data[index, :, :, :, :] = wave_disp
            harmonics_data[index, :] = np.array(harmonics_pkl['Value'])
            
        print("Finished processing all models.")

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

def read_fourier_coef(Run_dir):
    nelement = 3648
    fourier_dim = 16

    whole_disp_coef = np.ndarray((nelement, 16, 3, 15))

    # mantle&crust
    na_grid, data_time, list_element_na, list_element_coords, \
    dict_list_element, dict_data_wave = read_element_output(f"{Run_dir}/output/elements/full_coeff")

    disp_coef = np.squeeze(dict_data_wave[fourier_dim],axis=(2,))
    nelem, ngll = list_element_coords.shape[0], list_element_coords.shape[1]
    coords = list_element_coords.reshape((nelem * ngll), 2)
    mapping = {tuple(point): index for index, point in enumerate(coords_ref)}
    indices = [mapping[tuple(point)] for point in coords]
    coords = coords[indices]
    assert (coords == coords_ref).all()
    whole_disp_coef[:,:,:,:] = disp_coef[indices,:,:,:]

    return data_time, whole_disp_coef # nelement, nfourier, ncomponent, ntime (16470, 16, 3, 20)

def save_wffourier_hdf5(StartNum,EndNum):
    '''
    Transforming wavefield nc files and harmonics pkl files to a single HDF5 file
    '''
    global coords_ref
    path = '../FinishedRuns'
    model_name = "LatinSphericalHarmonicsElasticBall"
    hdf5_file_path = f"../DataSet/wf_fourier_data_{StartNum:0>4d}_{EndNum:0>4d}.h5"
    num_models = EndNum-StartNum  # Total number of models

    element_coords_sz_dims = (3648, 2)
    disp_coef_dims = (num_models, 3648, 16, 3, 15)
    harmonics_dims = (num_models, 1215)
    timestep_dims = (num_models, 15)

    with h5py.File(hdf5_file_path, "w") as data_hdf5:
        # Preallocate datasets for each variable
        dtype = 'float32'
        element_coords_sz_data = data_hdf5.create_dataset("element_coords_sz", element_coords_sz_dims, dtype='float64')
        disp_coef_data = data_hdf5.create_dataset("disp_coef", disp_coef_dims, dtype=dtype)
        harmonics_data = data_hdf5.create_dataset("harmonics", harmonics_dims, dtype=dtype)
        timestep_data = data_hdf5.create_dataset("timestep", timestep_dims, dtype=dtype)

        # make reference element coords
        # sort mantle&crust element (12096, 2)
        na_grid, data_time, list_element_na, list_element_coords, \
        dict_list_element, dict_data_wave = read_element_output(f"{path}/LatinSphericalHarmonicsElasticBall0000/output/elements/full_coeff")
        nelem, ngll = list_element_coords.shape[0], list_element_coords.shape[1]
        coords_ref = list_element_coords.reshape((nelem * ngll), 2)
        dist = np.sum(coords_ref**2, axis=1)
        idx = np.argsort(dist)
        coords_ref = coords_ref[idx]

        # merge into whole
        element_coords_sz_data[:,:] = coords_ref[:,:]
        
        for index, model_id in enumerate(range(StartNum, EndNum)):
            print(f"Processing model {model_id}")

            # Define paths for seismogram and harmonics
            run_path = f"{path}/{model_name}{model_id:0>4d}"
            harmonics_path = f"{path}/{model_name}{model_id:0>4d}/Spherical_Harmonics.pkl"
            # Attempt to read files
            try:
                timestep, whole_disp_coef = read_fourier_coef(run_path)
                harmonics_pkl = pickle.load(open(harmonics_path, "rb"))
            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
                continue
            timestep_data[index, :] = timestep
            disp_coef_data[index, :, :, :, :] = whole_disp_coef
            harmonics_data[index, :] = np.array(harmonics_pkl['Value'])
        print("Finished processing all models.")

def read_slice_data(Run_dir, coords_ref):
    nelement = 3648
    nslice = 16
    whole_wave_disp = np.ndarray((nelement, nslice, 3, 15))

    # read
    na_grid, data_time, list_element_na, list_element_coords, \
    dict_list_element, dict_data_wave = read_element_output(f"{Run_dir}/output/elements/azimuthal_slices")

    wave_disp = np.squeeze(dict_data_wave[nslice],axis=(2,))
    nelem, ngll = list_element_coords.shape[0], list_element_coords.shape[1]
    coords = list_element_coords.reshape((nelem * ngll), 2)
    mapping = {tuple(point): index for index, point in enumerate(coords)}
    indices = [mapping[tuple(point)] for point in coords_ref]
    # coords_ref = coords_ref[indices]
    assert (coords[indices] == coords_ref).all()
    whole_wave_disp[:,:,:,:] = wave_disp[indices,:,:,:]
    # print('mantle&crust OK')

    return data_time, whole_wave_disp # nelement, nslice, ncomponent, ntime (16470, 16, 3, 30)

def save_wfslice_hdf5(StartNum,EndNum):
    '''
    Transforming wavefield nc files and harmonics pkl files to a single HDF5 file
    '''
    path = '../FinishedRuns'
    model_name = "LatinSphericalHarmonicsElasticBall"
    hdf5_file_path = f"../DataSet/wf_slice_data_{StartNum:0>4d}_{EndNum:0>4d}.h5"
    num_models = EndNum-StartNum  # Total number osf models
    timestep_dims = (num_models, 15)

    element_coords_sz_dims = (3648, 2)
    disp_dims = (num_models, 3648, 16, 3, 15) # model_id, element_id, slice_id, component_id, time_id
    harmonics_dims = (num_models, 1215)
    slice_dims = (16)

    with h5py.File(hdf5_file_path, "w") as data_hdf5:
        # Preallocate datasets for each variable
        dtype = 'float32'
        element_coords_sz_data = data_hdf5.create_dataset("element_coords_sz", element_coords_sz_dims, dtype='float64')
        disp_data = data_hdf5.create_dataset("disp", disp_dims, dtype=dtype)
        harmonics_data = data_hdf5.create_dataset("harmonics", harmonics_dims, dtype=dtype)
        slice_data = data_hdf5.create_dataset("slice", slice_dims, dtype=dtype)
        timestep_data = data_hdf5.create_dataset("timestep", timestep_dims, dtype=dtype)

        slice_data = np.radians(np.linspace(0, 360, 17)[0:-1])

        # sort mantle&crust element (12096, 2)
        na_grid, data_time, list_element_na, list_element_coords, \
        dict_list_element, dict_data_wave = read_element_output(f"{path}/LatinSphericalHarmonicsElasticBall0000/output/elements/azimuthal_slices")
        nelem, ngll = list_element_coords.shape[0], list_element_coords.shape[1]
        coords_ref = list_element_coords.reshape((nelem * ngll), 2)
        dist = np.sum(coords_ref**2, axis=1)
        idx = np.argsort(dist)
        coords_ref = coords_ref[idx]
        element_coords_sz_data[:,:] = coords_ref[:,:]

        for index, model_id in enumerate(range(StartNum, EndNum)):
            print(f"Processing model {model_id}")
            
            run_path = f"{path}/{model_name}{model_id:0>4d}"
            harmonics_path = f"{path}/{model_name}{model_id:0>4d}/Spherical_Harmonics.pkl"
            try:
                timestep, whole_wave_disp = read_slice_data(run_path, coords_ref)
                harmonics_pkl = pickle.load(open(harmonics_path, "rb"))
            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
                continue
            timestep_data[index, :] = timestep
            disp_data[index, :, :, :, :] = whole_wave_disp
            harmonics_data[index, :] = np.array(harmonics_pkl['Value'])
        print("Finished processing all models.")
        

        
def merge_hdf5(path1, path2, save_path):
    # merge wf slice hdf5 
    file1 = h5py.File(path1, 'r')
    file2 = h5py.File(path2, 'r')
    with h5py.File(save_path, 'w') as hdf:
        for key in file1.keys():
            if key == 'element_coords_cartesian':
                hdf.create_dataset(key, data=file1[key][:])
            else:
                hdf.create_dataset(key, data=np.concatenate([file1[key][:], file2[key][:2000]], axis=0))
    file1.close()
    file2.close()


if __name__ == "__main__":
    StartNum, EndNum = int(sys.argv[1]), int(sys.argv[2])
    # save_seismogram_hdf5(StartNum,EndNum)
    # save_wffourier_hdf5(StartNum,EndNum)
    save_wfslice_hdf5(StartNum,EndNum)
    # dir = '../data/Others/10000LatinSphericalHarmonicsAcousticBall/LatinSphericalHarmonicsAcousticBall/'
    # merge_hdf5(path1=dir+'wf_slice_data_6k.h5', path2=dir+'wf_slice_data_6k_10k.h5', save_path=dir+'wf_slice_data_8k.h5')

        


