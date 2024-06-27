import numpy as np
from netCDF4 import Dataset
import h5py
import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
import time


def save_seis_hdf5():
    '''
    Transforming seismogram nc files and harmonics pkl files to a single HDF5 file
    '''
    path = '../DataSet/10000LatinSphericalHarmonicsElasticBall'
    model_name = "LatinSphericalHarmonicsElasticBall"
    hdf5_file_path = f"{path}/seismogram_data.h5"
    num_models = 10000  # Total number of models

    disp_dims = (num_models, 37, 37, 3, 150)
    station_coords_cartesian_dims = (num_models, 37, 37, 3)
    station_coords_spherical_dims = (num_models, 37, 37, 3)
    time_dims = (num_models, 150)
    harmonics_dims = (num_models, 1215)

    with h5py.File(hdf5_file_path, "w") as data_hdf5:
        # Preallocate datasets for each variable
        disp_data = data_hdf5.create_dataset('disp', disp_dims, dtype='float32')
        station_coords_cartesian_data = data_hdf5.create_dataset("station_coords_cartesian", station_coords_cartesian_dims, dtype='float32')
        station_coords_spherical_data = data_hdf5.create_dataset("station_coords_spherical", station_coords_spherical_dims, dtype='float32')
        time_data = data_hdf5.create_dataset("time", time_dims, dtype='float32')
        harmonics_data = data_hdf5.create_dataset("harmonics", harmonics_dims, dtype='float32')
        
        for model_id in range(0, num_models):
            print(f"Processing model {model_id}")
            
            # Define paths for seismogram and harmonics
            seis_nc_path = f"{path}/{model_name}{model_id:0>4d}/seismogram_displacement_SYN.nc"
            harmonics_path = f"{path}/{model_name}{model_id:0>4d}/Spherical_Harmonics.pkl"

            # Attempt to read files
            try:
                seismogram_nc = Dataset(seis_nc_path, "r")
                harmonics_pkl = pickle.load(open(harmonics_path, "rb"))
            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
                continue

            disp_data[model_id, :, :, :, :] = np.array(seismogram_nc['disp'])
            station_coords_cartesian_data[model_id, :, :, :] = np.array(seismogram_nc['station_coords_cartesian'])
            station_coords_spherical_data[model_id, :, :, :] = np.array(seismogram_nc['station_coords_spherical'])
            time_data[model_id, :] = np.array(seismogram_nc['time'])
            harmonics_data[model_id, :] = np.array(harmonics_pkl['Value'])
            
            # Close the NetCDF file
            seismogram_nc.close()
            
        print("Finished processing all models.")

def save_wffourier_hdf5():
    '''
    Transforming wavefield nc files and harmonics pkl files to a single HDF5 file
    '''
    path = '../DataSet/10000LatinSphericalHarmonicsElasticBall'
    model_name = "LatinSphericalHarmonicsElasticBall"
    hdf5_file_path = f"{path}/wf_fourier_data.h5"
    num_models = 10000  # Total number of models

    element_coords_sz_dims = (3648, 2)
    disp_coef_dims = (num_models, 15, 3648, 16, 3)
    harmonics_dims = (num_models, 1215)

    with h5py.File(hdf5_file_path, "w") as data_hdf5:
        # Preallocate datasets for each variable
        dtype = 'float32'
        element_coords_sz_data = data_hdf5.create_dataset("element_coords_sz", element_coords_sz_dims, dtype='float64')
        disp_coef_data = data_hdf5.create_dataset("disp_coef", disp_coef_dims, dtype=dtype)
        harmonics_data = data_hdf5.create_dataset("harmonics", harmonics_dims, dtype=dtype)
        
        for model_id in range(0, num_models):
            print(f"Processing model {model_id}")
            for snapshot_id in range(0, 15):
                # Define paths for seismogram and harmonics
                wf_nc_path = f"{path}/{model_name}{model_id:0>4d}/snapshot_coeff/disp_coef_time{snapshot_id}.nc"
                # Attempt to read files
                try:
                    wf_nc = Dataset(wf_nc_path, "r")
                except FileNotFoundError as e:
                    print(f"File not found: {e.filename}")
                    continue
                coef = np.array(wf_nc['disp_coef']).transpose(1,0,2)
                coords = np.array(wf_nc['element_coords_sz'])
                if model_id == 0 and snapshot_id == 0:
                    coords0 = coords
                    dist = np.sqrt(np.sum(coords0**2, axis=1))
                    idx = np.argsort(dist)
                    coords0 = coords0[idx]
                    coef = coef[idx,:,:]
                    element_coords_sz_data[:, :] = coords0
                    # if snapshot_id == 10:
                    #     colors = plt.cm.viridis(coef[:, 0]*1e5)
                    #     plt.figure()
                    #     plt.scatter(x=coords0[:,0], y=coords0[:,1], c=colors, cmap='viridis')
                    #     plt.savefig(f'wf_{model_id}_{snapshot_id}.png')
                else:
                    mapping = {tuple(point): index for index, point in enumerate(coords)}
                    indices = [mapping[tuple(point)] for point in coords0]
                    coords = coords[indices]
                    assert (coords == coords0).all()
                    coef = coef[indices,:,:]
                    # if snapshot_id == 10:
                    #     colors = plt.cm.viridis(coef[:, 0]*1e5)
                    #     plt.figure()
                    #     plt.scatter(x=coords[:,0], y=coords[:,1], c=colors, cmap='viridis')
                    #     plt.savefig(f'wf_{model_id}_{snapshot_id}.png')

                disp_coef_data[model_id, snapshot_id, :, :, :] = coef

                # Close the NetCDF file
                wf_nc.close()

            harmonics_path = f"{path}/{model_name}{model_id:0>4d}/Spherical_Harmonics.pkl"
            try:
                harmonics_pkl = pickle.load(open(harmonics_path, "rb"))
            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
                continue
            harmonics_data[model_id, :] = np.array(harmonics_pkl['Value'])
        print("Finished processing all models.")

def save_wfslice_hdf5():
    '''
    Transforming wavefield nc files and harmonics pkl files to a single HDF5 file
    '''
    path = '../DataSet/10000LatinSphericalHarmonicsElasticBall'
    model_name = "LatinSphericalHarmonicsElasticBall"
    hdf5_file_path = f"{path}/wf_slice_data.h5"
    num_models = 10000  # Total number osf models

    element_coords_cartesian_dims = (16, 3648, 3)
    disp_dims = (num_models, 15, 16, 3648, 3) # model_id, snapshot_id, slice_id, element_id, component_id
    harmonics_dims = (num_models, 1215)

    with h5py.File(hdf5_file_path, "w") as data_hdf5:
        # Preallocate datasets for each variable
        dtype = 'float32'
        element_coords_cartesian_data = data_hdf5.create_dataset("element_coords_cartesian", element_coords_cartesian_dims, dtype='float64')
        disp_data = data_hdf5.create_dataset("disp", disp_dims, dtype=dtype)
        harmonics_data = data_hdf5.create_dataset("harmonics", harmonics_dims, dtype=dtype)
        for model_id in range(0, num_models):
            print(f"Processing model {model_id}")
            for snapshot_id in range(0, 15):
                for slice_id in range(0, 16):
                    # Define paths for seismogram and harmonics
                    wf_nc_path = f"{path}/{model_name}{model_id:0>4d}/snapshot/time{snapshot_id}/disp_slice{slice_id}.nc"
                    # Attempt to read files
                    try:
                        wf_nc = Dataset(wf_nc_path, "r")
                    except FileNotFoundError as e:
                        print(f"File not found: {e.filename}")
                        continue
                    disp = np.array(wf_nc['disp'])
                    coords = np.array(wf_nc['element_coords_cartesian'])
                    if model_id == 0 and snapshot_id == 0:
                        dist = np.sqrt(np.sum(coords**2, axis=1))
                        idx = np.argsort(dist)
                        coords = coords[idx]
                        element_coords_cartesian_data[slice_id, :, :] = coords
                        disp = disp[idx]
                    else:
                        # print(element_coords_cartesian_data[slice_id, :, :], coords)
                        mapping = {tuple(point): index for index, point in enumerate(coords)}
                        indices = [mapping[tuple(point)] for point in element_coords_cartesian_data[slice_id, :, :]]
                        assert (coords[indices] == element_coords_cartesian_data[slice_id, :, :]).all()
                        disp = disp[indices]

                    disp_data[model_id, snapshot_id, slice_id, :] = disp

                    # Close the NetCDF file
                    wf_nc.close()
                    # import pdb; pdb.set_trace()
                    
            harmonics_path = f"{path}/{model_name}{model_id:0>4d}/Spherical_Harmonics.pkl"
            try:
                harmonics_pkl = pickle.load(open(harmonics_path, "rb"))
            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
                continue
            harmonics_data[model_id, :] = np.array(harmonics_pkl['Value'])
        print("Finished processing all models.")
        
def process_model(model_id, path, model_name, ref_coords):
    print(f"Processing model {model_id}")
    results = []
    for snapshot_id in range(15):  # Assuming 15 snapshots
        for slice_id in range(16):  # Assuming 16 slices
            wf_nc_path = f"{path}/{model_name}{model_id:0>4d}/snapshot/time{snapshot_id}/disp_slice{slice_id}.nc"
            try:
                # open files with with open
                wf_nc = Dataset(wf_nc_path, "r")
                disp = np.array(wf_nc['disp'])
                coords = np.array(wf_nc['element_coords_cartesian'])

                mapping = {tuple(point): index for index, point in enumerate(coords)}
                indices = [mapping[tuple(point)] for point in ref_coords[slice_id, :, :]]
                assert (coords[indices] == ref_coords[slice_id, :, :]).all()

                results.append((model_id, snapshot_id, slice_id, disp[indices]))
                wf_nc.close()
            except FileNotFoundError:
                continue

    harmonics_path = f"{path}/{model_name}{model_id:0>4d}/Spherical_Harmonics.pkl"
    try:
        harmonics = pickle.load(open(harmonics_path, "rb"))['Value']
        results.append((model_id, harmonics))
    except FileNotFoundError:
        pass
    return results

def main():
    t1 = time.time()
    path = '../DataSet/10000LatinSphericalHarmonicsElasticBall'
    model_name = "LatinSphericalHarmonicsElasticBall"
    hdf5_file_path = f"{path}/wf_slice_data_parallel_8K_10K.h5"
    s_model = 8000  # Total number of models
    e_model = 10000
    num_models = e_model - s_model
    cpu_num = 20
    ref_coords = np.zeros((16, 3648, 3))
    for slice_id in range(16):
        wf_nc_path_0 = f"{path}/{model_name}{0:0>4d}/snapshot/time0/disp_slice{slice_id}.nc"
        wf_nc_0 = Dataset(wf_nc_path_0, "r")
        coords0 = np.array(wf_nc_0['element_coords_cartesian'])
        dist = np.sqrt(np.sum(coords0**2, axis=1))
        idx = np.argsort(dist)
        coords0 = coords0[idx]
        ref_coords[slice_id] = coords0
        wf_nc_0.close()

    with h5py.File(hdf5_file_path, "w") as data_hdf5:
        element_coords_cartesian_dims = (16, 3648, 3)
        disp_dims = (num_models, 15, 16, 3648, 3)  # Adjust dimensions appropriately
        harmonics_dims = (num_models, 1215)
        element_coords_cartesian_data = data_hdf5.create_dataset("element_coords_cartesian", element_coords_cartesian_dims, dtype='float64')
        disp_data = data_hdf5.create_dataset("disp", disp_dims, dtype='float32')
        harmonics_data = data_hdf5.create_dataset("harmonics", harmonics_dims, dtype='float32')

        pool = mp.Pool(processes=cpu_num)
        results = pool.starmap(process_model, [(i, path, model_name, ref_coords) for i in range(s_model, e_model)])
        pool.close()
        pool.join()
        t2 = time.time()
        print(f"Processing time {t2-t1}.")
        # import pdb; pdb.set_trace()
        for result in results:
            for data in result:
                if len(data) == 4:
                    model_id, snapshot_id, slice_id, disp = data
                    disp_data[model_id-s_model, snapshot_id, slice_id, :] = disp
                elif len(data) == 2:
                    model_id, harmonics = data
                    harmonics_data[model_id-s_model, :] = harmonics
        element_coords_cartesian_data[:, :, :] = ref_coords
        t3 = time.time()
        print(f"Saving time {t3-t2}.")


def compare_loading_speed():
    '''
    Compare original loading speed and HDF5 loading speed
    '''
    #Define the paths and names
    path = '../data/Others/10000LatinSphericalHarmonicsAcousticBall/LatinSphericalHarmonicsAcousticBall'
    model_name = "LatinSphericalHarmonicsAcousticBall"

    # Start the timer
    start_time_nc = time.time()

    for model_id in range(100):  # Assuming 100 models from 0 to 99
        seis_nc_path = f"{path}/{model_name}{model_id:0>4d}/seismogram_displacement_SYN.nc"
        harmonics_path = f"{path}/{model_name}{model_id:0>4d}/Spherical_Harmonics.pkl"
        velocity_path = f"{path}/{model_name}{model_id:0>4d}/degree8_random.nc"

        with Dataset(seis_nc_path, "r") as seismogram_nc:
            disp_data = seismogram_nc['disp'][:]  # Assuming you only need to read the "disp" data
        harmonics = pickle.load(open(harmonics_path, "rb"))
        h = harmonics["Value"]
    # Stop the timer
    end_time_nc = time.time()
    time_taken_nc = end_time_nc - start_time_nc
    print(f"Time taken to read 100 .nc files: {time_taken_nc} seconds.")

    #Define the path to the HDF5 file
    hdf5_file_path = '../data/Others/10000LatinSphericalHarmonicsAcousticBall/LatinSphericalHarmonicsAcousticBall/wf_data.h5'

    # Start the timer
    start_time_hdf5 = time.time()

    with h5py.File(hdf5_file_path, 'r') as hdf:
        for name in hdf.keys():
            print(name)
        # coords = hdf['element_coords_sz'][:100] / 1e3
        # import pdb; pdb.set_trace()
        for i in range(10000):
            if hdf['X_coef'][i].max() > 10 / 1e5:
                print(i)


    # Stop the timer
    end_time_hdf5 = time.time()
    time_taken_hdf5 = end_time_hdf5 - start_time_hdf5
    print(f"Time taken to read from one HDF5 file: {time_taken_hdf5} seconds.")

def check_abnormal():
    '''
    Check abnormal values in the data
    '''
    import time
    import h5py
    import numpy as np

    # Define the path to the HDF5 file
    hdf5_file_path = '../data/Others/10000LatinSphericalHarmonicsAcousticBall/LatinSphericalHarmonicsAcousticBall/data.h5'
    model_name = "LatinSphericalHarmonicsAcousticBall"
    # Start the timer
    start_time_hdf5 = time.time()

    # Initialize counters for abnormal values
    nan_count = 0
    inf_count = 0
    out_of_range_count = 0

    # Define your "normal" range for disp data
    min_normal_value = -10  # example minimum
    max_normal_value = 10   # example maximum

    with h5py.File(hdf5_file_path, 'r') as hdf:
        for name in hdf[f"{model_name}0000"].keys():
            print(name)
        for model_id in tqdm(range(10000)):  # Assuming 100 models from 0 to 99
            group_name = f"{model_name}{model_id:0>4d}"
            if group_name in hdf:
                hamonics_data = hdf[group_name]['harmonics'][:]
                min = 10000
                if np.abs(hamonics_data).sum() < min:
                    min = np.abs(hamonics_data).sum()
        print(min)
                # disp_data = hdf[group_name]['disp'][:,:,2,:]  # Reading "disp" data
                # # Check for NaN and infinity
                # nan_count += np.isnan(disp_data).sum()
                # inf_count += np.isinf(disp_data).sum()
                # # Check for values outside the defined "normal" range
                # out_of_range_count += np.sum((disp_data < min_normal_value) | (disp_data > max_normal_value))
                # if np.isnan(disp_data).sum() > 0:
                #     print(f"Model ID: {model_id} has NaN values.")
                # if np.isinf(disp_data).sum() > 0:
                #     print(f"Model ID: {model_id} has infinity values.")
                # if np.sum((disp_data < min_normal_value) | (disp_data > max_normal_value)) > 0:
                    # print(f"Model ID: {model_id} has values outside the normal range.")
    # Stop the timer
    end_time_hdf5 = time.time()
    time_taken_hdf5 = end_time_hdf5 - start_time_hdf5

    # Print the results
    print(f"Time taken to read from one HDF5 file: {time_taken_hdf5} seconds.")
    print(f"NaN count in disp data: {nan_count}")
    print(f"Infinity count in disp data: {inf_count}")
    print(f"Out of range values count in disp data: {out_of_range_count}")

    # except_id = [
    #     333, 547, 674, 681, 912, 1108, 1422, 1441, 1472, 1577, 1981, 2227, 2365, 2482, 2547, 2570,
    #     2675, 2863, 2963, 3348, 3491, 3613, 3800, 3926, 4080, 4301, 4333, 4363, 4423, 4596, 4671,
    #     4749, 4760, 4767, 4768, 5256, 5319, 5363, 5511, 5648, 5665, 5677, 5682, 5744, 5765, 5901,
    #     6077, 6080, 6154, 6163, 6391, 6404, 6486, 6507, 6568, 6965, 7032, 7334, 7394, 7562, 7629,
    #     7746, 7800, 8088, 8178, 8475, 8540, 8548, 9252, 9254, 9453, 9527, 9701, 9887
    # ]
    # snapshot_id = 2
    # for model_id in except_id:
    #     snapshot_coef = Dataset(f'../data/Others/10000LatinSphericalHarmonicsAcousticBall/LatinSphericalHarmonicsAcousticBall/LatinSphericalHarmonicsAcousticBall{model_id:04d}/snapshot_coeff/phi_coef_time{snapshot_id}.nc', 'r')
    #     degree8_random = Dataset(f'../data/Others/10000LatinSphericalHarmonicsAcousticBall/LatinSphericalHarmonicsAcousticBall/LatinSphericalHarmonicsAcousticBall{model_id:04d}/degree8_random.nc', 'r')
    #     print(snapshot_coef['X_coef'][:].max(), snapshot_coef['X_coef'][:].min())
    #     if model_id in except_id:
    #         new_snapshot_coef = Dataset(f'../data/Others/ERRORRunsAcoustic_Nrcoeff/LatinSphericalHarmonicsAcousticBall{model_id:04d}/snapshot_coeff/phi_coef_time{snapshot_id}.nc', 'r')
    #         new_degree8_random = Dataset(f'../data/Others/ERRORRunsAcoustic_Nrcoeff/LatinSphericalHarmonicsAcousticBall{model_id:04d}/degree8_random.nc', 'r')
    #         # import pdb; pdb.set_trace()
    #         print(new_snapshot_coef['X_coef'][:].max(), new_snapshot_coef['X_coef'][:].min())

def spherical_to_cylindrical(latitude, longitude, depth, R):
    phi = np.radians(latitude)
    theta = np.radians(longitude)
    
    r = (R - depth) * np.cos(phi)
    z = (R - depth) * np.sin(phi)
    
    return r, theta, z

def save_surface():
    raise NotImplementedError

def save_interior():
    '''
    Sample N (x z t m) for different models h and train PDE
    x range: -1~1, z range: -1~1, t range: 0-3, m range: 0-0.1
    Input: h sample_num
    Output: h, x, z, t, m
    '''
    path = '../data/Others/10000LatinSphericalHarmonicsAcousticBall/LatinSphericalHarmonicsAcousticBall'
    hdf5_file_path = f"{path}/interior_data.h5"
    model_num = 100
    model_name = "LatinSphericalHarmonicsAcousticBall"
    h_dims = (model_num, 405)
    x_dims = (181*5)
    theta_dims = (361)
    z_dims = (181*5)
    t_dims = (150)
    m_dims = (model_num, 361*181*5)
    with h5py.File(hdf5_file_path, 'w') as hdf:
        h_data = hdf.create_dataset('h', h_dims, dtype='float32')
        x_data = hdf.create_dataset('x', x_dims, dtype='float32')
        theta_data = hdf.create_dataset('theta', theta_dims, dtype='float32')
        z_data = hdf.create_dataset('z', z_dims, dtype='float32')
        t_data = hdf.create_dataset('t', t_dims, dtype='float32')
        m_data = hdf.create_dataset('m', m_dims, dtype='float32')
        for model_id in range(model_num):
            print(f"Processing model {model_id}")
            try:
                harmonics = pickle.load(open(path + f"/{model_name}{model_id:0>4d}/Spherical_Harmonics.pkl", "rb"))
                model_nc = Dataset(path + f"/{model_name}{model_id:0>4d}/degree8_random.nc", "r")
            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
                continue
            h = np.array(harmonics["Value"]).reshape(-1)
            m = model_nc["dvp"][:,:,:].reshape(-1)
            h_data[model_id, :] = h
            m_data[model_id, :] = m
            if model_id == 0:
                xs = model_nc["latitude"][:]
                ys = model_nc["longitude"][:]
                zs = model_nc["depth"][:] / 1e3
                t = np.linspace(0, 3, 150)
                # make grid 
                xs, ys, zs = np.meshgrid(xs, ys, zs)
                xs, ys, zs = xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)
                # spherical to cylindrical
                x, theta, z = spherical_to_cylindrical(xs, ys, zs, 1)
                x_data[:] = x.reshape(361,181,5)[0].reshape(-1)  # 0~1
                theta_data[:] = theta.reshape(361,181,5)[:,0,0]  # -pi~pi
                z_data[:] = z.reshape(361,181,5)[0].reshape(-1)  # -1~1
                t_data[:] = t
            model_nc.close()

def spherical_to_cartesian(latitude, longitude, depth, R):
    latitude_rad = np.radians(latitude)
    longitude_rad = np.radians(longitude)

    r = R - depth

    x = r * np.cos(latitude_rad) * np.cos(longitude_rad)
    y = r * np.cos(latitude_rad) * np.sin(longitude_rad)
    z = r * np.sin(latitude_rad)
    
    return x, y, z

def save_interior_slice():
    '''
    Sample N (x z t m) for different models h and train PDE
    x range: -1~1, z range: -1~1, t range: 0-3, m range: 0-0.1
    Input: h sample_num
    Output: h, x, z, t, m
    '''
    path = '../data/Others/10000LatinSphericalHarmonicsAcousticBall/LatinSphericalHarmonicsAcousticBall'
    hdf5_file_path = f"{path}/interior_slice_data.h5"
    model_num = 100
    model_name = "LatinSphericalHarmonicsAcousticBall"
    h_dims = (model_num, 405)
    x_dims = (181*361*5)
    y_dims = (181*361*5)
    z_dims = (181*361*5)
    t_dims = (150)
    m_dims = (model_num, 361*181*5)
    with h5py.File(hdf5_file_path, 'w') as hdf:
        h_data = hdf.create_dataset('h', h_dims, dtype='float32')
        x_data = hdf.create_dataset('x', x_dims, dtype='float32')
        y_data = hdf.create_dataset('y', y_dims, dtype='float32')
        z_data = hdf.create_dataset('z', z_dims, dtype='float32')
        t_data = hdf.create_dataset('t', t_dims, dtype='float32')
        m_data = hdf.create_dataset('m', m_dims, dtype='float32')
        for model_id in range(model_num):
            print(f"Processing model {model_id}")
            try:
                harmonics = pickle.load(open(path + f"/{model_name}{model_id:0>4d}/Spherical_Harmonics.pkl", "rb"))
                model_nc = Dataset(path + f"/{model_name}{model_id:0>4d}/degree8_random.nc", "r")
            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
                continue
            h = np.array(harmonics["Value"]).reshape(-1)
            m = model_nc["dvp"][:,:,:].reshape(-1)
            h_data[model_id, :] = h
            m_data[model_id, :] = m
            if model_id == 0:
                xs = model_nc["latitude"][:]
                ys = model_nc["longitude"][:]
                zs = model_nc["depth"][:] / 1e3
                t = np.linspace(0, 3, 150)
                # make grid 
                xs, ys, zs = np.meshgrid(xs, ys, zs)
                xs, ys, zs = xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)
                # spherical to cylindrical
                x, y, z = spherical_to_cartesian(xs, ys, zs, 1)
                x_data[:] = x  # -1~1
                y_data[:] = y  # -1~1
                z_data[:] = z  # -1~1
                t_data[:] = t

            model_nc.close()

        
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
    # save_seis_hdf5()
    # save_wffourier_hdf5()
    # compare_loading_speed()
    # check_abnormal()
    # save_interior()
    # save_interior_slice()
    # save_wfslice_hdf5()
    main()
    # dir = '../data/Others/10000LatinSphericalHarmonicsAcousticBall/LatinSphericalHarmonicsAcousticBall/'
    # merge_hdf5(path1=dir+'wf_slice_data_6k.h5', path2=dir+'wf_slice_data_6k_10k.h5', save_path=dir+'wf_slice_data_8k.h5')

        


