import glob
import shutil 
from netCDF4 import Dataset
import numpy as np
import os
import yaml 

from obspy.core import Stream, Trace, UTCDateTime, Stats
from obspy.core.event import read_events
import matplotlib.pyplot as plt

import time as tt
from multiprocessing import Pool

nproc = 20


ModelNameList = []
for imodel in np.arange(600,10000):
    ModelName = 'LatinSphericalHarmonicsElasticBall%04d' %imodel
    if os.path.exists('../Runs/%s/output/' %ModelName):
        ModelNameList.append(ModelName)

# wave dimension to animation
output_channel = 'RTZ'
wave_dim_1 = output_channel.index('R')
wave_dim_2 = output_channel.index('T')
wave_dim_3 = output_channel.index('Z')
# wave_dim_X = output_channel.index('X')
BallRadius = 1000

displacement_or_potential = 'displacement'
nlat = 37
nlon = 37

# for ModelName in ModelNameList:
def Process(ModelName):
    # data_dir = '../Runs/%s/output/elements/orthogonal_azimuthal_slices' %ModelName
    RunPath = '../Runs/%s' %ModelName

    NETCDFDir = '../DataSet/%s' %ModelName
    if not os.path.exists(NETCDFDir):
        os.mkdir(NETCDFDir)
    StationInfo = np.loadtxt(RunPath+'/input/Synthetic_Stations_Ball.txt',dtype=str, skiprows=3)
    stalatlon_dict = {}
    for AppliedNetwork in ['SYN']:


        try:
            for item in StationInfo:
                # filter out unnecessary newwork
                if item[1] != AppliedNetwork:
                    continue
                stkey = item[1]+'.'+item[0]
                x = float(item[2])
                y = float(item[3])
                depth = float(item[5])
                if stkey not in stalatlon_dict.keys():
                    stalatlon_dict[stkey] = []
                stalatlon_dict[stkey].append((x, y, depth))

            StartTime, EndTime = 0, 3.0    # note in Second
            resample_rate = 0.02    # note in Second
            lowpass_freq = 20    # note in Hz
            PointPerTrace = int((EndTime - StartTime)/resample_rate)

            istation = 0

            # initial wave & location array
            wave_disp = np.ndarray((nlat, nlon, 3, PointPerTrace))
            station_coords_cartesian = np.ndarray((nlat, nlon, 3))
            station_coords_spherical = np.ndarray((nlat, nlon, 3))


            GSNDir = RunPath + '/output/stations/Synthetic_Stations'

            # read rank-station info
            rank_station_info = np.loadtxt(GSNDir + '/rank_station.info', dtype=str, skiprows=1)

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
                f = Dataset(GSNDir + '/axisem3d_synthetics.nc.rank%s' %rank, 'r')
                time = f.variables['data_time'][:]


                for [StationName, inrank_index] in rank_station_dict[rank]:

                    if not StationName.startswith(AppliedNetwork):
                        continue

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
                    for ich, ch in enumerate(output_channel):
                        stats.channel = ch  
                        # default unit is km
                        stream.append(Trace(f.variables['data_wave'][int(inrank_index)][ich], header=stats))

                    stream.filter('lowpass', freq=lowpass_freq)
                    stream.resample(1/resample_rate)
                    # stream = stream.slice(UTCDateTime(int(arrivals[0].time)+StartTime), UTCDateTime(int(arrivals[0].time)+EndTime))
                    stream = stream.slice(UTCDateTime(StartTime), UTCDateTime(EndTime))
                    # npoint_persta = len(stream[0].data)

                    # # nc['time'][istation*npoint_persta:(istation+1)*npoint_persta] = stream[0].times() + int(arrivals[0].time) + StartTime
                    # nc['time'][istation*npoint_persta:(istation+1)*npoint_persta] = stream[0].times()

                    if displacement_or_potential == 'displacement':
                        wave_disp[ilat, ilon, 0, :] = stream[wave_dim_1].data
                        wave_disp[ilat, ilon, 1, :] = stream[wave_dim_2].data
                        wave_disp[ilat, ilon, 2, :] = stream[wave_dim_3].data
                        # nc['disp_x'][istation*npoint_persta:(istation+1)*npoint_persta] = stream[wave_dim_s].data * np.cos(phi) - stream[wave_dim_p].data * np.sin(phi)
                        # nc['disp_y'][istation*npoint_persta:(istation+1)*npoint_persta] = stream[wave_dim_s].data * np.sin(phi) + stream[wave_dim_p].data * np.cos(phi)
                        # nc['disp_z'][istation*npoint_persta:(istation+1)*npoint_persta] = stream[wave_dim_z].data

            ncfilepath = NETCDFDir+'/seismogram_%s_%s.nc' %(displacement_or_potential, AppliedNetwork)

            if os.path.exists(ncfilepath):
                os.remove(ncfilepath)

            nc = Dataset(ncfilepath, 'w')
            # nc.createDimension('nstation', size=len(stalatlon_dict))
            nc.createDimension('nlat', size=nlat)
            nc.createDimension('nlon', size=nlon)
            nc.createDimension('ntime', size=PointPerTrace)
            nc.createDimension('d3', size=3)

            nc.createVariable('station_coords_cartesian', float, dimensions=('nlat','nlon','d3'))
            nc['station_coords_cartesian'][:,:,:] = station_coords_cartesian[:,:,:]
            nc.createVariable('station_coords_spherical', float, dimensions=('nlat','nlon','d3'))
            nc['station_coords_spherical'][:,:,:] = station_coords_spherical[:,:,:]

            nc.createVariable('time', float, dimensions=('ntime'))
            nc['time'][:] = np.linspace(StartTime, EndTime, PointPerTrace+1)[0:PointPerTrace]

            
            nc.createVariable('disp', float, dimensions=('nlat','nlon','d3','ntime'))
            nc['disp'][:,:,:,:] = wave_disp[:,:,:,:]

            nc.close()


            print(stream[0].times()[0], stream[0].times()[-1])
            print(ncfilepath, " set up and saved!")

        except:
            print("processing %s ERROR !!!!!!!!!!!!!!!!" %ModelName)


with Pool(nproc) as p:
    p.map(Process,ModelNameList)  # Multiprocessing DownloadEvent

        