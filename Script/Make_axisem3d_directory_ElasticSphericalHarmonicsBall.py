from obspy.core.event import read_events
import obspy
import numpy as np
import pandas as pd
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import os.path
import shutil
from obspy import UTCDateTime
from netCDF4 import Dataset
from matplotlib import cm
from skimage.filters import gaussian
from mpl_toolkits.basemap import Basemap
from itertools import chain

from scipy.stats import qmc
import h5py
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")

nproc = 20

sample_scaled = np.load('sampling_array_30000_elastic.npy')
ExampleInputDir = '../Runs/ExampleSphericalHarmonicsElasticBall/input'

# SourceDepthRange = [0,300]
DvpRange = 0.025
DvsRange = 0.025
DrhoRange = 0.025

l_max = 8
DepthList = [0. , 200, 400, 600, 800, 1000]
# for imodel in np.arange(580,10000):
def Process(imodel):

    para_index = 0

    ModeName = 'LatinSphericalHarmonicsElasticBall%04d' %imodel
    EventParDir='../Runs/%s' %(ModeName)  
    if not os.path.exists(EventParDir):
        os.makedirs(EventParDir)
    print(EventParDir, " created")

    if not os.path.exists(EventParDir+'/input'):
        os.makedirs(EventParDir+'/input')

    # copy parameter file
    # shutil.copy(ExampleInputDir+'/TomoNet_LowerMantle__10s.e',EventParDir+'/input/')

    shutil.copy(ExampleInputDir+'/inparam.model.yaml',EventParDir+'/input/')

    shutil.copy(ExampleInputDir+'/inparam.nr.yaml',EventParDir+'/input/')

    shutil.copy(ExampleInputDir+'/inparam.advanced.yaml',EventParDir+'/input/')

    shutil.copy(ExampleInputDir+'/inparam.source.yaml',EventParDir+'/input/')
    # update event source
    # EventDepth = (sample_scaled[imodel,para_index]+1)/2*(SourceDepthRange[1]-SourceDepthRange[0])+SourceDepthRange[0]
    # para_index += 1
    MomentTensor = sample_scaled[imodel,para_index:para_index+6]*1e10
    para_index += 6
    with open(EventParDir+'/input/inparam.source.yaml','r') as file:
        filetxt = file.read()
    # filetxt = filetxt.replace("latitude_longitude: [-56.24, 26.34]", "latitude_longitude: [%.2f, %.2f]" %(EventLat, EventLon))
    # filetxt = filetxt.replace("depth: 200.0e0", "depth: %.1fe0" %(EventDepth))
    filetxt = filetxt.replace("data: [1e10, 1e10, 1e10, 1e10, 1e10, 1e10]", "data: [%e, %e, %e, %e, %e, %e]"  %(MomentTensor[0], MomentTensor[1], MomentTensor[2], MomentTensor[3], MomentTensor[4], MomentTensor[5]))
    with open(EventParDir+'/input/inparam.source.yaml','w') as file:
        file.write(filetxt)

    shutil.copy(ExampleInputDir+'/inparam.output.yaml',EventParDir+'/input/')

    shutil.copy(ExampleInputDir+'/Synthetic_Stations_Ball.txt',EventParDir+'/input/')

    # # generate random model
    ### Real spherical harmonics
    coeff = {}

    ModelCoeff = dict()
    ModelCoeff['variable'] = []
    ModelCoeff['Depth'] = []
    ModelCoeff['l'] = []
    ModelCoeff['m'] = []
    ModelCoeff['Value'] = []



    for depth in DepthList:
        coeff[depth] = {}

    for variable in ['dvp','dvs','drho']:
        for depth in DepthList[0:-1]:
            
            for l in range(0,l_max+1):
                for m in np.arange(-l,l+1):
                    name = '%s_%s_%s' %(variable, l, m)
                    # print(l,m)
                    ModelCoeff['variable'].append(variable)
                    ModelCoeff['Depth'].append(depth)
                    ModelCoeff['l'].append(l)
                    ModelCoeff['m'].append(m)

                    if variable == 'dvp':
                        Val = sample_scaled[imodel,para_index]*DvpRange  # Latin Hypercube Sampling
                    elif variable == 'dvs':
                        Val = sample_scaled[imodel,para_index]*DvsRange  # Latin Hypercube Sampling
                    elif variable == 'drho':
                        Val = sample_scaled[imodel,para_index]*DrhoRange  # Latin Hypercube Sampling

                    ModelCoeff['Value'].append(Val)
                    coeff[depth][name] = Val 

                    para_index += 1

    df = pd.DataFrame(data=ModelCoeff)
    df.to_pickle(EventParDir+"/Spherical_Harmonics.pkl")

    grid_depth = np.array(DepthList)
    grid_lat = np.linspace(-90, 90, 181)
    grid_lon = np.linspace(-180, 180, 361)

    # Make sure RADISU and Coordinates are ascendingly sorted
    grid_depth.sort()
    grid_lat.sort()
    grid_lon.sort()

    LON, LAT = np.meshgrid(grid_lon, grid_lat)
    DvpMLTomo = np.zeros([len(grid_lat), len(grid_lon), len(grid_depth)])
    DvsMLTomo = np.zeros([len(grid_lat), len(grid_lon), len(grid_depth)])
    DrhoMLTomo = np.zeros([len(grid_lat), len(grid_lon), len(grid_depth)])

    for variable in ['dvp','dvs','drho']:
        for i, SlicingDepth in enumerate(DepthList):

            if SlicingDepth == 1000:
                DvpMLTomo[:,:,i] = 0
                DvsMLTomo[:,:,i] = 0
                DrhoMLTomo[:,:,i] = 0
                continue
            # initiate TomoSum
            TomoSum = np.zeros([len(grid_lat),len(grid_lon)])

            for l in range(0,l_max+1):
                for m in np.arange(-l,l+1):
                    # print('l, m = ', l, m)
                    name = '%s_%s_%s' %(variable, l, m)
                    Y_grid = sph_harm(m, l, np.radians(LON-180), np.radians(90-LAT))

                    if m < 0:
                        Y_grid = np.sqrt(2) * (-1)**(-m) * Y_grid.imag
                    elif m > 0:
                        Y_grid = np.sqrt(2) * (-1)**m * Y_grid.real

                    TomoSum[:,:] = TomoSum[:,:] + coeff[SlicingDepth][name] * Y_grid
            if variable == 'dvp':
                DvpMLTomo[:,:,i] = TomoSum[:,:]
            elif variable == 'dvs':
                DvsMLTomo[:,:,i] = TomoSum[:,:]
            elif variable == 'drho':
                DrhoMLTomo[:,:,i] = TomoSum[:,:]
    
    print(DvpMLTomo.min(), DvpMLTomo.max())
    print(DvsMLTomo.min(), DvsMLTomo.max())
    print(DrhoMLTomo.min(), DrhoMLTomo.max())

    # # Fig Preparation
    # dpi = 200
    # fig = plt.figure(figsize=(3.5,3),dpi=200)
    # ax = fig.add_subplot(111)

    # map = Basemap(projection='moll',lon_0=0,resolution='l') # moll Projection
    # PLOT = map.pcolormesh(LON, LAT, TomoSum, latlon=True, cmap=plt.get_cmap('jet'))
    # cbar = plt.colorbar(PLOT, ax=ax, shrink=0.5)
    # ax.set_title('Depth Slice at %s m to degrees %d' %(SlicingDepth, l_max))
    # draw_map(map)
    # map.drawcoastlines(linewidth=0.1)

    NCName = "degree8_random.nc"
    # write to file
    if os.path.exists(EventParDir+'/input/'+NCName):
        os.remove(EventParDir+'/input/'+NCName)

    nc = Dataset(EventParDir+'/input/'+NCName, 'w')
    nc.createDimension('nlat', size=len(grid_lat))
    nc.createDimension('nlon', size=len(grid_lon))
    nc.createDimension('ndepth', size=len(grid_depth))
    nc.createVariable('latitude', float, dimensions=('nlat'))
    nc['latitude'][:] = grid_lat
    nc.createVariable('longitude', float, dimensions=('nlon'))
    nc['longitude'][:] = grid_lon
    nc.createVariable('depth', float, dimensions=('ndepth'))
    nc['depth'][:] = grid_depth
    nc.createVariable('dvp', float, dimensions=('nlat', 'nlon','ndepth'))
    nc['dvp'][:,:,:] = DvpMLTomo[:,:,:]
    nc.createVariable('dvs', float, dimensions=('nlat', 'nlon','ndepth'))
    nc['dvs'][:,:,:] = DvsMLTomo[:,:,:]
    nc.createVariable('drho', float, dimensions=('nlat', 'nlon','ndepth'))
    nc['drho'][:,:,:] = DrhoMLTomo[:,:,:]

    if imodel == 0:
        nc['dvp'][:,:,:] = np.zeros(np.shape(DvpMLTomo[:,:,:]))
        nc['dvs'][:,:,:] = np.zeros(np.shape(DvsMLTomo[:,:,:]))
        nc['drho'][:,:,:] = np.zeros(np.shape(DrhoMLTomo[:,:,:]))
        nc.close()
        return



with Pool(nproc) as p:
    p.map(Process,np.arange(9734,30000))  # Multiprocessing DownloadEvent