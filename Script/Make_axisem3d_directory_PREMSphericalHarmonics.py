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

sample_scaled = np.load('sampling_array_10000_prem_iso.npy')

ExampleInputDir = '../Runs/ExampleSphericalHarmonicsPREM/input'

SourceDepthRange = [0,800]
DvpRange = 0.02
DvsRange = 0.02
DrhoRange = 0.01

RadiusList = [6371.0, 6356.0, 6356.0, 6346.6, 6346.6, 6291.0, 6291.0, 6151.0, 6151.0, 5971.0, 5971.0, 5771.0, 5771.0, 5701.0, 5701.0, 5600.0, 5600.0, 3630.0, 3630.0, 3480.0, 3480.0, 1221.5, 1221.5, 0.0]
RadiusNotationList = ['6371.0','6356.0a', '6356.0b', '6346.6a', '6346.6b', '6291.0a', '6291.0b', '6151.0a', '6151.0b', '5971.0a', '5971.0b', '5771.0a', '5771.0b', '5701.0a', '5701.0b', '5600.0a', '5600.0b', '3630.0a', '3630.0b', '3480.0a', '3480.0b', '1221.5a', '1221.5b', '0.0']

# for imodel in np.arange(580,10000):
def Process(imodel):
    para_index = 0

    ModeName = 'LatinSphericalHarmonicsPREM%04d' %imodel
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
    # update event source (9 parameters)
    EventDepth = (sample_scaled[imodel,para_index]+1)/2*(SourceDepthRange[1]-SourceDepthRange[0])+SourceDepthRange[0]
    para_index += 1
    MomentTensor = sample_scaled[imodel,para_index:para_index+6]*1e24
    para_index += 6
    EventLat = (sample_scaled[imodel,para_index]+1)/2*180+(-90)
    para_index += 1
    EventLon = (sample_scaled[imodel,para_index]+1)/2*360+(-180)
    para_index += 1

    with open(EventParDir+'/input/inparam.source.yaml','r') as file:
        filetxt = file.read()
    filetxt = filetxt.replace("latitude_longitude: [90, 0]", "latitude_longitude: [%.2f, %.2f]" %(EventLat, EventLon))
    filetxt = filetxt.replace("depth: 200.0e0", "depth: %.1fe3" %(EventDepth))
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
    ModelCoeff['Radius'] = []
    ModelCoeff['l'] = []
    ModelCoeff['m'] = []
    ModelCoeff['Value'] = []

    l_max = 8
    RadiusList = [6371.0, 6356.0, 6356.0, 6346.6, 6346.6, 6291.0, 6291.0, 6151.0, 6151.0, 5971.0, 5971.0, 5771.0, 5771.0, 5701.0, 5701.0, 5600.0, 5600.0, 3630.0, 3630.0, 3480.0, 3480.0, 1221.5, 1221.5, 0.0]
    RadiusNotationList = ['6371.0','6356.0a', '6356.0b', '6346.6a', '6346.6b', '6291.0a', '6291.0b', '6151.0a', '6151.0b', '5971.0a', '5971.0b', '5771.0a', '5771.0b', '5701.0a', '5701.0b', '5600.0a', '5600.0b', '3630.0a', '3630.0b', '3480.0a', '3480.0b', '1221.5a', '1221.5b', '0.0'] 

    for Radius in RadiusNotationList:
        coeff[Radius] = {}

    for variable in ['dvp','dvs','drho']:
        for Radius in RadiusNotationList[0:-1]:

            if Radius=='3480.0b' and variable=='dvs':
                continue
            if Radius=='1221.5a' and variable=='dvs':
                continue

            
            for l in range(0,l_max+1):
                for m in np.arange(-l,l+1):
                    name = '%s_%s_%s' %(variable, l, m)
                    # print(l,m)
                    ModelCoeff['variable'].append(variable)
                    ModelCoeff['Radius'].append(Radius)
                    ModelCoeff['l'].append(l)
                    ModelCoeff['m'].append(m)

                    if variable == 'dvp':
                        Val = sample_scaled[imodel,para_index]*DvpRange  # Latin Hypercube Sampling
                    elif variable == 'dvs':
                        Val = sample_scaled[imodel,para_index]*DvsRange  # Latin Hypercube Sampling
                    elif variable == 'drho':
                        Val = sample_scaled[imodel,para_index]*DrhoRange  # Latin Hypercube Sampling

                    ModelCoeff['Value'].append(Val)
                    coeff[Radius][name] = Val 
                    para_index += 1


    df = pd.DataFrame(data=ModelCoeff)
    df.to_pickle(EventParDir+"/Spherical_Harmonics.pkl")

    # Make sure RADIUS and Coordinates are ascendingly sorted
    grid_lat = np.linspace(-90, 90, 181)
    grid_lon = np.linspace(-180, 180, 361)
    grid_lat.sort()
    grid_lon.sort()

    LON, LAT = np.meshgrid(grid_lon, grid_lat)
    DvpMLTomo = np.zeros([len(grid_lat), len(grid_lon), 2])
    DvsMLTomo = np.zeros([len(grid_lat), len(grid_lon), 2])
    DrhoMLTomo = np.zeros([len(grid_lat), len(grid_lon), 2])

    for TopDiscontinuity, BotDiscontinuity in zip(RadiusNotationList[0::2], RadiusNotationList[1::2]):
        # print('TopDiscontinuity: ', TopDiscontinuity, 'BotDiscontinuity: ', BotDiscontinuity)
        grid_radius = np.array([float(BotDiscontinuity.strip('a')), float(TopDiscontinuity.strip('b'))])

        for variable in ['dvp','dvs','drho']:
            if TopDiscontinuity=='3480.0b' and variable=='dvs': #Skip the outer core vs setting
                continue

            for i, SlicingRadius in enumerate([BotDiscontinuity, TopDiscontinuity]):
                
                # define the center value 
                if SlicingRadius == '0.0':
                    if variable == 'dvp':
                        DvpMLTomo[:,:,i] = 0.0
                    elif variable == 'dvs':
                        DvsMLTomo[:,:,i] = 0.0
                    elif variable == 'drho':
                        DrhoMLTomo[:,:,i] = 0.0
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

                        TomoSum[:,:] = TomoSum[:,:] + coeff[SlicingRadius][name] * Y_grid
                if variable == 'dvp':
                    DvpMLTomo[:,:,i] = TomoSum[:,:]
                elif variable == 'dvs':
                    DvsMLTomo[:,:,i] = TomoSum[:,:]
                elif variable == 'drho':
                    DrhoMLTomo[:,:,i] = TomoSum[:,:]
            
    

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
        # print('DvpMLTomo min: ', DvpMLTomo.min(), 'DvpMLTomo max: ', DvpMLTomo.max())

        # make discountinity nc file for 
        NCName = "degree8_random%sto%s.nc" %(TopDiscontinuity, BotDiscontinuity)
        # write to file
        if os.path.exists(EventParDir+'/input/'+NCName):
            os.remove(EventParDir+'/input/'+NCName)

        nc = Dataset(EventParDir+'/input/'+NCName, 'w')
        nc.createDimension('nlat', size=len(grid_lat))
        nc.createDimension('nlon', size=len(grid_lon))
        nc.createDimension('nradius', size=len(grid_radius))
        nc.createVariable('latitude', float, dimensions=('nlat'))
        nc['latitude'][:] = grid_lat
        nc.createVariable('longitude', float, dimensions=('nlon'))
        nc['longitude'][:] = grid_lon
        nc.createVariable('radius', float, dimensions=('nradius'))
        nc['radius'][:] = grid_radius
        nc.createVariable('dvp', float, dimensions=('nlat', 'nlon','nradius'))
        nc['dvp'][:,:,:] = DvpMLTomo[:,:,:]
        nc.createVariable('dvs', float, dimensions=('nlat', 'nlon','nradius'))
        nc['dvs'][:,:,:] = DvsMLTomo[:,:,:]
        nc.createVariable('drho', float, dimensions=('nlat', 'nlon','nradius'))
        nc['drho'][:,:,:] = DrhoMLTomo[:,:,:]

        if imodel == 0:
            nc['dvp'][:,:,:] = np.zeros(np.shape(DvpMLTomo[:,:,:]))
            nc['dvs'][:,:,:] = np.zeros(np.shape(DvsMLTomo[:,:,:]))
            nc['drho'][:,:,:] = np.zeros(np.shape(DrhoMLTomo[:,:,:]))
            nc.close()
            continue

        nc.close()

with Pool(nproc) as p:
    p.map(Process,np.arange(1000,10000))  # Multiprocessing DownloadEvent