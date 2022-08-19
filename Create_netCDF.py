# -*- coding: utf-8 -*-
"""

Created on Sat Jan 15 14:23:29 2022



MPH, NSW-IMOS, m.hemming@unsw.edu.au

"""

# Import Packages
#----------------------------------------------------------------------------
# Import netCDF package
from netCDF4 import Dataset,num2date,date2num
# Import datetime for creation date of file
import datetime
# Import numpy and h5py to load in .mat files
import numpy as np
import h5py
# Import array to create integer arrays
import array
# import for time ranges
import datetime as dt
from dateutil import rrule, parser
import pandas as pd
from scipy.io import loadmat
import os

# change directory
os.chdir('C:\\Users\\mphem\\Documents\\Work\\UEA\\UEA_work\\NCP_Scripts\\data\\')

#----------------------------------------------------------------------------

# %% -----------------------------------------------------------------------------------------------
# Time-related functions


# -----------------------------------------------------------------------------------------------
# datetime to datetime64

def to_date64(TIME):
        t = []
        for nt in range(len(TIME)):
            o = TIME[nt]
            if '64' not in str(type(o)):
                t.append(np.datetime64(o.strftime("%Y-%m-%dT%H:%M:%S")))
            else:
                t.append(o)
        TIME = np.array(t)

        return TIME

# -----------------------------------------------------------------------------------------------
# datevec function

def datevec(TIME):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    TIME : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # If not a datetime64, convert from dt.datetime
    TIME = to_date64(TIME)
    # allocate output
    out = np.empty(TIME.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [TIME.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (TIME - D).astype("m8[h]") # hour
    out[..., 4] = (TIME - h).astype("m8[m]") # minute
    out[..., 5] = (TIME - m).astype("m8[s]") # second
    out[..., 6] = (TIME - s).astype("m8[us]") # microsecond

    yr = out[:,0]; mn = out[:,1]; dy = out[:,2]; hr = out[:,3];
    yday = []
    for n in range(len(yr)):
        yday.append(dt.date(yr[n], mn[n], dy[n]).timetuple().tm_yday)

    return yr, mn, dy, hr, yday

def to_datetime(TIME):
    if 'xarray' in str(type(TIME)):
        TIME = np.array(TIME)
    if np.size(TIME) == 1:
        TIME = TIME.tolist()
    else:
        t = []
        # Check that input is xarray data array
        # if 'xarray' not in str(type(TIME)):
        #     TIME = xr.DataArray(TIME)
        for nt in range(len(TIME)):
            o = TIME[nt]
            if '64' in str(type(o)):
                t_str = str(np.array(o))
                if len(t_str) > 10:
                    yr = int(t_str[0:4])
                    mn = int(t_str[5:7])
                    dy = int(t_str[8:10])
                    hr = int(t_str[11:13])
                    mins = int(t_str[14:16])
                    secs = int(t_str[17:19])
                    t.append(dt.datetime(yr,mn,dy,hr,mins,secs))
                if len(t_str) == 10:
                    yr = int(t_str[0:4])
                    mn = int(t_str[5:7])
                    dy = int(t_str[8:10])
                    t.append(dt.datetime(yr,mn,dy))
                if len(t_str) == 7:
                    yr = int(t_str[0:4])
                    mn = int(t_str[5:7])
                    t.append(dt.datetime(yr,mn,1))
                if len(t_str) == 4:
                    t.append(dt.datetime(yr,1,1))
        TIME = np.array(t)

    return TIME

def time_range(start,end,res,time_format):
    """
    start / end = can either be integer years, or numpy
                  datetime64/datetime dates (don't mix)
    res = 'monthly','daily','yearly'
    time_format = 'np64' or 'datetime'

    """
    if 'int' not in str(type(start)):
        if '64' not in str(type(start)):
            start = np.datetime64(start)
            end = np.datetime64(end)
        if 'monthly' in res:
                time = np.arange(start,end,np.timedelta64(1, 'M'),
                                 dtype='datetime64[M]')
        if 'daily' in res:
            time = np.arange(start, end, np.timedelta64(1, 'D'),
                             dtype='datetime64[D]')
        if 'yearly' in res:
            time = np.arange(start, end, np.timedelta64(1, 'Y'),
                             dtype='datetime64[Y]')
        time = np.array(time)
    else:

        if 'monthly' in res:
            time = np.arange(np.datetime64(str(start) + '-01-01'),
                             np.datetime64(str(end) + '-01-01'),
                             np.timedelta64(1, 'M'),
                             dtype='datetime64[M]')
        if 'daily' in res:
            time = np.arange(np.datetime64(str(start) + '-01-01'),
                             np.datetime64(str(end) + '-01-01'),
                             np.timedelta64(1, 'D'),
                             dtype='datetime64[D]')
        if 'yearly' in res:
            time = np.arange(np.datetime64(str(start)),
                             np.datetime64(str(end)),
                             np.timedelta64(1, 'Y'),
                             dtype='datetime64[Y]')

    if 'np64' not in time_format:
        time = to_datetime(np.array(time))

    return time


# Convert to MATLAB datenum

# def datetime2matlabdn(d):
#    mdn = d + dt.timedelta(days = 366)
#    frac_seconds = (d-dt.datetime(d.year,d.month,d.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
#    frac_microseconds = d.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
#    return mdn.toordinal() + frac_seconds + frac_microseconds

def datetime2matlabdn(python_datetime):

    if np.size(python_datetime) != 1:

        datenum = []
        for n in range(len(python_datetime)):

            mdn = python_datetime[n] + dt.timedelta(days = 366)
            frac_seconds = (python_datetime[n]-dt.datetime(
               python_datetime[n].year,python_datetime[n].month,
               python_datetime[n].day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
            frac_microseconds = python_datetime[n].microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)

            datenum.append(mdn.toordinal() + frac_seconds + frac_microseconds)
    else:
        mdn = python_datetime + dt.timedelta(days = 366)
        frac_seconds = (python_datetime-dt.datetime(
           python_datetime.year,python_datetime.month,
           python_datetime.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
        frac_microseconds = python_datetime.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
        datenum = mdn.toordinal() + frac_seconds + frac_microseconds

    return datenum

def matlabdn2datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    time = []
    for n in range(len(datenum)):
        dn = np.float(datenum[n])
        days = dn % 1
        d = dt.datetime.fromordinal(int(dn)) \
           + dt.timedelta(days=days) \
           - dt.timedelta(days=366)
        time.append(d)

    return time



# %%---------------------------------------------------------
# Load in a mat file, convert a dictionary to an object

class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

def load_MATfile(filename):

    try:
        # if not 'v7.3'
        data = loadmat(filename)
    except:
        # if 'v7.3'
        data = h5py.File(filename,'r')
    # convert dictionary to class
    if 'dict' in str(type(data)):
        # if not 'v7.3'
        data = Dict2Obj(data)
    else:
        # if 'v7.3'
        keys = list(data.keys())
        keys = keys[2:-1]
        # data = Dict2Obj(data)
        data_dict = {}
        for n_keys in range(len(keys)):
            d = np.array(eval('data.get("' + keys[n_keys] + '")'))
            data_dict[keys[n_keys]] = []
            data_dict[keys[n_keys]].append(d)

    return data


# %%---------------------------------------------------------
# Load in glider prcdata file

filename = ('C:\\Users\\mphem\\Documents\\Work\\UEA\\UEA_work\\' +
            'BODC\\BODC_data.mat')

data = load_MATfile(filename)

# get variables
t64 = to_date64(matlabdn2datetime(np.array(data.TIME[0])))
t = np.array(data.TIME[0])-712224
T = data.TEMP[0]
S = data.PSAL[0]
P = data.PRES[0]
D = data.DEPTH[0]
lon = data.LONGITUDE[0]
lat = data.LATITUDE[0]
pH = data.pH[0]
pH_counts = data.pH_COUNTS[0]
ALKd = data.ALK[0]
DICd = data.DIC[0]
O2 = data.DOX[0]
O2_raw = data.DOX_RAW[0]
chl = data.CHL[0]
Scatter_470 = data.BBP470[0]
Scatter_700 = data.BBP700[0]
MLDd = data.MLD[0]
downup = data.DNUP[0]
DIVESd = data.DIVES[0]
DACSu = data.DACu[0]
DACSv = data.DACv[0]
# quality control variables
t_QC = data.TIME_quality_control[0]
T_QC = data.TEMP_quality_control[0]
S_QC = data.PSAL_quality_control[0]
P_QC = data.PRES_quality_control[0]
D_QC = data.DEPTH_quality_control[0]
lon_QC = data.LONGITUDE_quality_control[0]
lat_QC = data.LATITUDE_quality_control[0]
O2_raw_QC = data.DOX_RAW_quality_control[0]
chl_QC = data.CHL_quality_control[0]
Scatter_470_QC = data.BBP470_quality_control[0]
Scatter_700_QC = data.BBP700_quality_control[0]
DACSu_QC = data.DACu_quality_control[0]
DACSv_QC = data.DACv_quality_control[0]

#%%
# Create filename
#----------------------------------------------------------------------------
# create file name
# get current date
dtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = ("UEAGLIDER_START_" + '20160307T134032' +
            "Z_SG537_MISSION30_BOUSSOLE_" +
            'END_20160405T063821' + "Z_C" + dtime + "Z.nc")

# create file
dataset = Dataset(filename, 'w',  format='NETCDF4_CLASSIC')
print('Filename created:' + filename)
#----------------------------------------------------------------------------

#%%
# define fill value for whole file
#----------------------------------------------------------------------------
fillvalue = 999999
#----------------------------------------------------------------------------

#%%
# Define dimensions,coordinates,variables of netCDF file
#----------------------------------------------------------------------------
#Define a set of dimensions used for your variables:
TIME_DIM = dataset.createDimension('TIME', np.size(t))
# variables
DEPLOYMENT = dataset.createVariable('DEPLOYMENT', np.float32, (), fill_value=fillvalue)
PLATFORM = dataset.createVariable('PLATFORM', np.float32, (), fill_value=fillvalue)
SENSOR1 = dataset.createVariable('SENSOR1', np.float32, (), fill_value=fillvalue)
SENSOR2 = dataset.createVariable('SENSOR2', np.float32, (), fill_value=fillvalue)
SENSOR3 = dataset.createVariable('SENSOR3', np.float32, (), fill_value=fillvalue)
SENSOR4 = dataset.createVariable('SENSOR4', np.float32, (), fill_value=fillvalue)
TIME = dataset.createVariable('TIME', np.float32, ('TIME',), fill_value=fillvalue)
TIME_QC = dataset.createVariable('TIME_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
TEMP = dataset.createVariable('TEMP', np.float32, ('TIME',), fill_value=fillvalue)
TEMP_QC = dataset.createVariable('TEMP_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
DEPTH = dataset.createVariable('DEPTH', np.float32, ('TIME',), fill_value=fillvalue)
DEPTH_QC = dataset.createVariable('DEPTH_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
PRES = dataset.createVariable('PRES', np.float32, ('TIME',), fill_value=fillvalue)
PRES_QC = dataset.createVariable('PRES_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
LATITUDE = dataset.createVariable('LATITUDE', np.float32, ('TIME',), fill_value=fillvalue)
LATITUDE_QC = dataset.createVariable('LATITUDE_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
LONGITUDE = dataset.createVariable('LONGITUDE', np.float32, ('TIME',), fill_value=fillvalue)
LONGITUDE_QC = dataset.createVariable('LONGITUDE_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
PSAL = dataset.createVariable('PSAL', np.float32, ('TIME',), fill_value=fillvalue)
PSAL_QC = dataset.createVariable('PSAL_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
DOX = dataset.createVariable('DOX', np.float32, ('TIME',), fill_value=fillvalue)
DOX_RAW = dataset.createVariable('DOX_RAW', np.float32, ('TIME',), fill_value=fillvalue)
DOX_RAW_QC = dataset.createVariable('DOX_RAW_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
CPHL = dataset.createVariable('CPHL', np.float32, ('TIME',), fill_value=fillvalue)
CPHL_QC = dataset.createVariable('CPHL_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
BBP_470 = dataset.createVariable('BBP_470', np.float32, ('TIME',), fill_value=fillvalue)
BBP_470_QC = dataset.createVariable('BBP_470_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
BBP_700 = dataset.createVariable('BBP_700', np.float32, ('TIME',), fill_value=fillvalue)
BBP_700_QC = dataset.createVariable('BBP_700_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
PH = dataset.createVariable('PH', np.float32, ('TIME',), fill_value=fillvalue)
PH_COUNTS = dataset.createVariable('PH_COUNTS', np.float32, ('TIME',), fill_value=fillvalue)
ALK = dataset.createVariable('ALK', np.float32, ('TIME',), fill_value=fillvalue)
DIC = dataset.createVariable('DIC', np.float32, ('TIME',), fill_value=fillvalue)
MLD = dataset.createVariable('MLD', np.float32, ('TIME',), fill_value=fillvalue)
DOWNUP = dataset.createVariable('DOWNUP', np.float32, ('TIME',), fill_value=fillvalue)
DIVES = dataset.createVariable('DIVES', np.float32, ('TIME',), fill_value=fillvalue)
DACSU = dataset.createVariable('UCUR', np.float32, ('TIME',), fill_value=fillvalue)
DACSU_QC = dataset.createVariable('UCUR_quality_control', np.float32, ('TIME',), fill_value=fillvalue)
DACSV = dataset.createVariable('VCUR', np.float32, ('TIME',), fill_value=fillvalue)
DACSV_QC = dataset.createVariable('VCUR_quality_control', np.float32, ('TIME',), fill_value=fillvalue)

print('netCDF dimensions, coordinates, and variables defined')
#----------------------------------------------------------------------------

#%%
# Produce global attributes
#----------------------------------------------------------------------------
# Ensure to update these attributes later!
dtime_modified = dtime[0:4] + '-' + dtime[4:6] + '-' + dtime[6:8] + 'T' + dtime[8:10] + ':' + dtime[10:12] + ':' + dtime[12:14] + 'Z'
dataset.date_created = dtime_modified
dataset.title = 'Deployment of SG537 at the Boussole time series site (Mediterranean Sea) and pH/p(CO2) sensor testing'
dataset.abstract = ('Data for University of East Anglia (UEA) Seaglider sg537 ("'"Fin"'") deployed near' +
                    ' the BOUSSOLE/DyFAMed sites in the northwestern Mediterranean Sea in' +
                    ' March-April 2016. The aim of the deployment was to test two' +
                    ' experimental ISFET pH-pCO2 sensors, and to estimate net community' +
                    ' production. Please see Hemming et al. (Ocean Science, 2022) for more information.')
# dataset.conventions = 'CF-1.6,IMOS-1.4'
dataset.acknowledgement = ('Any users of this data are required to acknowledge' +
                ' the University of East Anglia, and cite the dataset' +
                ' (see citation attribute)')
dataset.citation = ('The citation in a list of references is: "BOUSSOLE sg537 [year-of-data-download]' +
                    ', [Title], [data-access-URL], accessed [date-of-access]."')
dataset.file_version = 'Level 1 - Processed and quality controlled data set'
dataset.institution = 'University of East Anglia'
dataset.geospatial_lat_unit = 'degrees_north'
dataset.geospatial_lon_units = 'degrees_east'
dataset.geospatial_vertical_units = 'meter'
dataset.geospatial_lat_max = np.round(np.nanmax(lat),2)
dataset.geospatial_lat_min = np.round(np.nanmin(lat),2)
dataset.geospatial_lon_max = np.round(np.nanmax(lon),2)
dataset.geospatial_lon_min = np.round(np.nanmin(lon),2)
dataset.geospatial_vertical_max = np.round(np.nanmax(P),2)
dataset.geospatial_vertical_min = 0
dataset.geospatial_vertical_positive = 'down'
dataset.data_mode = 'Delayed Mode'
dataset.local_time_zone = 1.0
dataset.netcdf_version = 4.0
dataset.platform_code = 'SG537'
dataset.principal_investigator = 'Kaiser, Jan'
dataset.principal_investigator_email = 'j.kaiser@uea.ac.uk'
dataset.dataset_creator = 'Hemming, Michael'
dataset.dataset_creator_email = 'm.hemming@unsw.edu.au'
dataset.project = 'BOUSSOLE'
dataset.references = 'https://ueaglider.uea.ac.uk/mission30'
dataset.data_centre = 'British Oceanographic Data Centre (BODC)'
dataset.data_centre_email = 'glidersbodc@bodc.ac.uk'
dataset.disclaimer = 'This data set is provided "as is" without any warranty as to fitness for a particular purpose.'
dataset.distribution_statement = 'Data may be re-used, provided that related metadata explaining the data \
has been reviewed by the user, and the data is appropriately acknowledged.'
dataset.license = 'http://creativecommons.org/licenses/by/4.0/'
dataset.standard_name_vocabulary = 'NetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table 60'
dataset.time_coverage_start = str(np.nanmin(t64))
dataset.time_coverage_end = str(np.nanmax(t64))
#----------------------------------------------------------------------------
#%%
# Produce variable attributes
#----------------------------------------------------------------------------
# time units
time_unit_out= "days since 1950-01-01 00:00:00 UTC"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Time
TIME.units = time_unit_out
TIME.long_name = 'analysis time'
TIME.standard_name = 'time'
TIME.max = np.max(t)
TIME.min = np.min(t)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Time QC
TIME_QC.standard_name = "time status_flag"
TIME_QC.long_name = "quality control flag for time from the CTD"
TIME_QC.valid_min = 0
TIME_QC.valid_max = 9
TIME_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
TIME_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
TIME_QC.quality_control_conventions = "IMOS standard set using the IODE flags"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DEPTH
DEPTH.units = 'm'
DEPTH.long_name = 'sea water depth'
DEPTH.standard_name = 'sea_water_depth'
DEPTH.max = np.max(D)
DEPTH.min = np.min(D)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DEPTH QC
DEPTH_QC.standard_name = "depth status_flag"
DEPTH_QC.long_name = "quality control flag for depth from the CTD"
DEPTH_QC.valid_min = 0
DEPTH_QC.valid_max = 9
DEPTH_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
DEPTH_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
DEPTH_QC.quality_control_conventions = "IMOS standard set using the IODE flags"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 # Deployment info
DEPLOYMENT.deployment_start_date = str(np.nanmin(t64))
DEPLOYMENT.deployment_end_date = str(np.nanmax(t64))
DEPLOYMENT.deployment_start_longitude = np.round(lon[0],2)
DEPLOYMENT.deployment_start_latitude = np.round(lat[0],2)
DEPLOYMENT.long_name = 'deployment_information'
DEPLOYMENT.glider_technicians = 'Lee, Gareth and Cobas-Garcia, Marcos and Shitashima, Kiminori'
DEPLOYMENT.deployment_end_longitude = np.round(lon[-1],2)
DEPLOYMENT.deployment_end_latitude = np.round(lat[-1],2)
DEPLOYMENT.deployment_end_status = 'recovered'
DEPLOYMENT.deployment_pilot = 'UEA Seaglider Facility'
DEPLOYMENT.comment = 'Issue with stand-alone ISFET pH-pCO2 sensor. Data not used.'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Platform info
PLATFORM.trans_system_id= 'Iridium'
PLATFORM.positioning_system= 'GPS'
PLATFORM.platform_type= 'Seaglider 1KA with Ogive Fairing'
PLATFORM.long_name= 'Platform information'
PLATFORM.platform_maker= 'iRobot'
PLATFORM.battery_type= 'Lithium'
PLATFORM.glider_serial_no = 'SG537'
PLATFORM.glider_owner= 'UEA Seaglider Facility'
PLATFORM.operating_institution= 'University of East Anglia (UEA)'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SENSOR1
SENSOR1.sensor_type= 'CTD'
SENSOR1.sensor_maker= 'Seabird'
SENSOR1.sensor_model= 'SBE_CT'
SENSOR1.serial_number = 'SN: 0135'
SENSOR1.long_name= 'Sensor1 information'
SENSOR1.sensor_parameters= 'TEMP, PRES, DEPTH, PSAL'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SENSOR2
SENSOR2.sensor_type= 'ECO Puck'
SENSOR2.sensor_maker= 'Wetlabs'
SENSOR2.sensor_model= 'BBFL2VMT'
SENSOR2.serial_number = 'SN: 816'
SENSOR2.long_name= 'Sensor2 information'
SENSOR2.sensor_parameters= 'CPHL, BBP'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SENSOR3
SENSOR3.sensor_type= 'oxygen sensor'
SENSOR3.sensor_maker= 'Aanderaa'
SENSOR3.sensor_model= 'aa4330'
SENSOR3.serial_number = 'SN: 251'
SENSOR3.long_name= 'Sensor3 information'
SENSOR3.sensor_parameters= 'DOX'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SENSOR4
SENSOR4.sensor_type= 'Ion-sensitive Field-Effect Transistor'
SENSOR4.sensor_maker= 'Shitashima, Kiminori (Tokyo University of Marine Science and Technology)'
SENSOR4.sensor_model= 'Experimental'
SENSOR4.serial_number = 'N/A, non-commercial sensor'
SENSOR4.long_name= 'Sensor4 information'
SENSOR4.sensor_parameters= 'PH, ALK, DIC'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRES
PRES.standard_name= 'sea_water_pressure'
PRES.long_name= 'sea water pressure'
PRES.units= 'dbar'
PRES.min= np.round(np.nanmin(P),2)
PRES.max=  np.round(np.nanmax(P),2)
PRES.coordinates= 'TIME LATITUDE LONGITUDE DEPTH'
PRES.comment= 'pressure measured by the CTD'
PRES.observation_type= 'measured'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRES QC
PRES_QC.standard_name = "pressure status_flag"
PRES_QC.long_name = "quality control flag for pressure from the CTD"
PRES_QC.valid_min = 0
PRES_QC.valid_max = 9
PRES_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
PRES_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
PRES_QC.quality_control_conventions = "IMOS standard set using the IODE flags"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# latitude
LATITUDE.standard_name= 'latitude'
LATITUDE.long_name= 'latitude'
LATITUDE.units= 'degrees_north'
LATITUDE.axis= 'Y'
LATITUDE.min= np.round(np.nanmin(lat),2)
LATITUDE.max= np.round(np.nanmax(lat),2)
LATITUDE.comment= 'obtained from GPS fixes'
LATITUDE.reference_datum= 'geographical coordinates, WGS84 projection'
LATITUDE.observation_type= 'measured'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LATITUDE QC
LATITUDE_QC.standard_name = "latitude status_flag"
LATITUDE_QC.long_name = "quality control flag for latitude"
LATITUDE_QC.valid_min = 0
LATITUDE_QC.valid_max = 9
LATITUDE_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
LATITUDE_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
LATITUDE_QC.quality_control_conventions = "IMOS standard set using the IODE flags"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# longitude
LONGITUDE.standard_name= 'longitude'
LONGITUDE.long_name= 'longitude'
LONGITUDE.units= 'degrees_east'
LONGITUDE.axis= 'X'
LONGITUDE.min= np.round(np.nanmin(lon),2)
LONGITUDE.max= np.round(np.nanmax(lon),2)
LONGITUDE.comment= 'obtained from GPS fixes'
LONGITUDE.reference_datum= 'geographical coordinates, WGS84 projection'
LONGITUDE.observation_type= 'measured'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LONGITUDE QC
LONGITUDE_QC.standard_name = "longitude status_flag"
LONGITUDE_QC.long_name = "quality control flag for longitude"
LONGITUDE_QC.valid_min = 0
LONGITUDE_QC.valid_max = 9
LONGITUDE_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
LONGITUDE_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
LONGITUDE_QC.quality_control_conventions = "IMOS standard set using the IODE flags"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Salinity
PSAL.standard_name = 'sea_water_salinity'
PSAL.long_name = 'sea water salinity (PSS-78)'
PSAL.min = np.round(np.nanmin(S),2)
PSAL.max = np.round(np.nanmax(S),2)
PSAL.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
PSAL.observation_type = 'computed'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PSAL QC
PSAL_QC.standard_name = "Salinity status_flag"
PSAL_QC.long_name = "quality control flag for salinity from the CTD"
PSAL_QC.valid_min = 0
PSAL_QC.valid_max = 9
PSAL_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
PSAL_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
PSAL_QC.quality_control_conventions = "IMOS standard set using the IODE flags"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Oxygen
DOX.standard_name = 'moles_of_oxygen_per_unit_mass_in_sea_water'
DOX.long_name = 'amount content of corrected dissolved oxygen in seawater'
DOX.units = 'µmol kg-1'
DOX.min = np.round(np.nanmin(O2),2)
DOX.max = np.round(np.nanmax(O2),2)
DOX.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
DOX.observation_type = 'computed'
DOX.comment = 'These are the corrected dissolved oxygen concentrations. Please see Hemming et al. (Ocean Science, 2022) for more information.'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Oxygen raw
DOX_RAW.standard_name = 'moles_of_oxygen_per_unit_mass_in_sea_water'
DOX_RAW.long_name = 'Amount content of uncorrected raw dissolved oxygen in seawater'
DOX_RAW.units = 'µmol kg-1'
DOX_RAW.min = np.round(np.nanmin(O2_raw),2)
DOX_RAW.max = np.round(np.nanmax(O2_raw),2)
DOX_RAW.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
DOX_RAW.observation_type = 'computed'
DOX_RAW.comment = 'These are the uncorrected raw dissolved oxygen concentrations. Please see Hemming et al. (Ocean Science, 2022) for more information.'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DOX_RAW QC
DOX_RAW_QC.standard_name = "uncorrected raw oxygen status_flag"
DOX_RAW_QC.long_name = "quality control flag for uncorrected raw oxygen from the CTD"
DOX_RAW_QC.valid_min = 0
DOX_RAW_QC.valid_max = 9
DOX_RAW_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
DOX_RAW_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
DOX_RAW_QC.quality_control_conventions = "IMOS standard set using the IODE flags"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Chlorophyll
CPHL.standard_name = 'mass_concentration_of_chlorophyll_in_sea_water'
CPHL.long_name = 'mass concentration of chlorophyll a in sea water'
CPHL.units = 'mg m-3'
CPHL.min = np.round(np.nanmin(chl),2)
CPHL.max = np.round(np.nanmax(chl),2)
CPHL.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
CPHL.observation_type = 'computed'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CPHL QC
CPHL_QC.standard_name = "chlorophyll status_flag"
CPHL_QC.long_name = "quality control flag for chlorophyll from the ECO Puck"
CPHL_QC.valid_min = 0
CPHL_QC.valid_max = 9
CPHL_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
CPHL_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
CPHL_QC.quality_control_conventions = "IMOS standard set using the IODE flags"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BBP 470
BBP_470.long_name = 'particle backscattering coefficient [470nm]'
BBP_470.units = 'sr-1 nm-1'
BBP_470.min = np.round(np.nanmin(Scatter_470),2)
BBP_470.max = np.round(np.nanmax(Scatter_470),2)
BBP_470.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
BBP_470.observation_type = 'computed'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BBP_470 QC
BBP_470_QC.standard_name = "backscatter [470nm] status_flag"
BBP_470_QC.long_name = "quality control flag for backscatter from the ECO Puck"
BBP_470_QC.valid_min = 0
BBP_470_QC.valid_max = 9
BBP_470_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
BBP_470_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
BBP_470_QC.quality_control_conventions = "IMOS standard set using the IODE flags"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BBP 700
BBP_700.long_name = 'particle backscattering coefficient [700nm]'
BBP_700.units = 'sr-1 nm-1'
BBP_700.min = np.round(np.nanmin(Scatter_700),2)
BBP_700.max = np.round(np.nanmax(Scatter_700),2)
BBP_700.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
BBP_700.observation_type = 'computed'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BBP_700 QC
BBP_700_QC.standard_name = "backscatter [700nm] status_flag"
BBP_700_QC.long_name = "quality control flag for backscatter from the ECO Puck"
BBP_700_QC.valid_min = 0
BBP_700_QC.valid_max = 9
BBP_700_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
BBP_700_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
BBP_700_QC.quality_control_conventions = "IMOS standard set using the IODE flags"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pH
PH.long_name = 'pH'
PH.min = np.round(np.nanmin(pH),2)
PH.max = np.round(np.nanmax(pH),2)
PH.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
PH.observation_type = 'computed'
PH.comment = 'These are the corrected pH measurements. Please see Hemming et al. (Ocean Science, 2022) for more information.'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pH counts
PH_COUNTS.long_name = 'pH sensor raw counts'
PH_COUNTS.min = np.round(np.nanmin(pH_counts),2)
PH_COUNTS.max = np.round(np.nanmax(pH_counts),2)
PH_COUNTS.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
PH_COUNTS.observation_type = 'measured'
PH_COUNTS.comment = 'These are the uncorrected raw pH sensor counts. Please see Hemming et al. (Ocean Science, 2022) for more information.'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ALK
ALK.long_name = 'total alkalinity'
ALK.units = 'µmol kg-1'
ALK.min = np.round(np.nanmin(ALKd),2)
ALK.max = np.round(np.nanmax(ALKd),2)
ALK.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
ALK.observation_type = 'computed'
ALK.comment = ('Variable derived from an alkalinity-salinity relationship using discrete ship' +
                'samples collected in spring 2016 above the salinity maximum. ' +
                'Please see Hemming et al. (Ocean Science, 2022) for more information.')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DIC
DIC.long_name = 'Dissolved Inorganic Carbon content'
DIC.units = 'µmol kg-1'
DIC.min = np.round(np.nanmin(DICd),2)
DIC.max = np.round(np.nanmax(DICd),2)
DIC.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
DIC.observation_type = 'computed'
DIC.comment = ('Variable computed using salinity-derived alkalinity, corrected pH, and the CO2SYS software package.' +
                'Please see Hemming et al. (Ocean Science, 2022) for more information.')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MLD
MLD.long_name = 'Mixed Layer Depth'
MLD.units = 'm'
MLD.min = np.round(np.nanmin(MLDd),2)
MLD.max = np.round(np.nanmax(MLDd),2)
MLD.coordinates = 'TIME LATITUDE LONGITUDE'
MLD.observation_type = 'computed'
MLD.comment = 'mixed layer depths derived using the potential density criteria of the hybrid scheme of Holte and Talley (2009).'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DOWNUP
DOWNUP.long_name = 'Index for selecting descending or ascending glider dive profiles'
DOWNUP.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
DOWNUP.index_values = '1, 2'
DOWNUP.index_meanings = 'descending profiles, ascending profiles'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DIVES
DIVES.long_name = 'dive number'
DIVES.min = 1
DIVES.max = 147
DIVES.coordinates = 'TIME LATITUDE LONGITUDE DEPTH'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DACu
DACSU.standard_name = 'eastward_sea_water_velocity'
DACSU.long_name = 'eastward_sea_water_velocity'
DACSU.units = 'm s-1'
DACSU.min = np.round(np.nanmin(DACSu),2)
DACSU.max = np.round(np.nanmax(DACSu),2)
DACSU.coordinates = 'TIME LATITUDE LONGITUDE'
DACSU.comment = ('Average eastward velocity of the seawater over all ' +
                'the water that the glider travels through between surfacing. ' +
                'The values are rough estimates derived from engineering parameters.')
DACSU.observation_type = 'computed'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DACSU QC
DACSU_QC.standard_name = "average eastward velocity status_flag"
DACSU_QC.long_name = "quality control flag for average eastward velocity"
DACSU_QC.valid_min = 0
DACSU_QC.valid_max = 9
DACSU_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
DACSU_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
DACSU_QC.quality_control_conventions = "IMOS standard set using the IODE flags"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DACv
DACSV.standard_name = 'northward_sea_water_velocity'
DACSV.long_name = 'northward_sea_water_velocity'
DACSV.units = 'm s-1'
DACSV.min = np.round(np.nanmin(DACSv),2)
DACSV.max = np.round(np.nanmax(DACSv),2)
DACSV.coordinates = 'TIME LATITUDE LONGITUDE'
DACSV.comment = ('Average northward velocity of the seawater over all ' +
                'the water that the glider travels through between surfacing. ' +
                'The values are rough estimates derived from engineering parameters.')
DACSV.observation_type = 'computed'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DACSV QC
DACSV_QC.standard_name = "average northward velocity status_flag"
DACSV_QC.long_name = "quality control flag for average northward velocity"
DACSV_QC.valid_min = 0
DACSV_QC.valid_max = 9
DACSV_QC.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
DACSV_QC.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_values missing_values"
DACSV_QC.quality_control_conventions = "IMOS standard set using the IODE flags"

#%%
# Assign data to netCDF file
#----------------------------------------------------------------------------
TIME[:] = t
TIME_QC[:] = t_QC
DEPTH[:] = D
DEPTH_QC[:] = D_QC
PRES[:] = P
PRES_QC[:] = P_QC
TEMP[:] = T
TEMP_QC[:] = T_QC
LONGITUDE[:] = lon
LONGITUDE_QC[:] = lon_QC
LATITUDE[:] = lat
LATITUDE_QC[:] = lat_QC
PSAL[:] = S
PSAL_QC[:] = S_QC
DOX[:] = O2
DOX_RAW[:] = O2_raw
DOX_RAW_QC[:] = O2_raw_QC
CPHL[:] = chl
CPHL_QC[:] = chl_QC
BBP_470[:] = Scatter_470
BBP_470_QC[:] = Scatter_470_QC
BBP_700[:] = Scatter_700
BBP_700_QC[:] = Scatter_700_QC
PH[:] = pH
PH_COUNTS[:] = pH_counts
ALK[:] = ALKd
DIC[:] = DICd
MLD[:] = MLDd
DOWNUP[:] = downup
DIVES[:] = DIVESd
DACSU[:] = DACSu
DACSU_QC[:] = DACSu_QC
DACSV[:] = DACSv
DACSV_QC[:] = DACSv_QC

dataset.close()# and the file is written
print('File created.')
#----------------------------------------------------------------------------
