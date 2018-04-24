#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : jlerat
## Created : 2018-04-23 22:19:45.500351
## Comment : Get obs data from EHPDB
##
## ------------------------------
import sys, os, re, json, math
import numpy as np
import pandas as pd

from datetime import  datetime

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from hydrodiy.io import csv, iutils

import pyproj

import datasets

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

name = sys.argv[1]
version = sys.argv[2]
nsites = 20

# GDA94
proj = pyproj.Proj('+init=EPSG:3112')

# Columns selected
cc = ['rain[mm/d]', 'evap[mm/d]', 'runoff[mm/d]']

# Start end of period
start = datetime(1970, 1, 1)
end = datetime(2015, 12, 31)

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

fout = os.path.join(froot, 'input_data')
os.makedirs(fout, exist_ok=True)

basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
dset = datasets.Dataset(name, version)
sites = dset.sites

idx = sites.suspicious.astype(str) == 'nan'
sites = sites.loc[idx, :]


# Circular index
twopi = 2*math.pi
x0, y0 = proj(134.5, -26.5)
x = sites.loc[:, 'xGDA94_centroid[m]']
y = sites.loc[:, 'yGDA94_centroid[m]']
dx = x-x0
dy = y-y0

theta = np.arctan(dy/dx)
theta[dx<0] += math.pi
theta = (theta+math.pi/2)/twopi

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# Select catchments
selected = []
for th in np.linspace(0, 1, nsites):
    siteid = (np.abs(theta-th)).idxmin()
    selected.append(siteid)
    theta[siteid] = np.inf

sites = sites.loc[selected, :]
sites['number'] = range(1, len(sites)+1)

# Write data

for i, (siteid, row) in enumerate(sites.iterrows()):

    LOGGER.info('dealing with {0} ({1}/{2})'.format( \
        siteid, i, len(sites)))

    daily, _ = dset.get(siteid, 'rainfall_runoff', 'D')
    daily = daily.loc[start:end, cc]
    daily.index.name = 'day'

    # Replace missing with -9.999
    daily.fillna(-9.999, inplace=True)

    fd = os.path.join(fout, 'input_data_{0:02d}.csv'.format(i+1))
    comment = {\
        'comment': 'Rainfall runoff test data', \
        'siteid': siteid, \
        'coords': '{0:0.3f}, {1:0.3f}'.format(row['lon_outlet'], \
                            row['lat_outlet'])}
    csv.write_csv(daily, fd, comment, \
        source_file, compress=False, write_index=True)

fs = os.path.join(fout, 'sites.csv')
csv.write_csv(sites, fs, 'List of sites', \
    source_file, compress=False, write_index=True)

LOGGER.info('Process completed')

