#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : jlerat
## Created : 2018-06-13 14:15:04.340754
## Comment : Run GR4J with problematic setup
##
## ------------------------------
import sys, os, re, json, math
import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

from hydrodiy.io import csv, iutils
import matplotlib.pyplot as plt

import gr4j

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------


#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

fdata = os.path.join(froot, '..', 'data')

fout = froot

basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------

fi = os.path.join(fdata, '606002_inputs.csv')
inputs, _ = csv.read_csv(fi, index_col=0, parse_dates=True)
inputs = inputs.iloc[:5000, :]

fp = os.path.join(fdata, '606002_params.csv')
params, _ = csv.read_csv(fp, index_col=0, parse_dates=True)
params = params.loc[:, ['X1', 'X2', 'X3', 'X4']].squeeze()

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# Uh ordinates
nord = 500
nuh1, ord1, nuh2, ord2 = gr4j.uh_ord(nord, params.X4)

uh1 = np.zeros(nord)
uh2 = np.zeros(nord)

states = [params[0]*0.5, params[2]*0.3]
outputs = gr4j.run(LOGGER, inputs.iloc[:, 0].values, inputs.iloc[:, 1], \
                params.values, states, ord1, ord2, uh1, uh2, nuh1, nuh2)

plt.close('all')
plt.plot(outputs.Q)
plt.show()

LOGGER.info('Process completed')

