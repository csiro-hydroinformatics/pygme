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

from pygme.models.gr4j import GR4J

import pygr

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

nval = 2000

fi = os.path.join(fdata, '606002_inputs.csv')
inputs, _ = csv.read_csv(fi, index_col=0, parse_dates=True)
inputs = inputs.iloc[:nval, :]

fp = os.path.join(fdata, '606002_params.csv')
params, _ = csv.read_csv(fp, index_col=0, parse_dates=True)
params = params.loc[:, ['X1', 'X2', 'X3', 'X4']].squeeze()

fd = os.path.join(fdata, '..', 'output_data', 'GR4J_timeseries_606002.csv')
airgr, _ = csv.read_csv(fd)
airgr.columns = [re.sub('\"', '', cn) for cn in airgr.columns]
airgr = airgr.loc[:nval, :]

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------
# Initialise
states = [params.X1/2, params.X3*0.3]

# Pygme
mod = GR4J()
mod.allocate(np.ascontiguousarray(inputs.values), 9)
for key, value in params.items():
    mod.params[key] = value

mod.initialise(states=states)
mod.run()
pyg = pd.DataFrame(mod.outputs, columns=mod.outputs_names)

# pygr
# ... Uh ordinates
nord = 100
nuh1, ord1, nuh2, ord2 = pygr.uh_ord(nord, params.X4)
uh1 = np.zeros(nord)
uh2 = np.zeros(nord)
# .. simulation
outputs = pygr.run(LOGGER, inputs.iloc[:, 0].values, inputs.iloc[:, 1], \
                params.values, states, ord1, ord2, uh1, uh2, nuh1, nuh2)

# plot
plt.close('all')
fig, ax = plt.subplots()
#ax.plot(airgr.Prod, label='airgr')
#ax.plot(pyg.S, label='pygme')
#ax.plot(outputs.S, label='pygr')
ax.legend()
plt.show()

LOGGER.info('Process completed')

