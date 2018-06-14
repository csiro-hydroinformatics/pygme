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

nval = 1500

fp = os.path.join(fdata, '606002_params.csv')
params, _ = csv.read_csv(fp, index_col=0, parse_dates=True)
params = params.loc[:, ['X1', 'X2', 'X3', 'X4']].squeeze()
#params.X3 = 2

#LOGGER.info('Running airgr in R')
#cmd = 'Rscript {0}/get_sim_airGR_606002.r {1}'.format(os.path.join(fout, '..'), params.X3)
#os.system(cmd)

LOGGER.info('Read airgr outputs')
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
mod.allocate(np.ascontiguousarray(airgr.loc[:, ['Precip', 'PotEvap']].values), 9)
for key, value in params.items():
    mod.params[key] = value

pyg = {}
for X3 in [5, 6, 7, 10, 15, 20]:
    mod.params.X3 = X3
    mod.initialise(states=states)
    mod.run()
    pyg[X3] = pd.DataFrame(mod.outputs.copy(), columns=mod.outputs_names)

# plot
plt.close('all')
fig, ax = plt.subplots()
idx = np.arange(1110, 1140)

X3min = np.min(np.array(list(pyg.keys())))

tax = ax.twinx()

tax.plot(10-0.5*pyg[X3min].loc[idx, 'PR'], 'k--', label='PR')

for X3 in pyg:
    ax.plot(pyg[X3].loc[idx, 'Q'], label='X3={0:0.1f}'.format(X3))

ax.legend(loc=3)
ax.set_xlabel('Time step')
ax.set_ylabel('Runoff [mm/d]')
ax.set_title('GR4J simulations with varying X3')

fig.set_size_inches((15, 10))
fig.tight_layout()
fp = os.path.join(fout, 'gr4j_X3_explore.png')
fig.savefig(fp)

LOGGER.info('Process completed')

