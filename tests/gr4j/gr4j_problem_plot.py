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
#outputs = pygr.run(LOGGER, airgr.Precip.values, airgr.PotEvap.values, \
#                params.values, states, ord1, ord2, uh1, uh2, nuh1, nuh2)

# plot
plt.close('all')
fig, ax = plt.subplots()
idx = np.arange(1050, 1150)

ax.plot(airgr.loc[idx, 'Qsim'], color='indianred', label='airgr Qsim', lw=2)
#ax.plot(outputs.loc[idx, 'Q'], label='pygr')
#ax.plot(pyg.loc[idx, 'Q'], label='pygme')

tax = ax.twinx()
vmax = 60
tax.plot(vmax-airgr.loc[idx, 'Precip'], color='navy')
tax.plot(vmax-airgr.loc[idx, 'PotEvap'], color='green')
ytk = ['{0:0.0f}'.format(vmax-t) for t in tax.get_yticks()]
tax.set_yticklabels(ytk)
tax.set_ylim(0, vmax)

ax.plot([], [], color='navy', label='precip')
ax.plot([], [], color='green', label='pet')


ax.legend(loc=3)
ax.set_xlabel('Time step')
ax.set_ylabel('Runoff [mm/d]')
tax.set_ylabel('Climate inputs [mm/d]')
ptxt = ' '.join(['{0}={1:0.1f}'.format(k, v) for k, v in params.items()])
ax.set_title('GR4J simulation with\n{0}'.format(ptxt))


#tax = ax.twinx()
#tax.plot(pyg.loc[idx, 'Q']-outputs.loc[idx, 'Q'], 'r-', lw=0.8, label='airgr-pygme')
#tax.legend(loc=4)

fig.tight_layout()
fp = os.path.join(fout, 'gr4j_problem.png')
fig.savefig(fp)

LOGGER.info('Process completed')

