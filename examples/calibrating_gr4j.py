import os, sys
import numpy as np
import pandas as pd

from pygme.calibration import ObjFunBCSSE
from pygme.models.gr4j import GR4J, CalibrationGR4J

from hydrodiy.stat import metrics

# Add access to test data
source_file = os.path.abspath(__file__)
froot = os.path.join(os.path.dirname(source_file), '..')
sys.path.append(os.path.join(froot, 'tests'))
import testdata

# initialise GR4J
gr = GR4J()
warmup = 365*6

# Get data
i = 10
data = testdata.read('GR4J_timeseries_{0:02d}.csv'.format(i+1), \
                        source='output', has_dates=False)
inputs = np.ascontiguousarray(\
                data.loc[:, ['Precip', 'PotEvap']], \
                np.float64)

obs = data.Qobs.values

# Define calibration period
print('Define calibration period')
idx_cal = np.arange(len(inputs))>=warmup
idx_cal = idx_cal & (obs >= 0)

# Calibrate
print('Calibrating GR4J')
calib = CalibrationGR4J(ObjFunBCSSE())
calib.workflow(obs, inputs, ical=idx_cal)

params = gr.params.values
print('Calibrated parameters:')
print('\t' + '\n\t'.join(['X{0} = {1:0.1f}'.format(i+1, params[i]) \
                for i in range(4)]))

# Run model
print('Running GR4J and compute stats')
gr = calib.model
gr.inputs = inputs
gr.run()
sim = gr.outputs[:, 0]

# Statistics over calibration period
obsc, simc = obs[idx_cal], sim[idx_cal]
bias = metrics.bias(obsc, simc)
nse = metrics.nse(obsc, simc)
print(' bias = {0:.3f}\n NSE  = {1:.3f}'.format(bias, nse))

