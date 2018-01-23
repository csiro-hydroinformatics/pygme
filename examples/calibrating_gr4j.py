import os
import numpy as np

from pygme.calibration import ObjFunBCSSE
from pygme.models.gr4j import GR4J, CalibrationGR4J
from hydrodiy.stat import metrics

gr = GR4J()
warmup = 365*6

# Open data file
print('Reading data')
siteid = 1
filename = os.path.abspath(__file__)
FHERE = os.path.dirname(filename)
fts = '{0}/../tests/data/GR4J_timeseries_{1:02d}.csv'.format(FHERE, siteid)
data = np.loadtxt(fts, delimiter=',')

# Get input and obs data
inputs = data[:, [1, 2]]
inputs = np.ascontiguousarray(inputs, np.float64)
obs = data[:, 3]

# Define calibration period
idx_cal = np.arange(len(inputs))>=warmup
idx_cal = idx_cal & (obs >= 0)

# Calibrate
print('Calibrating GR4J')
calib = CalibrationGR4J(ObjFunBCSSE())
calparams, _, _ = calib.workflow(obs, inputs, ical=idx_cal)

# Get model parameters
gr = calib.model
params = gr.params.values

# Run model
print('Running GR4J and compute stats')
gr.inputs = inputs
gr.run()
sim = gr.outputs[:, 0]

# Mode statistics
bias = metrics.bias(obs, sim)
nse = metrics.nse(obs, sim)
print(' bias = {0:.3f}\n NSE  = {1:.3f}'.format(bias, nse))

