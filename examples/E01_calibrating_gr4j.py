import os
import numpy as np

from pygme import calibration
from pygme.models.gr4j import GR4J, CalibrationGR4J

gr = GR4J()
warmup = 365*6

# Open data file
id = 1
filename = os.path.abspath(__file__)
FHERE = os.path.dirname(filename)
fts = '{0}/../tests/data/GR4J_timeseries_{1:02d}.csv'.format(FHERE, id)
data = np.loadtxt(fts, delimiter=',')

# Get input and obs data
inputs = data[:, [1, 2]]
inputs = np.ascontiguousarray(inputs, np.float64)
obs = data[:, 3]

# Define calibration period
idx_cal = np.arange(len(inputs))>=warmup
idx_cal = idx_cal & (obs >= 0)

# Calibrate
calib = CalibrationGR4J()
calib.errfun = calibration.ssqe_bias
calparams, _, _ = calib.fullfit(obs, inputs)

# Get model parameters
gr = calib.model
params = gr.params.data

# run model
gr.inputs.data = inputs
gr.run()
sim = gr.outputs.data[:, 0]

# Mode statistics
meano = np.mean(obs)
err = obs-sim
err0 = obs-meano
bias = np.mean(err)/meano * 100
nse = (1-np.sum(err*err)/np.sum(err0*err0))* 100
print('Calibration stats\n\tbias = {0:.2f}%\n\tNSE = {1:.1f}%'.format(bias, nse))

