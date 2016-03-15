import os
import re
import math
import unittest

import time

import numpy as np
import pandas as pd

from calendar import month_abbr as month

from pygme.models.basics import MonthlyPattern, NodeModel, SinusPattern
from pygme.data import Matrix
from pygme.calibration import Calibration, sse_qreg50, sse_qreg90, sse_qreg10

import c_pygme_models_utils as utils

import matplotlib.pyplot as plt
from hyplot import putils


class MonthlyPatternTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> MonthlyPatternTestCase')

    def test_print(self):
        dm = MonthlyPattern()
        str_dm = '%s' % dm

    def test_monthlypattern1(self):
        dm = MonthlyPattern()

        for i in range(12):
            dm.config[month[i+1]] = 1. + np.sin(float(i)/12 * np.pi) * 10.

        dt = pd.date_range('2010-01-01', '2020-12-31')
        nval = len(dt)

        # Maximum extraction
        inputs = np.zeros((nval, 0))
        dm.allocate(inputs, 1)

        dm.initialise()
        dm.run()
        extrac = pd.Series(dm.outputs[:, 0], index=dt)
        extracm = extrac.resample('MS', 'sum')
        v1 = extracm.groupby(extracm.index.month).mean().values
        v2 = extracm.groupby(extracm.index.month).std().values
        v2[np.isnan(v2)] = 0.

        ck = np.allclose(v1, dm.config.data)
        self.assertTrue(ck)

        ck = np.all(v2 < 1e-6)
        self.assertTrue(ck)


class NodeModelTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> NodeModelTestCase')

    def test_print(self):
        nd = NodeModel(4, 2)
        str_nd = '%s' % nd

    def test_nodemodel_sum(self):
        ninputs = 4
        nd = NodeModel(ninputs)

        nval = 100
        inputs = np.random.uniform(0, 1, (nval, ninputs))
        nd.allocate(inputs)

        nd.run()
        ss = inputs.sum(axis=1)
        ck = np.allclose(ss, nd.outputs[:, 0])

        self.assertTrue(ck)

    def test_nodemodel_split(self):
        ninputs = 4
        noutputs = 5
        nd = NodeModel(ninputs, noutputs)

        nval = 100
        inputs = np.random.uniform(0, 1, (nval, ninputs))
        nd.allocate(inputs, noutputs)

        nd.run()
        ss = inputs.sum(axis=1)
        p = np.diag([1./noutputs] * noutputs)
        o = np.dot(np.repeat(ss.reshape((nval, 1)), noutputs, axis=1), p)
        ck = np.allclose(o, nd.outputs)
        self.assertTrue(ck)

        nd.params = range(1, noutputs+1)
        nd.run()
        ss = inputs.sum(axis=1)
        p = np.diag(nd.params/np.sum(nd.params))
        o = np.dot(np.repeat(ss.reshape((nval, 1)), noutputs, axis=1), p)
        ck = np.allclose(o, nd.outputs)
        self.assertTrue(ck)

    def test_nodemodel_clip(self):
        dm = NodeModel()

        dm.config['min'] = 1.
        dm.config['max'] = 5.

        # Maximum extraction
        nval = 1000
        inputs = np.random.uniform(0, 10, (nval, 1))
        dm.allocate(inputs, 1)

        dm.initialise()
        dm.run()

        clip = np.clip(inputs[:,0], dm.config['min'], dm.config['max'])
        ck = np.allclose(dm.outputs[:, 0], clip)

        self.assertTrue(ck)



class SinusPatternTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> SinusPatternTestCase')

        self.FHERE = os.path.dirname(os.path.abspath(__file__))

    def test_print(self):
        sp = SinusPattern()
        str_sp = '%s' % sp

    def test_sinuspattern_run(self):

        params = [0., 1., 0., 3.]
        vmin = 0.
        vmax = 10.

        sp = SinusPattern()

        nval = 365
        sp.allocate(np.zeros((nval, 0)))
        sp.config['vmin'] = vmin
        sp.config['vmax'] = vmax
        sp.config['year_monthstart'] = 2.
        sp.initialise()
        sp.params = params
        sp.run()

        u = np.arange(1, 366).astype(float)
        x = (np.sin((u/365-params[2])*2*np.pi) + 1)/2
        nue = math.sinh(params[3])
        y = (np.exp(nue*x)-1)/(np.exp(nue) - 1)

        A = vmin + params[0] *(vmax-vmin)
        B = A + params[1] *(vmax-A)
        y = A + (B-A) * y
        ck = np.allclose(sp.outputs[:, 0], y)

        self.assertTrue(ck)


    def test_sinuspattern_runlong(self):

        params = [0.1, 0.9, 0., 1.5]
        vmin = -2.
        vmax = 100.

        sp = SinusPattern()

        nval = 1000
        dt = pd.date_range('2001-01-01', freq='D', periods=nval)
        doy = np.array([d.timetuple().tm_yday for d in dt])

        sp.allocate(np.zeros((nval, 0)))
        sp.config['vmin'] = vmin
        sp.config['vmax'] = vmax
        sp.initialise()
        sp.params = params
        sp.run()

        x = (np.sin((doy.astype(float)/365-params[2])*2*np.pi) + 1)/2
        nue = math.sinh(params[3])
        y = (np.exp(nue*x)-1)/(np.exp(nue) - 1)

        A = vmin + params[0] *(vmax-vmin)
        B = A + params[1] *(vmax-A)
        y = A + (B-A) * y

        ck = np.allclose(sp.outputs[:, 0], y)
        self.assertTrue(ck)

    def test_sinuspattern_runcumulative(self):

        params = [0.1, 0.9, 0., 1.5]
        vmin = -2.
        vmax = 100.

        sp = SinusPattern()

        nval = 1000
        dt = pd.date_range('2001-01-01', freq='D', periods=nval)
        doy = np.array([d.timetuple().tm_yday for d in dt])

        sp.allocate(np.zeros((nval, 0)))
        sp.config['vmin'] = vmin
        sp.config['vmax'] = vmax
        sp.config['year_monthstart'] = 2
        sp.config['is_cumulative'] = 1
        sp.initialise()
        sp.params = params
        sp.run()

        x = (np.sin((doy.astype(float)/365-params[2])*2*np.pi) + 1)/2
        nue = math.sinh(params[3])
        y = (np.exp(nue*x)-1)/(np.exp(nue) - 1)

        A = vmin + params[0] *(vmax-vmin)
        B = A + params[1] *(vmax-A)
        y = A + (B-A) * y
        yc = y*0.
        utils.accumulate(sp.config['startdate'], sp.config['year_monthstart'], y, yc)

        ck = np.allclose(sp.outputs[:, 0], yc)
        self.assertTrue(ck)


    def test_sinuspattern_calibrate(self):

        mstart = 7

        i = 6
        fts = '{0}/data/GR4J_timeseries_{1:02d}.csv'.format( \
                self.FHERE, i+1)
        data = np.loadtxt(fts, delimiter=',')

        values = np.zeros(len(data))
        inp = data[:, 3].copy()
        inp[inp<0] = np.nan
        utils.accumulate(data[0, 0], mstart, inp, values)
        obs = Matrix.from_data('obs', values)

        inputs = Matrix.from_dims('inputs', obs.nval, 0)

        sp = SinusPattern()
        sp.config['startdate'] = data[0, 0]
        sp.config['vmax'] = np.nanmax(obs.data)
        sp.config['is_cumulative'] = 1
        sp.config['year_monthstart'] = mstart
        sp.allocate(inputs)

        cal = Calibration(sp, 4, errfun=sse_qreg50)
        final, out, outfun = cal.run(obs, inputs, ftol=1e-5)

        #plt.close('all')
        #fig, ax = plt.subplots()
        #ax.plot(out[:, 0])
        #ax.plot(values)
        #fig.savefig(os.path.join(self.FHERE, 'tmp.png'))
        #import pdb; pdb.set_trace()

