import sys, os, re
import unittest
from itertools import product as prod

from timeit import Timer
import time
import math

import logging

import numpy as np

from scipy.optimize import fmin, fmin_bfgs

from hydrodiy.stat.transform import BoxCox2
from hydrodiy.data.containers import Vector

from pygme.model import Model
from pygme.calibration import Calibration, CalibParamsVector
from pygme.calibration import ObjFunSSE, ObjFunBCSSE, ObjFunKGE

from dummy import Dummy, CalibrationDummy, ObjFunSSEargs

BC = BoxCox2()

# Set logger
LOGGER = logging.getLogger('pygme.Calibration')
fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
ft = logging.Formatter(fmt)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(ft)
LOGGER.addHandler(sh)


class ObjFunTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> ObjFunTestCase')

        nval = 1000
        obs = np.random.uniform(0., 1, size=nval)
        idx = np.random.choice(np.arange(nval), nval//100)
        obs[idx] = np.nan
        self.obs = obs

        sim = np.random.uniform(0., 1, size=nval)
        idx = np.random.choice(np.arange(nval), nval//100)
        sim[idx] = np.nan
        self.sim = sim


    def test_print(self):
        of = ObjFunBCSSE(0.2)
        print(of)

        of = ObjFunSSE()
        print(of)

        of = ObjFunKGE()
        print(of)


    def test_SSE(self):
        obs, sim = self.obs, self.sim
        idx = (~np.isnan(obs)) & (~np.isnan(sim))

        of = ObjFunSSE()
        value = of.compute(obs[idx], sim[idx])
        err = self.obs-self.sim
        expected = np.nansum(err*err)
        self.assertTrue(np.allclose(value, expected))

        value = of.compute(obs, sim)
        self.assertTrue(np.isnan(value))


    def test_KGE(self):
        of = ObjFunKGE()
        obs, sim = self.obs, self.sim
        idx = (~np.isnan(obs)) & (~np.isnan(sim))

        value = of.compute(obs[idx], sim[idx])

        obsok, simok = obs[idx], sim[idx]
        bias = np.mean(simok)/np.mean(obsok)
        rstd = np.std(simok)/np.std(obsok)
        corr = np.corrcoef(obsok, simok)[0, 1]
        expected = 1-math.sqrt((1-bias)**2+(1-rstd)**2+(1-corr)**2)
        self.assertTrue(np.allclose(value, expected))

        value = of.compute(obs, sim)
        self.assertTrue(np.isnan(value))


    def test_BCSSE(self):
        ''' test the BCSSE objfun '''
        obs, sim = self.obs, self.sim
        idx = (~np.isnan(obs)) & (~np.isnan(sim))

        for lam, nu in prod([0.1, 0.5, 1., 2], [1e-4, 1e-2, 1]):
            of = ObjFunBCSSE(lam, nu)
            value = of.compute(obs[idx], sim[idx])

            BC.lam = lam
            BC.nu = nu
            err = BC.forward(self.obs)-BC.forward(self.sim)
            expected = np.nansum(err*err)

            self.assertTrue(np.isclose(value, expected))

            value = of.compute(obs, sim)
            self.assertTrue(np.isnan(value))


class CalibParamsVectorTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> CalibParamsVectorTestCase')

        config = Vector([])
        nval = 10
        params = Vector(['X{0}'.format(k) for k in range(1, nval+1)],
                    defaults=np.ones(nval), mins=np.zeros(nval), \
                    maxs=np.ones(nval)*5)
        states = Vector(['S{0}'.format(k) for k in range(1, 3)])
        self.model = Model('test', config, params, states, 2, 2)


    def test_default(self):
        ''' Test setting default values '''
        calparams = CalibParamsVector(self.model)

        self.assertTrue(np.all([s1==s2 for s1, s2 in zip(calparams.names, \
                                    self.model.params.names)]))

        self.assertTrue(np.allclose(calparams.defaults, \
                                    self.model.params.defaults))


    def test_errors_infinite(self):
        ''' Test errors for finite values in calibrated params '''
        nval = self.model.params.nval
        cp = Vector(['X{0}'.format(k) for k in range(1, nval+1)])
        try:
            calparams = CalibParamsVector(self.model, cp)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected no infinite'))
        else:
            raise ValueError('Problem with error handling')


    def test_errors_funs(self):
        ''' Test errors related to trans2true and true2trans '''
        nval = self.model.params.nval
        cp = Vector(['X{0}'.format(k) for k in range(1, nval+1)])
        cp = Vector(['tX{0}'.format(k) for k in range(1, nval+1)],\
                    defaults=[0]*nval, mins=[-1]*nval, maxs=[1]*nval)

        fun1 = lambda x: 'string1'
        fun2 = lambda x: 'string2'
        try:
            calparams = CalibParamsVector(self.model, cp, fun1, fun2)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Problem with trans2true for'))
        else:
            raise ValueError('Problem with error handling')

        fun = lambda x: np.column_stack([x, x])
        try:
            calparams = CalibParamsVector(self.model, cp, fun, fun)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Problem with trans2true for'))
        else:
            raise ValueError('Problem with error handling')


    def test_identity(self):
        nval = self.model.params.nval
        cp = Vector(['tX{0}'.format(k) for k in range(1, nval+1)],\
                    defaults=[0]*nval, mins=[-1]*nval, maxs=[1]*nval)

        calparams = CalibParamsVector(self.model, cp)

        for i in range(10):
            val = np.random.uniform(0, 1, nval)
            calparams.values = val
            self.assertTrue(np.allclose(self.model.params.values, val))

            val = np.random.uniform(0, 1, nval)
            calparams.truevalues = val
            self.assertTrue(np.allclose(calparams.values, val))


    def test_common_transform(self):
        nval = self.model.params.nval
        cp = Vector(['tX{0}'.format(k) for k in range(1, nval+1)],\
                    defaults=[0]*nval, mins=[-1]*nval, maxs=[1]*nval)

        for i, trans in enumerate(['exp', 'sinh']):
            calparams = CalibParamsVector(self.model, cp, trans2true=trans)
            if i == 0:
                trans2true = np.exp
                true2trans = np.log
            elif i == 1:
                trans2true = np.sinh
                true2trans = np.arcsinh

            for i in range(10):
                val = np.random.uniform(0, 1, nval)
                calparams.values = val
                self.assertTrue(np.allclose(calparams.truevalues, \
                                                trans2true(val)))
                self.assertTrue(np.allclose(self.model.params.values, \
                                                trans2true(val)))

                val = np.random.uniform(math.exp(-1), 1, nval)
                calparams.truevalues = val
                self.assertTrue(np.allclose(calparams.values, \
                                                true2trans(val)))

    def test_fixed(self):
        nval = self.model.params.nval
        cp = Vector(['tX{0}'.format(k) for k in range(1, nval+1)],\
                    defaults=[0]*nval, mins=[-5]*nval, maxs=[5]*nval)

        # Choose a fixed value below the max value
        x1 = 4
        fixed = {'X1':x1}

        calparams = CalibParamsVector(self.model, cp, fixed=fixed)

        for i in range(10):
            val = np.random.uniform(0, 1, nval)
            calparams.values = val
            val2 = val.copy()
            val2[0] = x1
            self.assertTrue(np.allclose(self.model.params.values, val2))

            val = np.random.uniform(0, 1, nval)
            calparams.truevalues = val
            val2 = val.copy()
            val2[0] = x1
            self.assertTrue(np.allclose(calparams.truevalues, val2))
            self.assertTrue(np.allclose(calparams.values, val2))



class CalibrationTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> CalibrationTestCase')


    def test_calibration_instance_print(self):
        ''' Test printing of calibration object '''
        dum = Dummy()
        inputs = np.random.uniform(0, 1, (1000, 2))
        dum.allocate(inputs)

        params = [0.5, 10.]
        dum.params.values = params
        dum.run()
        obs = dum.outputs[:, 0].copy()

        calib = CalibrationDummy(warmup=10)
        calib.allocate(obs, inputs)

        str = '{0}'.format(calib)


    def test_calibration_errors(self):
        ''' Test calibration errors '''
        inputs = np.random.uniform(0, 1, (1000, 2))
        obs = np.random.uniform(0, 1, 1000)

        cp = Vector(['tX1', 'tX2'], mins=[-10]*2, maxs=[10]*2, \
                defaults=[1, 0])
        calparams = CalibParamsVector(Dummy(), cp, trans2true='exp')
        calib = Calibration(calparams)

        try:
            plib = calib.paramslib
        except ValueError as err:
            self.assertTrue(str(err).startswith(\
                    'Trying to access paramslib, but '))
        else:
            raise ValueError('Problem with error handling')

        try:
            calib.ical = obs==obs
        except ValueError as err:
            self.assertTrue(str(err).startswith('Trying to get obs, but '))
        else:
            raise ValueError('Problem with error handling')


    def test_explore(self):
        ''' Test explore function '''
        inputs = np.random.uniform(0, 1, (1000, 2))
        dum = Dummy()
        dum.allocate(inputs, 2)
        dum.initialise()

        calib = CalibrationDummy(warmup=10)
        params = calib.paramslib[0, :]
        dum.params.values = params.copy()
        dum.run()

        obs = dum.outputs[:, 0]
        calib.allocate(obs, inputs)
        calib.ical = np.arange(obs.shape[0])>10

        start, explo, explo_ofun = calib.explore()
        self.assertTrue(np.allclose(start, params))


    def test_explore_fit(self):
        ''' Test explore and fit functions '''
        inputs = np.random.exponential(1, (100, 2))
        dum = Dummy()
        dum.allocate(inputs, 2)
        dum.initialise()

        calib = CalibrationDummy(warmup=10)
        params = calib.paramslib[0, :]+0.1
        dum.params.values = params
        dum.run()

        obs = dum.outputs[:, 0]
        calib = CalibrationDummy(warmup=10)
        calib.allocate(obs, inputs)
        calib.ical = np.arange(10, obs.shape[0])

        start, _, _ = calib.explore()
        final, _, _ = calib.fit(iprint=10,
                                    maxfun=100000, ftol=1e-8)
        ck = np.allclose(calib.model.params.values, params, \
                            atol=1e-5, rtol=0.)
        self.assertTrue(ck)


    def test_explore_fit_args(self):
        ''' Test passing arguments to objective function '''
        inputs = np.random.exponential(1, (100, 2))
        dum = Dummy()
        dum.allocate(inputs, 2)
        dum.initialise()

        calib = CalibrationDummy(warmup=10)
        params = calib.paramslib[0, :]+0.1
        dum.params.values = params
        dum.run()

        obs = dum.outputs[:, 0]

        ical = np.arange(10, obs.shape[0])
        kwargs = {'lam':1.0, 'idx':np.arange(len(ical))}
        calib = CalibrationDummy(objfun=ObjFunSSEargs(), \
                    warmup=10, \
                    objfun_kwargs=kwargs)
        calib.allocate(obs, inputs)
        calib.ical = ical

        start, _, _ = calib.explore()
        final, _, _ = calib.fit(iprint=10,
                                    maxfun=100000, ftol=1e-8)
        ck = np.allclose(calib.model.params.values, params, \
                            atol=1e-3, rtol=0.)

        self.assertTrue(ck)


    def test_explore_fit_fixed(self):
        ''' Test calibration with fixed parameters '''
        inputs = np.random.exponential(1, (100, 2))
        dum = Dummy()
        dum.allocate(inputs, 2)
        dum.initialise()

        calib = CalibrationDummy(warmup=10)
        params = calib.paramslib[0, :]+0.1
        dum.params.values = params
        dum.run()

        obs = dum.outputs[:, 0]

        # Test error
        fixed = {'X10':params[0]+3}
        try:
            calib = CalibrationDummy(warmup=10, fixed=fixed)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected names '+\
                                'of fixed parameters'))
        else:
            raise ValueError('Problem with error handling')

        fixed = {'X1':params[0]+3}
        calib = CalibrationDummy(warmup=10, fixed=fixed)
        calib.allocate(obs, inputs)
        calib.ical = np.arange(10, obs.shape[0])

        start, _, _ = calib.explore()
        final, _, _ = calib.fit(iprint=10,
                                    maxfun=100000, ftol=1e-8)

        self.assertEqual(fixed, calib.fixed)
        self.assertTrue(np.allclose(fixed['X1'], start[0]))
        self.assertTrue(np.allclose(fixed['X1'], final[0]))
        self.assertTrue(np.allclose(fixed['X1'], \
                            calib.model.params.values[0]))


    def test_workflow(self):
        ''' Test calibration workflow (i.e. explore+fit) '''
        inputs = np.random.exponential(1, (100, 2))
        dum = Dummy()
        dum.allocate(inputs, 2)

        calib = CalibrationDummy(warmup=10)
        params = calib.paramslib[0, :]+0.1
        dum.params.values = params
        dum.run()
        obs = dum.outputs[:, 0]

        # Instanciate a new calib obj
        calib = CalibrationDummy(warmup=10)

        # Check parameter are not close at the beginning
        ck = ~np.allclose(calib.model.params.values, params)
        self.assertTrue(ck)

        # Run calibration
        ical = np.arange(10, obs.shape[0])
        calib.workflow(obs, inputs, ical, iprint=0,
                maxfun=100000, ftol=1e-8)

        # Test parameters at the end
        ck = np.allclose(calib.model.params.values, params, \
                            atol=1e-5, rtol=0.)
        self.assertTrue(ck)


    def test_customised_objfun(self):
        ''' Test customised objective function '''
        inputs = np.random.exponential(1, (100, 2))
        dum = Dummy()
        dum.allocate(inputs, 2)

        # Run model
        calib = CalibrationDummy(warmup=10)
        params = calib.paramslib[0, :]+0.1
        dum.params.values = params
        dum.run()
        obs = dum.outputs[:, 0]

        # Define a customized objective function
        objfun = ObjFunBCSSE(lam=0.8, nu=1e-5)

        # Instanciate a new calib obj and applies objfun
        calib = CalibrationDummy(warmup=10, objfun=objfun)

        # Check parameter are not close at the beginning
        ck = ~np.allclose(calib.model.params.values, params)
        self.assertTrue(ck)

        # Run calibration
        ical = np.arange(10, obs.shape[0])
        calib.workflow(obs, inputs, ical, iprint=0,
                maxfun=100000, ftol=1e-8)

        # Test parameters at the end
        ck = np.allclose(calib.model.params.values, params, atol=1e-3)
        self.assertTrue(ck)


    def test_optimizers(self):
        ''' Test a range of optimizer from scipy '''
        inputs = np.minimum(np.random.exponential(1, (100, 2)), 10)
        dum = Dummy()
        dum.allocate(inputs, 2)
        dum.initialise()

        calib = CalibrationDummy(warmup=10)
        params = calib.paramslib[0, :]+0.1
        dum.params.values = params
        dum.run()
        obs = dum.outputs[:, 0]

        calib = CalibrationDummy(objfun=ObjFunSSE(), \
                    warmup=10)
        calib.allocate(obs, inputs)
        ical = np.arange(10, obs.shape[0])
        calib.ical = ical

        start, _, _ = calib.explore()

        for iopt, opt in enumerate([fmin, fmin_bfgs]):
            final, _, _ = calib.fit(start=start, iprint=10, optimizer=opt)
            ck = np.allclose(calib.model.params.values, params, \
                    atol=5e-3, rtol=0.)
            if not ck:
                print(('Failing optimizer test {0} '+\
                    'expected params={1}, got {2}').format(\
                        iopt+1, \
                        ' '.join(list(np.round(params, 2).astype(str))), \
                        ' '.join(list(\
                        np.round(calib.model.params.values, 2).astype(str)))
                ))

            self.assertTrue(ck)


if __name__ == "__main__":
    unittest.main()
