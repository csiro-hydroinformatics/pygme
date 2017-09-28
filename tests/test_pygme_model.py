import os
import re
import unittest

from timeit import Timer
import time

import numpy as np
np.seterr(all='print')

from hydrodiy.data.containers import Vector
from pygme.model import Model, NUHMAXLENGTH, UH, UHNAMES, ParamsVector
from dummy import Dummy, MassiveDummy


class UHTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> UHTestCase')


    def test_init(self):
        for nm in UHNAMES:
            u = UH(nm)
            s = str(u)

        try:
            u = UH('pignouf')
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected UH name'))
        else:
            raise ValueError('Problem with error handling')


    def test_set_param(self):
        for nm in UHNAMES:
            u = UH(nm)
            for p in np.linspace(0, 100, 500):
                u.param = p
                ck = np.allclose(np.sum(u.ord[:u.nuh]), 1., atol=1e-8)
                self.assertTrue(ck)
                self.assertTrue(np.allclose(u.states, 0.))


    def test_set_states(self):
        for nm in UHNAMES:
            u = UH(nm)
            u.states[:10] = 10.
            u.param = 20
            self.assertTrue(np.allclose(u.states[:u.nuh], 0.))


    def test_initialise(self):
        for nm in UHNAMES:
            u = UH(nm)
            u.param = 5.5
            nuh = u.nuh

            u.initialise()
            self.assertTrue(np.allclose(u.states, np.zeros(nuh)))

            s = np.random.uniform(size=nuh)
            u.initialise(s)
            self.assertTrue(np.allclose(u.states, s))

            try:
                u.initialise(s[:2])
            except ValueError as err:
                self.assertTrue(str(err).startswith('Expected state vector'))
            else:
                raise ValueError('Problem in error handling')


    def test_uh_lag(self):
        u = UH('lag')

        u.param = 5.5
        o = np.zeros(u.nuhmax)
        o[5:7] = 0.5
        self.assertTrue(np.allclose(u.ord, o))

        u.param = 5
        o = np.zeros(u.nuhmax)
        o[5] = 1
        self.assertTrue(np.allclose(u.ord, o))


    def test_uh_triangle(self):
        u = UH('triangle')

        u.param = 2.5
        o = [0.08, 0.24, 0.36, 0.24, 0.08, 0.]
        self.assertTrue(np.allclose(u.ord[:len(o)], o))

        u.param = 9.2
        o = [0.005907372, 0.017722117, 0.029536862, 0.041351607, 0.053166352, \
                0.064981096, 0.076795841, 0.088610586, 0.100425331, \
                0.104678639, 0.093336484, 0.081521739, 0.069706994, \
                0.057892250, 0.046077505, 0.034262760, 0.022448015, \
                0.010633270, 0.000945180, 0.000000000]
        self.assertTrue(np.allclose(u.ord[:len(o)], o))


    def test_uh_flat(self):
        u = UH('flat')

        u.param = 2.5
        o = [0.4, 0.4, 0.2]
        self.assertTrue(np.allclose(u.ord[:len(o)], o))

        u.param = 9.2
        o = [0.108695652]*9+[0.021739130, 0.]
        self.assertTrue(np.allclose(u.ord[:len(o)], o))


    def test_uh_gr4j_ssh1_daily(self):
        u = UH('gr4j_ss1_daily')

        u.param = 2.5
        o = [0.10119288512539, 0.47124051711456, 0.42756659776005, 0.]
        self.assertTrue(np.allclose(u.ord[:len(o)], o))

        u.param = 1.3
        o = [0.51896924219351, 0.48103075780649, 0.]
        self.assertTrue(np.allclose(u.ord[:len(o)], o))


    def test_uh_gr4j_ssh2_daily(self):
        u = UH('gr4j_ss2_daily')
        u.param = 2.5

        o = [0.05059644256269, 0.23562025855728, 0.42756659776005, \
                            0.23562025855728, 0.05059644256269]
        self.assertTrue(np.allclose(u.ord[:len(o)], o))

        u.param = 1.3
        o = [0.25948462109675, 0.66815684654371, 0.07235853235954, 0.]
        self.assertTrue(np.allclose(u.ord[:len(o)], o))


class ParamsVectorTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> ParamsVectorTestCase')


    def test_init(self):
        vect = Vector(['X{0}'.format(k) for k in range(10)])
        uhs = [UH('lag', 3), UH('lag', 6), UH('triangle', 8)]
        pv = ParamsVector(vect, uhs)

        for k in range(len(uhs)):
            self.assertTrue(np.allclose(pv.uhs[k].param, 0.))
            self.assertTrue(np.allclose(pv.uhs[k].nuh, 1))

            ordi = np.zeros(pv.uhs[k].nuhmax)
            ordi[0] = 1
            self.assertTrue(np.allclose(pv.uhs[k].ord, ordi))

            states = np.zeros(pv.uhs[k].nuhmax)
            self.assertTrue(np.allclose(pv.uhs[k].states, states))


    def test_error_init(self):
        vect = Vector(['X{0}'.format(k) for k in range(10)])
        uhs = [UH('lag', 3), UH('lag', 13), UH('triangle', 8)]
        try:
            pv = ParamsVector(vect, uhs)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected uhs[1].iparam in '))
        else:
            raise ValueError('Problem with error handling')


    def test_set_params(self):
        vect = Vector(['X{0}'.format(k) for k in range(10)])
        uhs = [UH('lag', 3), UH('lag', 6), UH('triangle', 8)]
        pv = ParamsVector(vect, uhs)

        # Set params
        pv['X3'] = 10
        pv['X6'] = 2.5
        pv['X8'] = 5

        zero = np.zeros(uhs[0].nuhmax)

        o = zero.copy()
        o[10] = 1
        self.assertTrue(np.allclose(pv.uhs[0].ord, o))

        o = zero.copy()
        o[2:4] = 0.5
        self.assertTrue(np.allclose(pv.uhs[1].ord, o))



class ModelTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> ModelTestCase')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


    def test_print(self):
        dum = Dummy()
        str = '{0}'.format(dum)


    def test_allocate(self):
        dum = Dummy()
        inputs = np.random.uniform(0, 1, (1000, 2))

        try:
            n = dum.ntimesteps
        except ValueError as err:
            self.assertTrue(str(err).startswith('Inputs are not'))
        else:
            raise ValueError('Problem with error handling')

        dum.allocate(inputs)
        self.assertTrue(dum.ninputs, 2)
        self.assertTrue(dum.ntimesteps, 1000)
        self.assertTrue(np.allclose(dum.inputs.shape, (1000, 2)))


    def test_set_params(self):
        params = [0.5, 10., 0.1]
        dum = Dummy()
        dum.params.values = params
        self.assertTrue(np.allclose(dum.params.values, params))


    def test_initialise(self):
        dum = Dummy()
        states = [5, 6, 7]
        dum.initialise(states)
        self.assertTrue(np.allclose(dum.states.values, states))


    def test_set_inputs(self):
        nval = 100
        inputs1 = np.random.uniform(0, 1, (nval, 2))
        inputs2 = np.random.uniform(0, 1, (nval, 2))
        dum = Dummy()
        dum.inputs = inputs1
        self.assertTrue(dum.ntimesteps, nval)
        self.assertTrue(np.allclose(dum.inputs, inputs1))

        dum.inputs = inputs2
        self.assertTrue(dum.ntimesteps, nval)
        self.assertTrue(np.allclose(dum.inputs, inputs2))

        # Change inputs
        nval = 10
        inputs3 = np.random.uniform(0, 1, (nval, 2))
        dum.inputs = inputs3
        self.assertTrue(dum.ntimesteps, nval)


    def test_run(self):
        nval = 100
        inputs = np.random.uniform(0, 1, (nval, 2))
        params = [0.5, 10., 0.]
        dum = Dummy()
        dum.allocate(inputs, 3)
        dum.params.values = params
        dum.config['continuous'] = 1

        states = np.array([10., 0., 0.])
        dum.initialise(states=states)

        dum.index_start = 0
        dum.index_end = nval-1
        dum.run()

        expected = params[0] + params[1] * inputs
        expected = np.cumsum(expected, 0)
        expected = expected + states[:2][None, :]
        ck = np.allclose(expected, dum.outputs[:, :2])
        self.assertTrue(ck)


    def test_inputs(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10., 0.5]

        dum = Dummy()
        dum.inputs = inputs
        dum.params.values = params
        dum.initialise(states=[10, 5, 0])
        dum.config.values = [10]

        dum2 = dum.clone()
        dum2.allocate(inputs)

        d1 = dum.inputs
        d2 = dum2.inputs

        self.assertTrue(np.allclose(d1, d2))

        # Check that inputs were copied and not pointing to same object
        d2[0, 0] += 1
        self.assertTrue(np.allclose(d1[0, 0]+1, d2[0, 0]))


    def test_uh(self):
        dum = Dummy()
        inputs = np.random.uniform(0, 1, (10, 2))
        dum.allocate(inputs, 2)
        dum.params.values = np.zeros(3)

        uh = [0.25]*4 + [0.] * (len(dum.uh1.values)-4)
        uh = np.array(uh)
        self.assertTrue(np.allclose(dum.uh1.values, uh))

        dum.params.values = [1., 2., 0.4]
        self.assertTrue(np.allclose(dum.uh1.values[:4], 0.25))


    def test_run_default(self):
        dum = MassiveDummy()
        dum.params.values = 0.
        inputs = np.random.uniform(0, 1, (1000, 2))
        dum.allocate(inputs)
        dum.initialise(states=[0.])

        dum.index_start = 0
        dum.index_end = len(inputs)-1
        dum.run()


    def test_allocate_dummy(self):
        dum = Dummy()
        nval = 1000
        ninputs = 2
        dum.inputs = np.random.uniform(0, 1, (nval, ninputs))
        nts = dum.ntimesteps

        self.assertTrue(dum.params.nval == 3)
        self.assertTrue(dum.states.nval == 3)
        self.assertTrue(dum.uh1.nval == NUHMAXLENGTH)
        self.assertTrue(dum.inputs.shape  == (nts, 2))
        self.assertTrue(dum.outputs.shape  == (nts, 1))


    def test_run_startend(self):
        dum = Dummy()
        nval = 1000
        ninputs = 2
        inputs = np.random.uniform(0, 1, (nval, ninputs))
        dum.allocate(inputs)
        dum.params.value = [1., 2., 0.]

        dum.index_start = 10
        dum.index_end = nval-1
        dum.run()
        self.assertTrue(np.all(np.isnan(dum.outputs[:10, 0])))

        try:
            dum.index_end = nval+1
        except Exception as err:
            self.assertTrue(\
                err.message.startswith('With model dummy, index (1001)'))
        else:
            raise Exception('Problem with error generation')


if __name__ == '__main__':
    unittest.main()
