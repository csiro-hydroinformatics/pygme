import os
import re
import unittest

from timeit import Timer
import time

import numpy as np
np.seterr(all='print')

from pygme.model import Model

from dummy import Dummy, MassiveDummy


class ModelTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> ModelTestCase')
        FOUT = os.path.dirname(os.path.abspath(__file__))
        self.FOUT = FOUT


    def test_model1(self):
        dum = Dummy()
        str = '{0}'.format(dum)


    def test_model2(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        dum = Dummy()
        dum.allocate(len(inputs), 2)


    def test_model3(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10., 0.1]
        dum = Dummy()
        dum.allocate(len(inputs), 2)
        dum.params = params


    def test_model4(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10., 0.1]
        dum = Dummy()
        dum.allocate(len(inputs), 2)
        dum.params = params
        dum.initialise(states=[10, 0])
        dum.inputs = inputs


    def test_model5(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10., 0.]
        dum = Dummy()
        dum.allocate(len(inputs), 2)
        dum.params = params
        dum.initialise(states=[10, 0])
        dum.inputs = inputs

        dum.idx_start = 0
        dum.idx_end = 999
        dum.run()

        expected1 = params[0] + params[1] * np.cumsum(inputs[:, 0])
        ck1 = np.allclose(expected1, dum.outputs[:, 0])
        self.assertTrue(ck1)

        expected2 = params[0] + params[1] * np.cumsum(inputs[:, 1])
        ck2 = np.allclose(expected2, dum.outputs[:, 1])
        self.assertTrue(ck2)


    def test_model6(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10., 0.5]
        dum = Dummy()
        dum.allocate(len(inputs), 2)
        dum.params = params
        dum.initialise(states=[10, 0])
        dum.inputs = inputs

        dum.config.data = [10]

        dum2 = dum.clone()

        d1 = dum.inputs
        d2 = dum2.inputs

        self.assertTrue(np.allclose(d1, d2))

        # Check that inputs were copied and not pointing to same object
        d2[0, 0] += 1
        self.assertTrue(np.allclose(d1[0, 0]+1, d2[0, 0]))

    def test_model7(self):
        dum = Dummy()
        dum.allocate(10, 2)

        uh = [0.25]*4 + [0.] * (len(dum.uh)-4)
        uh = np.array(uh)
        self.assertTrue(np.allclose(dum.uh, uh))

        dum.params = [1., 2., 0.4]
        self.assertTrue(np.allclose(dum.uh[:4], 0.25))

    def test_model8(self):
        inputs = np.random.uniform(0, 1, (1000, 1))
        dum = MassiveDummy()
        dum.params = []
        dum.allocate(len(inputs), 1)
        dum.initialise(states=[])
        dum.inputs = inputs

        dum.idx_start = 0
        dum.idx_end = len(inputs)-1
        dum.run()


    def test_model9(self):

        dum = Dummy(nens_params=3,
            nens_states=4,
            nens_outputs=5)

        nval = 1000
        noutputs = 2
        nlead = 10
        nens = 2
        dum.allocate(nval, noutputs, nlead, nens)

        self.assertTrue(dum.get_dims('params') == (3, 3))
        self.assertTrue(dum.get_dims('states') == (2, 4))
        self.assertTrue(dum.get_dims('inputs') == (nval, 2, nlead, 2))
        self.assertTrue(dum.get_dims('outputs') == (nval, noutputs, nlead, 5))


    def test_model10(self):

        dum = Dummy(nens_params=3,
            nens_states=4,
            nens_outputs=5)

        nval = 1000
        noutputs = 2
        dum.allocate(nval, noutputs, nens_inputs = 2)

        dum.random('params', 'uniform')
        dum.random('states')
        dum.random('statesuh')


    def test_model11(self):

        dum = Dummy()

        nval = 1000
        noutputs = 1
        dum.allocate(nval, noutputs)
        _, ninputs, _, _ = dum.get_dims('inputs')
        dum.inputs = np.random.uniform(0, 1, (nval, ninputs))

        dum.idx_start = 10
        dum.idx_end = nval-1
        dum.run_ens()
        self.assertTrue(np.all(np.isnan(dum.outputs[:10, 0])))

        try:
            dum.idx_end = nval+1
        except Exception, e:
            pass

        self.assertTrue(e.message.startswith('Model dummy: idx_end < 0'))




if __name__ == '__main__':
    unittest.main()
