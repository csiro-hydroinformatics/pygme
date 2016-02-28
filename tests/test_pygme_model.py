import os
import re
import unittest

from timeit import Timer
import time

import numpy as np
np.seterr(all='print')

from pygme.model import Model
from pygme.data import Matrix

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
        dum.allocate(inputs)


    def test_model3(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10., 0.1]
        dum = Dummy()
        dum.allocate(inputs)
        dum.params = params

        self.assertTrue(np.allclose(dum._params.data, params))


    def test_model4(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10., 0.1]
        dum = Dummy()
        dum.allocate(inputs)
        dum.params = params

        states = [10, 0]
        dum.initialise(states)

        self.assertTrue(np.allclose(dum._states.data, states))


    def test_model5(self):
        nval = 100
        inputs = np.random.uniform(0, 1, (nval, 2))
        params = [0.5, 10., 0.]
        dum = Dummy()
        dum.allocate(inputs, 2)
        dum.params = params
        dum.config['continuous'] = 1

        states = np.array([10., 0.])
        dum.initialise(states=states)

        dum.index_start = 0
        dum.index_end = nval-1
        dum.run()

        expected = params[0] + params[1] * inputs
        expected = np.cumsum(expected, 0)
        expected = expected + states
        ck = np.allclose(expected, dum.outputs)
        self.assertTrue(ck)


    def test_model6(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10., 0.5]

        dum = Dummy()
        dum.allocate(inputs)
        dum.params = params
        dum.initialise(states=[10, 0])

        dum.config.data = [10]

        dum2 = dum.clone()
        dum2.inputs = dum2.inputs

        d1 = dum.inputs
        d2 = dum2.inputs

        self.assertTrue(np.allclose(d1, d2))

        # Check that inputs were copied and not pointing to same object
        d2[0, 0] += 1
        self.assertTrue(np.allclose(d1[0, 0]+1, d2[0, 0]))

    def test_model7(self):
        dum = Dummy()
        inputs = np.random.uniform(0, 1, (10, 2))
        dum.allocate(inputs, 2)
        dum.params = np.zeros(3)

        uh = [0.25]*4 + [0.] * (len(dum.uh)-4)
        uh = np.array(uh)
        self.assertTrue(np.allclose(dum.uh, uh))

        dum.params = [1., 2., 0.4]
        self.assertTrue(np.allclose(dum.uh[:4], 0.25))

    def test_model8(self):
        inputs = np.random.uniform(0, 1, (1000, 1))
        dum = MassiveDummy()
        dum.params = []
        dum.allocate(inputs)
        dum.initialise(states=[])
        dum.inputs = inputs

        dum.index_start = 0
        dum.index_end = len(inputs)-1
        dum.run()


    def test_model9(self):
        dum = Dummy(nens_params=3,
            nens_states=4,
            nens_outputs=5)

        nval = 1000
        noutputs = 2
        nlead = 10
        nens = 2
        inputs = Matrix.from_dims('inputs', nval, 2, nlead, nens)
        dum.allocate(inputs, noutputs)

        self.assertTrue(dum.get_dims('params') == (3, 3))
        self.assertTrue(dum.get_dims('states') == (2, 4))
        self.assertTrue(dum.get_dims('inputs') == (nval, 2, nlead, 2))
        self.assertTrue(dum.get_dims('outputs') == (nval, noutputs, nlead, 5))


    def test_model10(self):

        dum = Dummy(nens_params=3,
            nens_states=4,
            nens_outputs=5)

        nval = 1000
        inputs = Matrix.from_dims('inputs', nval, 2, 1, 10)
        noutputs = 2
        dum.allocate(inputs, noutputs)

        dum.random('params', 'uniform')
        dum.random('states')
        dum.random('statesuh')


    def test_model11(self):

        dum = Dummy()

        nval = 1000
        noutputs = 1
        _, ninputs, _, _ = dum.get_dims('inputs')
        inputs = np.random.uniform(0, 1, (nval, ninputs))
        dum.allocate(inputs)
        dum.params = [1., 2., 0.]

        dum.index_start = 10
        dum.index_end = nval-1
        dum.run()
        self.assertTrue(np.all(np.isnan(dum.outputs[:10, 0])))

        try:
            dum.index_end = nval+1
        except Exception, e:
            pass
        self.assertTrue(e.message.startswith('With model dummy, index (1001)'))


    def test_model12(self):

        dum = Dummy()

        nval = 100
        noutputs = 2
        _, ninputs, _, _ = dum.get_dims('inputs')
        inputs = np.random.uniform(0, 1, (nval, ninputs))
        dum.allocate(inputs)
        dum.params = [1., 2., 0.]

        dum.run_as_block = True
        dum.initialise()
        dum.run()
        o1 = dum.outputs.copy()

        dum.run_as_block = False
        dum.initialise()
        dum.run()
        o2 = dum.outputs.copy()

        ck = np.allclose(o1, o2)
        self.assertTrue(ck)



if __name__ == '__main__':
    unittest.main()
