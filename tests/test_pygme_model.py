import os
import re
import unittest

from timeit import Timer
import time

import numpy as np
np.seterr(all='print')

from pygme import model
model.set_seed()

from pygme.model import Model, Vector, Matrix

from dummy import Dummy, MassiveDummy


class VectorTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> VectorTestCase')

    def test_vector1(self):
        v = Vector('test', 3)
        v.names = ['a', 'b', 'c']
        self.assertTrue(list(v.names) == ['a', 'b', 'c'])
        try:
            v.names = ['a', 'b']
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('Vector test: tried setting _names'))


    def test_vector2(self):
        v = Vector('test', 3)
        v.min = [-1, 10, 2]
        self.assertTrue(list(v.min.astype(int)) == [-1, 10, 2])
        try:
            v.min = [5, 3]
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('Vector test: tried setting _min'))


    def test_vector3(self):
        v = Vector('test', 3)
        v.max = [10, 100, 20]
        self.assertTrue(np.allclose(v.max, [10, 100, 20]))
        try:
            v.max = [5, 3]
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('Vector test: tried setting _max'))


    def test_vector4(self):
        v = Vector('test', 3)
        v.min = [-1, 10, 2]
        self.assertTrue(~v.hitbounds)
        v.data = [-100, 50, 2.001]
        self.assertTrue(np.allclose(v.data, [-1, 50, 2.001]))
        self.assertTrue(v.hitbounds)

        try:
            v.data = [5, 3]
        except ValueError as e:
            pass

        self.assertTrue(str(e).startswith('Vector test / ensemble 0:'
            ' tried setting data with vector of wrong size (2 instead of 3)'))


    def test_vector5(self):
        v = Vector('test', 3)
        str = '{0}'.format(v)


    def test_vector6(self):
        v = Vector('test', 1)
        v.data = 10
        self.assertTrue(v.data.shape == (1, ) and v.data[0] == 10.)

    def test_vector7(self):
        v = Vector('test', 2, 10)
        for iens in range(10):
            v.iens = iens
            v.data = [iens, np.random.uniform(0, 1)]

        for iens in range(10):
            v.iens = iens
            ck = v.data.shape == (2, )
            ck = ck and v.data[0] == iens
            self.assertTrue(ck)

    def test_vector8(self):
        v = Vector('test', 2, 10)
        default = [1., 1.]
        v.default = default
        v.reset()
        default[0] = 2

        for iens in range(10):
            v.iens = iens
            ck = np.allclose(v.data, [1., 1.])
            self.assertTrue(ck)

        v.reset(10)

        for iens in range(10):
            v.iens = iens
            ck = np.allclose(v.data, [10., 10.])
            self.assertTrue(ck)


    def test_vector9(self):
        v = Vector('test', 5, 100, has_minmax=True)
        v.min = range(0, v.nval)
        v.max = range(1, v.nval+1)
        v.means = np.arange(0.5, v.nval+0.5)

        a = np.arange(1, v.nval+1) * 0.2
        b = np.random.uniform(-1, 1, (v.nval, v.nval))
        v.covar = np.dot(b.T, np.dot(np.diag(a), b))

        v.random()

        v2 = v.clone()
        v2.random(distribution='uniform')


    def test_vector10(self):
        v = Vector('test', 5, 100, prefix='P')

        v.iens = 10

        v['P2'] = 25.
        v.iens = 20

        self.assertTrue(np.allclose(v._data[10, 2], 25.))

        v.iens = 10
        self.assertTrue(np.allclose(v['P2'], 25.))

        try:
            v['P20'] = 10
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('Vector test:' +
            ' key P20 not in the list of names'))


class MatrixTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> MatrixTestCase')

    def test_matrix1(self):
        m1 = Matrix.fromdims('test', 100, 5)
        m1.names = ['a', 'b', 'c', 'd', 'e']
        try:
            m1.names = ['a', 'b']
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('test matrix: tried setting _names'))


    def test_matrix2(self):
        m1 = Matrix.fromdims('test', 100, 5)
        d0 = np.random.uniform(0, 1, (100, 5))
        m1.data = d0
        self.assertTrue(m1.data.shape == (100, 5))


    def test_matrix3(self):
        m1 = Matrix.fromdims('test', 100, 5)
        try:
            m1.data = np.random.uniform(0, 1, (10, 5))
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('test matrix: tried setting _data'))


    def test_matrix4(self):
        m1 = Matrix.fromdims('test', 100, 5)
        str = '{0}'.format(m1)


    def test_matrix5(self):
        m1 = Matrix.fromdims('test', 100, 1)
        m1.data = np.random.uniform(0, 1, 100)
        self.assertTrue(m1.data.shape == (100, 1))


    def test_matrix6(self):
        m1 = Matrix.fromdims('test', 100, 6)
        m2 = Matrix.fromdata('test', m1.data)
        self.assertTrue(m2.data.shape == (100, 6))


    def test_matrix7(self):
        nval = 10
        nvar = 5
        nens = 20
        m1 = Matrix.fromdims('test', nval, nvar, nens)

        data = np.random.uniform(0, 10, (nens, nval, nvar))

        for iens in range(nens):
            m1.iens = iens
            m1.data = data[iens]

        m2 = Matrix.fromdata('test', data)
        data = []
        for iens in range(nens):
            m1.iens = iens
            m2.iens = iens
            self.assertTrue(np.allclose(m2.data, m1.data))


    def test_matrix8(self):
        nval = 10
        nvar = 5
        nens = 20
        m1 = Matrix.fromdims('test', nval, nvar, nens)

        test = np.ones((nval, nvar)).astype(np.float64)

        m1.reset(2.)
        for iens in range(nens):
            m1.iens = iens
            ck = np.allclose(m1.data, 2.*test)
            self.assertTrue(ck)


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
        dum.allocate(nval, noutputs, nens_inputs = 2)

        self.assertTrue(dum.get_dims('params') == (3, 3))
        self.assertTrue(dum.get_dims('states') == (2, 4))
        self.assertTrue(dum.get_dims('inputs') == (nval, 2, 2))
        self.assertTrue(dum.get_dims('outputs') == (nval, noutputs, 5))


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
        dum.inputs = np.random.uniform(0, 1, (nval, dum.ninputs))

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
