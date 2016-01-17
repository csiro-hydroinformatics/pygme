import os
import re
import unittest

from timeit import Timer
import time

import numpy as np

from useme.model import Model, Vector, Matrix


class Dummy(Model):

    def __init__(self):
        Model.__init__(self, 'dummy',
            nconfig=1,\
            ninputs=2, \
            nparams=2, \
            nstates=2, \
            noutputs_max=2,
            inputs_names=['I1', 'I2'], \
            outputs_names=['O1', 'O2'])

        self.config.names = 'Config1'
        self.states.names = ['State1', 'State2']
        self.params.names = ['Param1', 'Param2']
        self.params.units = ['mm', 'mm']

    def run(self, idx_start, idx_end):
        par1 = self.params['Param1']
        par2 = self.params['Param2']

        outputs = par1 + par2 * np.cumsum(self.inputs.data, 0)
        self.outputs.data = outputs[idx_start:idx_end, :self.outputs.nvar]

        self.states.data = list(self.outputs.data[idx_end]) \
                    + [0.] * (2-self.outputs.nvar)

    def set_uh(self):
        nuh = self.uh.nval
        self.uh.data = [1.] * nuh


class MassiveDummy(Model):

    def __init__(self):
        Model.__init__(self, 'dummy',
            nconfig=0,\
            ninputs=1, \
            nparams=0, \
            nstates=0, \
            noutputs_max=1,
            inputs_names=[], \
            outputs_names=['O'])

    def run(self, idx_start, idx_end):
        nval = self.outputs.data.nval
        outputs = self.inputs.data + np.random(0, 1, (nval, 1))
        self.outputs.data = outputs[idx_start:idx_end, :]



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
        self.assertTrue(e.message.startswith('Vector test: tried setting _names'))


    def test_vector2(self):
        v = Vector('test', 3)
        v.units = ['m2', 'mm/d', 'GL']
        self.assertTrue(list(v.units) == ['m2', 'mm/d', 'GL'])
        try:
            v.units = ['m2', 'mm/d']
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('Vector test: tried setting _units'))


    def test_vector3(self):
        v = Vector('test', 3)
        v.min = [-1, 10, 2]
        self.assertTrue(list(v.min.astype(int)) == [-1, 10, 2])
        try:
            v.min = [5, 3]
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('Vector test: tried setting _min'))


    def test_vector4(self):
        v = Vector('test', 3)
        v.max = [10, 100, 20]
        self.assertTrue(np.allclose(v.max, [10, 100, 20]))
        try:
            v.max = [5, 3]
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('Vector test: tried setting _max'))


    def test_vector5(self):
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

        self.assertTrue(e.message.startswith('Vector test / ensemble 0:'
            ' tried setting data with vector of wrong size (2 instead of 3)'))


    def test_vector6(self):
        v = Vector('test', 3)
        str = '{0}'.format(v)


    def test_vector7(self):
        v = Vector('test', 1)
        v.data = 10
        self.assertTrue(v.data.shape == (1, ) and v.data[0] == 10.)

    def test_vector8(self):
        v = Vector('test', 2, 10)
        for iens in range(10):
            v.iens = iens
            v.data = [iens, np.random.uniform(0, 1)]
            v['X0'] = iens + 1
            v['X1'] = np.random.uniform(0, 1)

        for iens in range(10):
            v.iens = iens
            ck = v.data.shape == (2, )
            ck = ck and v['X0'] == iens + 1
            ck = ck and v.data[0] == iens + 1
            self.assertTrue(ck)

    def test_vector9(self):
        v = Vector('test', 2, 10)
        v.default = [1., 1.]
        v.reset()

        for iens in range(10):
            v.iens = iens
            ck = np.allclose(v.data, [1., 1.])
            self.assertTrue(ck)

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
        self.assertTrue(e.message.startswith('test matrix: tried setting _names'))


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
        self.assertTrue(e.message.startswith('test matrix: tried setting _data'))


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
        params = [0.5, 10.]
        dum = Dummy()
        dum.allocate(len(inputs), 2)
        dum.params.data = params


    def test_model4(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10.]
        dum = Dummy()
        dum.allocate(len(inputs), 2)
        dum.params.data = params
        dum.initialise(states=[10, 0])
        dum.inputs.data = inputs


    def test_model5(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10.]
        dum = Dummy()
        dum.allocate(len(inputs), 2)
        dum.params.data = params
        dum.initialise(states=[10, 0])
        dum.inputs.data = inputs

        idx_start = 0
        idx_end = 999
        dum.run(idx_start, idx_end)

        expected1 = params[0] + params[1] * np.cumsum(inputs[:, 0])
        ck1 = np.allclose(expected1, dum.outputs.data[:, 0])
        self.assertTrue(ck1)

        expected2 = params[0] + params[1] * np.cumsum(inputs[:, 1])
        ck2 = np.allclose(expected2, dum.outputs.data[:, 1])
        self.assertTrue(ck2)


    def test_model6(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10.]
        dum = Dummy()
        dum.allocate(len(inputs), 2)
        dum.params.data = params
        dum.initialise(states=[10, 0])
        dum.inputs.data = inputs

        dum2 = dum.clone()

        d1 = dum.inputs.data
        d2 = dum2.inputs.data

        self.assertTrue(np.allclose(d1, d2))

        d2[0, 0] += 1
        self.assertTrue(np.allclose(d1[0, 0]+1, d2[0, 0]))

    def test_model7(self):
        dum = Dummy()
        dum.allocate(10, 2)

        uh = [1.] + [0.] * (dum.uh.nval-1)
        uh = np.array(uh)
        self.assertTrue(np.allclose(dum.uh.data, uh))

        dum.params.data = [1., 2.]

        self.assertTrue(np.allclose(dum.uh.data, 1.))

    def test_model8(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10.]
        dum = Dummy()
        out = dum.fullrun(inputs, params)

        expected1 = params[0] + params[1] * np.cumsum(inputs[:, 0])
        ck1 = np.allclose(expected1, out[:, 0])
        self.assertTrue(ck1)

    def test_model9(self):
        inputs = np.random.uniform(0, 1, (1000, 1))
        dum = MassiveDummy()
        dum.allocate(len(inputs), 1)
        dum.params.data = []
        dum.initialise(states=[])
        dum.inputs.data = inputs
        dum.run()


if __name__ == '__main__':
    unittest.main()
