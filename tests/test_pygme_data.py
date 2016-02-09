import os
import re
import unittest

from timeit import Timer
import time

import numpy as np
np.seterr(all='print')

from pygme.data import Vector, Matrix


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



if __name__ == '__main__':
    unittest.main()
