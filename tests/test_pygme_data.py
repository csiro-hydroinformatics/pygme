import os
import re
import unittest

import json

from itertools import product

from timeit import Timer
import time

import numpy as np
np.seterr(all='print')

from pygme.data import Vector, Matrix

FHERE = os.path.dirname(os.path.abspath(__file__))

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
        self.assertTrue(str(e).startswith('With test vector: tried setting _names'))


    def test_vector2(self):
        v = Vector('test', 3)
        v.min = [-1, 10, 2]
        self.assertTrue(list(v.min.astype(int)) == [-1, 10, 2])
        try:
            v.min = [5, 3]
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('With test vector: tried setting _min'))


    def test_vector3(self):
        v = Vector('test', 3)
        v.max = [10, 100, 20]
        self.assertTrue(np.allclose(v.max, [10, 100, 20]))
        try:
            v.max = [5, 3]
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('With test vector: tried setting _max'))


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

        self.assertTrue(str(e).startswith('With test vector / ensemble 0:'
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
        self.assertTrue(str(e).startswith('With test vector:' +
            ' key P20 not in the list of names'))


    def test_vector11(self):
        js = {}
        vectors = {}
        for iv in range(3):
            v = Vector('test', 5, 3, prefix='P')
            for iens in range(v.nens):
                v.iens = iens
                v.data = np.random.uniform(0, 1, v.nval)

            js['vector{0}'.format(iv)] = v.to_dict()
            vectors['vector{0}'.format(iv)] = v

        # Export vector and write it to json
        fd = os.path.join(FHERE, 'vector.json')
        with open(fd, 'w') as fd_obj:
            json.dump(js, fd_obj, indent=4)

        # Reads from the same file
        with open(fd, 'r') as fd_obj:
            js2 = json.load(fd_obj)

        for vjs2 in js:
            d = js[vjs2]
            v2 = Vector.from_dict(d)

            expected = vectors[vjs2]

            for iens in range(expected.nens):
                v2.iens = iens
                expected.iens = iens
                self.assertTrue(np.allclose(v2.data, expected.data))


class MatrixTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> MatrixTestCase')

    def test_matrix1(self):
        m1 = Matrix.from_dims('test', 100, 5)
        m1.names = ['a', 'b', 'c', 'd', 'e']
        try:
            m1.names = ['a', 'b']
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('With test matrix: tried setting _names'))


    def test_matrix2(self):
        m1 = Matrix.from_dims('test', 100, 5)
        d0 = np.random.uniform(0, 1, (100, 5))
        m1.data = d0
        self.assertTrue(m1.data.shape == (100, 5))


    def test_matrix3(self):
        m1 = Matrix.from_dims('test', 100, 5)
        try:
            m1.data = np.random.uniform(0, 1, (10, 5))
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('With test matrix: tried setting _data'))


    def test_matrix4(self):
        m1 = Matrix.from_dims('test', 100, 5)
        str = '{0}'.format(m1)


    def test_matrix5(self):
        m1 = Matrix.from_dims('test', 100, 1)
        m1.data = np.random.uniform(0, 1, 100)
        self.assertTrue(m1.data.shape == (100, 1))


    def test_matrix6(self):
        m1 = Matrix.from_dims('test', 100, 6)
        m2 = Matrix.from_data('test', m1.data)
        self.assertTrue(m2.data.shape == (100, 6))


    def test_matrix7(self):
        nval = 10
        nvar = 5
        nlead = 3
        nens = 20
        m1 = Matrix.from_dims('test', nval, nvar, nlead, nens)

        data = np.random.uniform(0, 10, (nval, nvar, nlead, nens))

        for ilead, iens in product(range(nlead), range(nens)):
            m1.ilead = ilead
            m1.iens = iens
            m1.data = data[:, :, ilead, iens]

        m2 = Matrix.from_data('test', data)
        data = []
        for ilead, iens in product(range(nlead), range(nens)):
            m1.ilead = ilead
            m2.ilead = ilead

            m1.iens = iens
            m2.iens = iens
            self.assertTrue(np.allclose(m2.data, m1.data))


    def test_matrix8(self):
        nval = 10
        nvar = 5
        nlead = 4
        nens = 20
        m1 = Matrix.from_dims('test', nval, nvar, nlead, nens)

        test = np.ones((nval, nvar)).astype(np.float64)

        m1.reset(2.)
        for ilead, iens in product(range(nlead), range(nens)):
            m1.iens = iens
            m1.ilead = ilead
            ck = np.allclose(m1.data, 2.*test)
            self.assertTrue(ck)


    def test_matrix9(self):
        nval = 50
        nvar = 3
        nlead = 2
        nens = 4

        fhdf = os.path.join(FHERE, 'matrix.hdf5')
        if os.path.exists(fhdf):
            os.remove(fhdf)

        # Create matrix and write to file
        mat1 = {}
        for i in range(4):
            m1 = Matrix.from_dims('test{0}'.format(i), nval, nvar, nlead, nens,
                    prefix='LONG')

            for ilead, iens in product(range(nlead), range(nens)):
                m1.ilead = ilead
                m1.iens = iens
                m1.data = np.random.uniform(i, i+1, (nval, nvar))

            key = 'matrix_{0}'.format(i)
            m1.to_hdf(fhdf, key)
            mat1[key] = m1

        # Read it back
        for k in mat1:
            m1 = mat1[k]
            m2 = Matrix.from_hdf(fhdf, k)

            for ilead, iens in product(range(nlead), range(nens)):
                m1.ilead = ilead
                m2.ilead = ilead

                m1.iens = iens
                m2.iens = iens

                ck = np.allclose(m2.data, m1.data)
                self.assertTrue(ck)


    def test_matrix10(self):
        nval = 50
        nvar = 3
        nlead = 1
        nens = 1

        # Create matrix
        ts_index = np.random.choice(range(1, 5), nval)
        ts_index = np.cumsum(ts_index)
        m1 = Matrix.from_dims('test', nval, nvar, nlead, nens,
                prefix='TEST',
                ts_index=ts_index)

        try:
            ts_index = np.arange(nval-10)
            m1 = Matrix.from_dims('test', nval, nvar, nlead, nens,
                    ts_index=ts_index)
        except ValueError, e:
            pass
        self.assertTrue(str(e).startswith('With test matrix: tried to set ts_index'))

        try:
            ts_index = np.random.choice(range(-5, 5), nval)
            ts_index = np.cumsum(ts_index)
            m1.ts_index = ts_index
        except ValueError, e:
            pass
        self.assertTrue(str(e).startswith('With test matrix: ts_index is not strictly'))


if __name__ == '__main__':
    unittest.main()
