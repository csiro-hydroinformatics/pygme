import os
import re
import unittest
import math

import json

from itertools import product

from timeit import Timer
import time

import pandas as pd
import numpy as np
np.seterr(all='print')

from scipy.special import kolmogorov

from pygme.data import Vector, Matrix

FHERE = os.path.dirname(os.path.abspath(__file__))

class VectorTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> VectorTestCase')

    def test_vector_nameserror(self):
        v = Vector('test', 3)
        v.names = ['a', 'b', 'c']
        self.assertTrue(list(v.names) == ['a', 'b', 'c'])
        try:
            v.names = ['a', 'b']
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('With test vector: tried setting _names'))


    def test_vector_minerror(self):
        v = Vector('test', 3)
        v.min = [-1, 10, 2]
        self.assertTrue(list(v.min.astype(int)) == [-1, 10, 2])
        try:
            v.min = [5, 3]
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('With test vector: tried setting _min'))


    def test_vector_maxerror(self):
        v = Vector('test', 3)
        v.max = [10, 100, 20]
        self.assertTrue(np.allclose(v.max, [10, 100, 20]))
        try:
            v.max = [5, 3]
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('With test vector: tried setting _max'))


    def test_vector_hitbounderror(self):
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


    def test_vector_print(self):
        v = Vector('test', 3)
        str = '{0}'.format(v)


    def test_vector_1dtest(self):
        v = Vector('test', 1)
        v.data = 10
        self.assertTrue(v.data.shape == (1, ) and v.data[0] == 10.)


    def test_vector_set(self):
        v = Vector('test', 2, 10)
        for iens in range(10):
            v.iens = iens
            v.data = [iens, np.random.uniform(0, 1)]

        for iens in range(10):
            v.iens = iens
            ck = v.data.shape == (2, )
            ck = ck and v.data[0] == iens
            self.assertTrue(ck)


    def test_vector_reset(self):
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


    def test_vector_random(self):
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


    def test_vector_setnamed(self):
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
            ' key "P20" not in the list of names'))


    def test_vector1_tojson(self):
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

    def test_matrix_nameserror(self):
        mat1 = Matrix.from_dims('test', 100, 5)
        mat1.names = ['a', 'b', 'c', 'd', 'e']
        try:
            mat1.names = ['a', 'b']
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('With test matrix: tried setting _names'))


    def test_matrix_setdata(self):
        mat1 = Matrix.from_dims('test', 100, 5)
        d0 = np.random.uniform(0, 1, (100, 5))
        mat1.data = d0
        self.assertTrue(mat1.data.shape == (100, 5))


    def test_matrix_setdataerror(self):
        mat1 = Matrix.from_dims('test', 100, 5)
        try:
            mat1.data = np.random.uniform(0, 1, (10, 5))
        except ValueError as e:
            pass
        self.assertTrue(str(e).startswith('With test matrix: tried setting _data'))


    def test_matrix_print(self):
        mat1 = Matrix.from_dims('test', 100, 5)
        str = '{0}'.format(mat1)


    def test_matrix_fromdata(self):
        mat1 = Matrix.from_dims('test', 100, 6)
        mat2 = Matrix.from_data('test', mat1.data)
        self.assertTrue(mat2.data.shape == (100, 6))


    def test_matrix_setdata_iensilead(self):
        nval = 10
        nvar = 5
        nlead = 3
        nens = 20
        mat1 = Matrix.from_dims('test', nval, nvar, nlead, nens)

        data = np.random.uniform(0, 10, (nval, nvar, nlead, nens))

        for ilead, iens in product(range(nlead), range(nens)):
            mat1.ilead = ilead
            mat1.iens = iens
            mat1.data = data[:, :, ilead, iens]

        mat2 = Matrix.from_data('test', data)
        data = []
        for ilead, iens in product(range(nlead), range(nens)):
            mat1.ilead = ilead
            mat2.ilead = ilead

            mat1.iens = iens
            mat2.iens = iens
            self.assertTrue(np.allclose(mat2.data, mat1.data))


    def test_matrix_reset(self):
        nval = 10
        nvar = 5
        nlead = 4
        nens = 20
        mat1 = Matrix.from_dims('test', nval, nvar, nlead, nens)

        test = np.ones((nval, nvar)).astype(np.float64)

        mat1.reset(2.)
        for ilead, iens in product(range(nlead), range(nens)):
            mat1.iens = iens
            mat1.ilead = ilead
            ck = np.allclose(mat1.data, 2.*test)
            self.assertTrue(ck)


    def test_matrix_tohdf(self):
        nval = 50
        nvar = 3
        nlead = 2
        nens = 4

        fhdf = os.path.join(FHERE, 'matrix.hdf5')
        if os.path.exists(fhdf):
            os.remove(fhdf)

        # Create matrix and write to file
        mat_all = {}
        for i in range(4):
            mat = Matrix.from_dims('test{0}_qqqq'.format(i), nval, nvar, nlead, nens,
                    prefix='LONG')

            for ilead, iens in product(range(nlead), range(nens)):
                mat.ilead = ilead
                mat.iens = iens
                mat.data = np.random.uniform(i, i+1, (nval, nvar))

            key = 'matrix_{0}'.format(i)
            mat.to_hdf(fhdf, key)
            mat_all[key] = mat

        # Read it back
        for k in mat_all:
            mat1 = mat_all[k]
            mat2 = Matrix.from_hdf(fhdf, k)

            for ilead, iens in product(range(nlead), range(nens)):
                mat1.ilead = ilead
                mat2.ilead = ilead

                mat1.iens = iens
                mat2.iens = iens

                ck = np.allclose(mat2.data, mat1.data)
                self.assertTrue(ck)


    def test_matrix_index(self):
        nval = 50
        nvar = 3
        nlead = 1
        nens = 1

        # Create matrix
        index = np.random.choice(range(1, 5), nval)
        index = np.cumsum(index)
        mat1 = Matrix.from_dims('test', nval, nvar, nlead, nens,
                prefix='TEST',
                index=index)

        try:
            index = np.arange(nval-10)
            mat1 = Matrix.from_dims('test', nval, nvar, nlead, nens,
                    index=index)
        except ValueError, e:
            pass
        self.assertTrue(str(e).startswith('With test matrix: tried to set index'))

        try:
            index = np.random.choice(range(-5, 5), nval)
            index = np.cumsum(index)
            mat1.index = index
        except ValueError, e:
            pass
        self.assertTrue(str(e).startswith('With test matrix: index is not strictly'))


    def test_matrix_slice(self):
        nval = 100
        nvar = 5
        index = np.cumsum(np.random.randint(1, 4, nval))
        data = np.random.uniform(size=(nval, nvar))
        mat1 = Matrix.from_data('data', data, index=index)

        kk = np.cumsum(np.random.randint(1, 3, 10))
        index2 = index[kk]
        mat2 = mat1.slice(index2)

        ck = np.allclose(mat1.data[kk, :], mat2.data)
        ck = ck & np.allclose(index2, mat2.index)
        self.assertTrue(ck)


    def test_matrix_random(self):
        nval = 1000
        nvar = 5
        nlead = 10
        nens = 100
        mat = Matrix.from_dims('data', nval, nvar, nlead, nens)
        mat.random()

        for ivar, ilead, iens in product(range(mat.nvar),
                        range(mat.nlead), range(mat.nens)):
            mat.iens = iens
            mat.ilead = ilead
            values = np.sort(mat.data[:, ivar])

            # Test that data is coming from uniform distribution
            err = np.abs(values - np.linspace(0, 1, nval))
            D = math.sqrt(nval) * np.max(err)
            ks = kolmogorov(D)
            self.assertTrue(ks >= 1e-5)


    def test_matrix_aggregateval(self):
        nval = 1000
        nvar = 5
        nlead = 10
        nens = 5
        mat = Matrix.from_dims('data', nval, nvar, nlead, nens)
        mat.random()

        # Define an index
        dt = pd.Series(1, index=pd.date_range('2001-01-01', freq='D', periods=nval))
        dt = dt.index
        months = dt.year*100 + dt.month
        nmonths = np.unique(months).shape[0]

        # Run aggregation - val
        matagg = mat.aggregate(months, aggfunc=np.sum, axis='val')

        ck = (matagg.nval, matagg.nvar, matagg.nlead, matagg.nens) == (nmonths, nvar, nlead, nens)
        self.assertTrue(ck)

        for ilead, iens in product(range(nlead), range(nens)):
            mat.ilead = ilead
            mat.iens = iens
            matagg.ilead = ilead
            matagg.iens = iens

            data = pd.DataFrame(mat.data)
            data['index'] = months
            aggdata = data.groupby('index').apply(np.sum).drop('index', axis=1)
            ck = np.allclose(matagg.data, aggdata.values)
            self.assertTrue(ck)

    def test_matrix_aggregatelead(self):
        nval = 100
        nvar = 5
        nlead = 180
        nens = 5
        mat = Matrix.from_dims('data', nval, nvar, nlead, nens)
        mat.random()

        # Define an index
        dt = pd.Series(1, index=pd.date_range('2001-01-01', freq='D', periods=nlead))
        dt = dt.index
        months = dt.year*100 + dt.month
        nmonths = np.unique(months).shape[0]

        # Run aggregation - lead
        matagg = mat.aggregate(months, aggfunc=np.sum, axis='lead')

        ck = (matagg.nval, matagg.nvar, matagg.nlead, matagg.nens) == (nval, nvar, nmonths, nens)
        self.assertTrue(ck)

        for ival, iens in product(range(nval), range(nens)):
            data = pd.DataFrame(mat._data[ival, :, :, iens])
            data['index'] = months
            aggdata = data.groupby('index').apply(np.sum).drop('index', axis=1)
            ck = np.allclose(matagg._data[ival, :, :, iens], aggdata.values)
            self.assertTrue(ck)


        import pdb; pdb.set_trace()




if __name__ == '__main__':
    unittest.main()
