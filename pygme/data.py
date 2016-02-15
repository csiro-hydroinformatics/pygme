
import re
from itertools import product

import math
import random
import numpy as np

import pandas as pd


def set_seed(seed=333):
    np.random.seed(seed)


class Vector(object):

    def __init__(self, id, nval, nens=1, prefix='X',
            has_minmax = True):

        self.id = id
        self._nval = nval
        self._nens = nens
        self.has_minmax = has_minmax
        self.prefix = prefix

        self._iens = 0

        self._data = np.nan * np.ones((nens, nval), dtype=np.float64)

        self._names = np.array(['{0}{1}'.format(prefix, i)
                                    for i in range(nval)])

        self._default = np.nan * np.ones(nval, dtype=np.float64)

        if has_minmax:
            self._min = -np.inf * np.ones(nval, dtype=np.float64)
            self._max = np.inf * np.ones(nval, dtype=np.float64)
            self._hitbounds = np.zeros(nens, dtype=np.bool)
        else:
            self._min = None
            self._max = None
            self._hitbounds = None

        self._means = 0 * np.ones(nval, dtype=np.float64)
        self._covar = np.eye(nval, dtype=np.float64)


    @classmethod
    def from_dict(cls, data):
        ''' Create vector object from dictionary '''

        vect = Vector(data['id'], int(data['nval']), int(data['nens']),
                    data['prefix'], bool(data['has_minmax']))

        vect.iens = int(data['iens'])

        for item in ['names', 'default',
            'min', 'max', 'means']:
            if item in data:
                value = data[item]
                setattr(vect, item, value)

        for iens in range(vect.nens):
            vect.iens = iens
            vect.data = data['ensembles'][iens]['data']
            vect._hitbounds[iens] = data['ensembles'][iens]['hitbounds']

        return vect


    def __str__(self):

        str = 'With {0} vector : nval={1} nens={2} {{'.format( \
            self.id, self.nval, self.nens)
        str += ', '.join(self._names)
        str += '}'

        return str


    def __findname__(self, key):
        if not key in self._names:
            raise ValueError(('With {0} vector: key {1} not in the' +
                ' list of names').format(self.id, key))

        return np.where(self._names == key)[0][0]


    def __setitem__(self, key, value):
        idx = self.__findname__(key)
        self._data[self._iens, idx] = value


    def __getitem__(self, key):
        idx = self.__findname__(key)
        return self._data[self._iens, idx]


    def __setattrib__(self, target, source):

        if target in ['_min', '_max'] and not self.has_minmax:
            raise ValueError(('With {0} vector: Cannot set min or max, '
                'vector do not have this attribute').format(self.id))

        if target == '_covar':
            _source = np.atleast_2d(source)
            dims = (self.nval, self.nval)
        else:
            _source = np.atleast_1d(source).flatten()
            dims = (self.nval, )

        if not target in ['_names']:
            _source = _source.astype(np.float64)

        if _source.shape != dims:
            raise ValueError(('With {0} vector: tried setting {1}, ' + \
                'got wrong size ({2} instead of {3})').format(\
                self.id, target, _source.shape, dims))

        setattr(self, target, _source)


    @property
    def nens(self):
        return self._nens

    @property
    def nval(self):
        return self._nval

    @property
    def data(self):
        ''' Get data for a given ensemble member set by iens '''
        return self._data[self._iens, :]

    @data.setter
    def data(self, value):
        ''' Set data for a given ensemble member set by iens '''

        _value = np.atleast_1d(value).flatten()

        if len(_value) != self.nval:
            raise ValueError(('With {0} vector / ensemble {1}: tried setting data ' + \
                'with vector of wrong size ({2} instead of {3})').format(\
                self.id, self._iens, len(_value), self.nval))

        if self.has_minmax:
            # Avoids raising a warning with NaN substraction
            with np.errstate(invalid='ignore'):
                hitb = np.subtract(_value, self._min) < 0.
                hitb = hitb | (np.subtract(self._max, _value) < 0.)

            self._hitbounds[self._iens] = np.any(hitb)
            self._data[self._iens, :] = np.clip(_value, self._min, self._max)

        else:
            self._data[self._iens, :] = _value


    @property
    def means(self):
        return self._means

    @means.setter
    def means(self, value):
        self.__setattrib__('_means', value)


    @property
    def covar(self):
        return self._covar

    @covar.setter
    def covar(self, value):
        self.__setattrib__('_covar', value)



    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        self.__setattrib__('_min', value)



    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self.__setattrib__('_max', value)


    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, value):
        self.__setattrib__('_default', value)

        if self.has_minmax:
            self._default = np.clip(self._default, self._min, self._max)


    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self.__setattrib__('_names', value)


    @property
    def hitbounds(self):
        return self._hitbounds[self._iens]


    @property
    def iens(self):
        return self._iens

    @iens.setter
    def iens(self, value):
        ''' Set the ensemble number. Checks that number is not greater that total number of ensembles '''

        value = int(value)
        if value >= self.nens or value < 0:
            raise ValueError(('With {0} vector: iens {1} ' \
                    '>= nens {2} or < 0').format( \
                                self.id, value, self.nens))

        self._iens = value


    def reset(self, value=None):
        ''' Set parameter data to a given value or default values if value is None '''
        nval = self.nval

        if value is None:
            value = self._default
        else:
            value = (value * np.ones(self.nval)).astype(np.float64)

        value = np.clip(value.flatten(), self.min, self.max)
        self._data = np.repeat(np.atleast_2d(value), self.nens, axis=0)


    def random(self, distribution='normal', seed=333):
        ''' Randomise vector data '''

        if not self.has_minmax and distribution=='uniform':
            raise ValueError(('With {0} vector: Cannot randomize with uniform ' +
                ' distribution if vector has no min/max').format(self.id))

        if (self._means is None or self._covar is None) and distribution=='uniform':
            raise ValueError(('With {0} vector: Cannot randomize with normal ' +
                ' distribution if vector has no._means or._covar').format(self.id))


        # Sample vector data
        if distribution == 'normal':
            self._data = np.random.multivariate_normal(self._means,
                    self._covar, (self.nens, ))

        elif distribution == 'uniform':
            self._data = np.random.uniform(self.min, self.max,
                    (self.nens, self.nval))
        else:
            raise ValueError(('With {0} vector: Random, distribution ' +
                '{1} not allowed').format(self.id, distribution))


    def clone(self, nens=None):

        if nens is None:
            nens = self.nens

        clone = Vector(self.id, self.nval, nens,
                    self.prefix, self.has_minmax)

        clone.iens = self.iens
        clone.names = self.names.copy()

        if self.has_minmax:
            clone.min = self.min.copy()
            clone.max = self.max.copy()
            clone._hitbounds = [h.copy() for h in self._hitbounds]

        clone._means = self._means.copy()
        clone._covar = self._covar.copy()

        # Copy data if the number of ensemble is identical
        if nens is None:
            clone._data = self._data.copy()

        return clone


    def to_dict(self):
        ''' Write vector data to json format '''

        data = {}
        for item in ['id', 'nval', 'nens', 'has_minmax',
                    'prefix', 'iens']:
            value  = getattr(self, item)
            if item == 'has_minmax':
                value = int(value)

            data[item] = value


        for item in ['names', 'default',
            'min', 'max', 'means']:
            obj = getattr(self, item)
            if not obj is None:
                data[item] = obj.tolist()

        data['ensembles'] = []
        for iens in range(self.nens):
            self.iens = iens
            values = {'iens':iens, 'data':self.data.tolist(),
                        'hitbounds':int(self.hitbounds)}
            data['ensembles'].append(values)

        return data




class Matrix(object):

    def __init__(self, id, nval, nvar, nlead=1, nens=1, data=None, ts_index=None, prefix='V'):
        self.id = id
        self.prefix = prefix
        self._iens = 0
        self._ilead = 0

        if nval is not None and nvar is not None and \
            nens is not None and nlead is not None:
            self._nval = nval
            self._nvar = nvar
            self._nens = nens
            self._nlead = nlead
            self._data = np.empty((nval, nvar, nlead, nens), dtype=np.float64)
            self._data.fill(np.nan)

        elif data is not None:
            if data.ndim < 4:
                shape = list(data.shape) + [1] * (4-data.ndim)
                _data = data.reshape(shape)

            else:
                _data = data.copy()

            self._nval, self._nvar, self._nlead, self._nens = _data.shape
            self._data = _data

        else:
            raise ValueError(('With {0} matrix, ' + \
                    'Wrong arguments to Matrix.__init__').format(self.id))

        self._names = ['{0}{1}'.format(prefix, i) for i in range(self.nvar)]

        if not ts_index is None:
            self.ts_index = ts_index
        else:
            self.ts_index = np.arange(self._nval)


    @classmethod
    def from_dims(cls, id, nval, nvar, nlead=1, nens=1, ts_index=None, prefix='V'):
        return cls(id, nval, nvar, nlead, nens, None)


    @classmethod
    def from_data(cls, id, data):
        return cls(id, None, None, None, None, data, ts_index=None, prefix='V')

    @classmethod
    def from_hdf(cls, filename, root_path=''):

        store = pd.HDFStore(filename)

        # Browse file structure
        if not root_path.startswith('/'):
            root_path = '/' + root_path
        keys = [k for k in store.keys() if k.startswith(root_path)]

        if len(keys) == 0:
            raise ValueError(('Cannot create matrix from {0},' +
                ' no matching root path {1}').format(filename, root_path))

        # Get number of lead times and ensembles
        leads = []
        ens = []
        ids = []
        id = None
        for k in keys:
            # Remove path
            kk = re.sub('.*/', '', k)

            # Skip ts index
            if kk.endswith('ts_index'):
                continue

            # Get id, ilead and iens
            match = re.match('(?P<id>[\d\w]+)_ilead(?P<ilead>\d+)_iens(?P<iens>\d+)', kk)
            if match is None:
                raise ValueError(('Cannot create matrix from {0},' +
                    ' key {1} cannot be decoded').format(filename, kk))

            match = match.groupdict()

            if id is None:
                id = match['id']
            else:
                if id != match['id']:
                    raise ValueError(('Cannot create matrix from {0},' +
                        ' id {1} is not used for all tables (e.g., I found this {2})').format(filename,
                            id, match['id']))

            leads.append(int(match['ilead']))
            ens.append(int(match['iens']))

        nlead = np.max(leads) + 1
        nens = np.max(ens) + 1

        # Allocate matrix
        df = store[keys[0]]
        nval, nvar = df.shape
        prefix = df.columns[0][0]
        mat = Matrix.from_dims(id, nval, nvar, nlead, nens, prefix=prefix)

        # Reads ts index
        path = '{0}/{1}_ts_index'.format(root_path, id)
        df = store[path]
        mat.ts_index = df.values

        # Populate data
        for ilead, iens in product(range(nlead), range(nens)):
            mat.ilead = ilead
            mat.iens = iens

            path = '{0}/{1}_ilead{2}_iens{3}'.format(root_path, id, ilead, iens)
            df = store[path]
            mat.data = df.values

        store.close()

        return mat


    def __str__(self):
        str = 'Matrix {0} : nval={1} nvar={2} nlead={3} nens={4} {{'.format( \
            self.id, self.nval, self.nvar, self.nlead, self.nens)
        str += ', '.join(self._names)
        str += '}'

        return str


    @property
    def nvar(self):
        return self._nvar

    @property
    def nval(self):
        return self._nval

    @property
    def nlead(self):
        return self._nlead

    @property
    def nens(self):
        return self._nens

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self._names = np.atleast_1d(value)

        if len(self._names) != self.nvar:
            raise ValueError(('With {0} matrix: tried setting _names, ' + \
                'got wrong size ({1} instead of {2})').format(\
                self.id, len(self._names), self.nvar))


    @property
    def data(self):
        return self._data[:, :, self._ilead, self._iens]

    @data.setter
    def data(self, value):
        _value = np.atleast_2d(value)

        if _value.shape[0] == 1:
            _value = _value.T

        if _value.shape[1] != self.nvar:
            raise ValueError(('With {0} matrix: tried setting _data,' + \
                    ' got wrong number of values ' + \
                    '({1} instead of {2})').format( \
                    self.id, _value.shape[1], self.nvar))

        if _value.shape[0] != self.nval:
            raise ValueError(('With {0} matrix: tried setting _data,' + \
                    ' got wrong number of variables ' + \
                    '({1} instead of {2})').format( \
                    self.id, _value.shape[0], self.nval))

        self._data[:, :, self._ilead, self._iens] = _value


    @property
    def ilead(self):
        return self._ilead

    @ilead.setter
    def ilead(self, value):
        ''' Set the lead time. Checks that number is not greater that maximum forecast horizon '''

        value = int(value)
        if value >= self.nlead or value < 0:
            raise ValueError(('With {0} matrix: ilead {1} ' \
                    '>= nlead({2}) or < 0').format( \
                                self.id, value, self.nlead))

        self._ilead = value


    @property
    def iens(self):
        return self._iens

    @iens.setter
    def iens(self, value):
        ''' Set the ensemble number. Checks that number is not greater that total number of ensembles '''

        value = int(value)
        if value >= self.nens or value < 0:
            raise ValueError(('With {0} matrix: iens {1} ' \
                    '>= nens({2}) or < 0').format( \
                                self.id, value, self.nens))

        self._iens = value


    @property
    def ts_index(self):
        return self._ts_index

    @ts_index.setter
    def ts_index(self, value):
        ''' Set the times series index '''

        value = np.atleast_1d(value).astype(np.int32)
        if value.shape[0] != self.nval:
            raise ValueError(('With {0} matrix: tried to set ts_index, got {1} values,' \
                    ' expected {2}').format( \
                                self.id, value.shape[0], self.nbval))

        self._ts_index = value


    def reset(self, value=0.):
        self._data.fill(value)


    def clone(self):
        clone = Matrix.from_data(self.id, self._data,
                ts_index=self.ts_index, prefix=self.prefix)

        return clone


    def to_hdf(self, filename, root_path=''):

        store = pd.HDFStore(filename)

        # Write ts index
        path = '{0}/{1}_ts_index'.format(root_path, self.id)
        df = pd.DataFrame(self.ts_index, columns = ['ts_index'])
        store.put(path, df)

        # Write data
        for ilead, iens in product(range(self.nlead), range(self.nens)):
            self.ilead = ilead
            self.iens = iens
            df = pd.DataFrame(self.data, columns = self.names)

            path = '{0}/{1}_ilead{2}_iens{3}'.format(root_path, self.id, ilead, iens)
            store.put(path, df)

        store.close()

