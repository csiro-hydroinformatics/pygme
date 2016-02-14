import os
import re
import unittest

import datetime
import time

import numpy as np
import pandas as pd

from pygme.models.knn import KNN, dayofyear
from pygme import calibration
from pygme.model import Matrix

class KNNTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> KNNTestCase')
        filename = os.path.abspath(__file__)
        self.FHERE = os.path.dirname(filename)


    def test_print(self):
        nval = 5000
        nvar = 5
        var = np.random.uniform(0, 1, (nval, nvar))
        weights = np.ones(nval)

        kn = KNN(var, weights)

        str_kn = '%s' % kn


    def test_knn_dumb(self):
        return

        cycle = 10
        nval = 4 * cycle
        nvar = 1
        cpi = 3

        tmp = np.arange(cycle)[range(cpi, cycle) + range(cpi)]
        tmp = tmp.reshape((cycle, 1)).repeat(nval/cycle, 1)
        tmp = tmp + 0.01* np.arange(tmp.shape[1])
        var = tmp.T.flat[:]
        kn = KNN(var)

        kn.config['halfwindow'] = 1
        kn.config['nb_nn'] = 3
        kn.config['cycle_length'] = cycle
        kn.config['cycle_position_ini'] = cpi

        nrand = 10
        kn.allocate(nrand)

        value = 8.3
        states = [value, int(value)]
        kn.initialise(states)

        kn.run(seed=333)

        expected = np.arange(cycle)[range(int(value)+1, cycle) + range(int(value)+1)]
        ck = np.allclose(np.floor(kn.outputs[:,0]), expected)
        self.assertTrue(ck)


    def test_knn_rainfall(self):
        ''' Test to check that KNN can reproduce rainfall stats '''

        # Function to compute rainfall stats
        def stats(x):
            return pd.Series({'mean':np.sum(x.values),
                    'std':np.std(x.values),
                    'plow':float(np.sum(x.values<1))/x.shape[0]})

        fp = '{0}/data/GR4J_params.csv'.format(self.FHERE)
        params = np.loadtxt(fp, delimiter=',')

        nsample = 100
        nsites = params.shape[0]

        for i in [1, 7, 6]:#np.random.choice(range(nsites), nsites, False):

            fts = '{0}/data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.FHERE, i+1)
            data = np.loadtxt(fts, delimiter=',')

            dates = pd.Series(data[:,0]).apply(lambda x:
                            datetime.datetime.strptime('{0:0.0f}'.format(x),
                                '%Y%m%d'))

            #doy = dates.apply(dayofyear)
            #nkern = 30
            #cum = np.convolve(data[:, 1], np.ones(nkern), 'same')[:, None]
            #var = np.concatenate([p_pe, cum], axis=1)
            #doys = np.sin(doy.astype(float)/365*2*np.pi) * np.mean(data[:, 1])
            #var = np.concatenate([data[:, 1][:, None], doys[:, None]], axis=1)

            #var = np.atleast_2d(data[:, 2]).T

            var = np.concatenate([data[2:, [1,2]],
                data[1:-1, [1, 2]], data[:-2, [1,2]]], axis=1)
            var = np.ascontiguousarray(var)
            kn = KNN(var)

            kn.config['halfwindow'] = 10
            kn.config['nb_nn'] = 6
            kn.config['cycle_position_ini'] = 0

            cycle = 365.25
            kn.config['cycle_length'] = cycle

            nrand = data.shape[0]
            #nrand = 5
            kn.allocate(nrand)

            # KNN sample
            rain = {}
            idx = {}
            dta = 0
            states = var[0,:].copy().tolist() + [0]

            for k in range(nsample):
                t0 = time.time()

                kn.initialise(states)
                seed = np.random.randint(0, 1000000)
                kn.run(seed)

                #import pdb; pdb.set_trace()
                #import matplotlib.pyplot as plt
                #plt.close('all')
                #x = np.arange(nrand) % 365.25

                #kk = np.arange(500) #len(x))
                ##plt.plot(kn.knn_idx[kk] % cycle, '-o')
                #plt.plot(x[kk], kn.knn_idx[kk] % cycle, '-o')
                #plt.plot(x[kk], x[kk], '--')
                #plt.savefig(os.path.join(self.FHERE, 'tmp.png'))
                #import pdb; pdb.set_trace()

                t1 = time.time()
                dta += 1000 * (t1-t0)

                # Get rainfall stats
                nm = 'R{0:03d}'.format(k)
                tmp = pd.Series(kn.outputs[:,0], index=dates)
                tmp2 = tmp.groupby(tmp.index.month).apply(stats).reset_index()
                tmp3 = pd.pivot_table(tmp2, index='level_0', columns='level_1')[0]
                rain[nm] = tmp3

                # Get nn month
                tmp4 = [{'month':d.month, 'doy':d.timetuple().tm_yday} for d in dates[kn.knn_idx]]
                idx[nm] = pd.DataFrame(tmp4, index=dates).groupby(tmp.index.month).mean()

            dta = dta/nsample/nrand * 3650

            # Check rainfall is sampled from proper month
            idx = pd.Panel(idx).mean(0)
            errm = idx['month'].values - np.arange(1,13)
            ee_month = np.max(np.abs(errm[1:-1]))

            # Check rainfall stats are correct
            rain = pd.Panel(rain)
            rain_qt1 = rain.apply(np.percentile, 0, q=30)
            rain_qt2 = rain.apply(np.percentile, 0, q=70)

            tmp = pd.Series(data[:, 1], index=dates)
            tmp = tmp.groupby(tmp.index.month).apply(stats).reset_index()
            rain_obs = pd.pivot_table(tmp, index='level_0', columns='level_1')[0]

            errv = (rain_obs - rain_qt1) >= 0
            errv = errv & ((rain_qt2 - rain_obs) >= 0)
            ee_value = (~errv).sum()

            ck = (ee_month < 5e-2)
            ck = ck & (np.max(ee_value) <= 3)

            if ck:
                print(('\t\tTEST KNN RAINFALL {0:02d} : ' +
                    'runtime = {1:0.5f}ms/10years').format(i+1, dta))
            else:
                print(('\t\tTEST KNN RAINFALL {0:02d} : ' +
                    'runtime = {1:0.5f}ms/10years\n\t\t  KNN FAILED TO BRACKET OBS (nb month): ' +
                    'ee_month={5:0.2f} ' +
                    'mean={2:0.0f} plow={3:0.0f} std={4:0.0f}').format(i+1, dta,
                        ee_value['mean'], ee_value['plow'], ee_value['std'], ee_month))



if __name__ == '__main__':
    unittest.main()
