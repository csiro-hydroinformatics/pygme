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

def rain_stats(x, plow, nmonths):
    xv = x.values
    return pd.Series({'mean':np.nansum(xv)/nmonths,
            'mean2':np.nanmean(xv),
            'std':np.nanstd(xv),
            'plow':float(np.nansum(xv<plow))/x.shape[0]})

def compute_stats(rain, dates, plow=1):
    tmp = pd.DataFrame({'rain':rain}, index=dates)
    tmp['lag1_prod'] = tmp['rain'].shift(1) * tmp['rain']

    # -> number of months
    nmonths = tmp.resample('MS', 'sum').shape[0]

    # -> mean/std/percentage of low values (<1)
    tmp2 = tmp['rain'].groupby(tmp.index.month).apply(rain_stats,
                plow, nmonths)

    tmp3 = pd.pivot_table(tmp2.reset_index(),
        index='level_0', columns='level_1')[0]

    # -> lag 1 corr
    tmp2b = tmp['lag1_prod'].groupby(tmp.index.month).apply(rain_stats,
                plow, nmonths)

    tmp3b = pd.pivot_table(tmp2b.reset_index(), index='level_0', columns='level_1')[0]
    tmp3['rho'] = (tmp3b['mean2'] - tmp3['mean2']**2)/ tmp3['std']**2

    return tmp3.loc[:, ['mean', 'plow', 'std', 'rho']]


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
        return

        fp = '{0}/data/GR4J_params.csv'.format(self.FHERE)
        params = np.loadtxt(fp, delimiter=',')

        nsample = 100
        nsites = params.shape[0]
        lag = 0

        for i in [6]:#np.random.choice(range(nsites), nsites, False):

            fts = '{0}/data/GR4J_timeseries_{1:02d}.csv'.format( \
                    self.FHERE, i+1)
            data = np.loadtxt(fts, delimiter=',')

            dates = pd.Series(data[:,0]).apply(lambda x:
                            datetime.datetime.strptime('{0:0.0f}'.format(x),
                                '%Y%m%d'))

            # Build lag matrix
            d = []
            nval = data.shape[0]
            for l in range(lag+1):
                d.append(data[l:nval-lag+l, [1, 2]])
            var_out = np.concatenate(d, axis=1)

            # Configure KNN
            var_in = var_out
            kn = KNN(input_var = var_in, output_var = var_out)

            kn.config['halfwindow'] = 5
            kn.config['nb_nn'] = 8
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
            states = var_in[0,:].copy().tolist() + [0]

            for k in range(nsample):
                t0 = time.time()

                kn.initialise(states)
                seed = np.random.randint(0, 1000000)
                kn.run(seed)

                #import matplotlib.pyplot as plt
                #plt.close('all')
                #x = np.arange(nrand) % 365.25
                #kk = np.arange(1000) #len(x))
                #plt.plot(kn.knn_idx[kk] % cycle, '-o')
                #plt.plot(kk, kk, '--')

                #xx = np.arange(0, np.max(kk), cycle)
                #plt.plot(xx, np.zeros(len(xx)), 'x', markersize=20)
                ##plt.plot(x[kk], kn.knn_idx[kk] % cycle, '-o')
                ##plt.plot(x[kk], x[kk], '--')
                #plt.savefig(os.path.join(self.FHERE, 'tmp.png'))
                #import pdb; pdb.set_trace()

                t1 = time.time()
                dta += 1000 * (t1-t0)

                # Get rainfall stats
                nm = 'R{0:03d}'.format(k)
                rain[nm] = compute_stats(kn.outputs[:,0], dates)

                # Get month number
                tmp = pd.DataFrame([{'month':d.month, 'doy':d.timetuple().tm_yday}
                    for d in dates[kn.knn_idx]], index=dates)
                idx[nm] = tmp.groupby(tmp.index.month).mean()

            dta = dta/nsample/nrand * 3650

            # Check rainfall is sampled from proper month
            idx = pd.Panel(idx).mean(0)
            errm = idx['month'].values - np.arange(1,13)
            ee_month = np.max(np.abs(errm[1:-1]))

            # Check rainfall stats are correct
            rain = pd.Panel(rain)
            rain_qt = {}
            for qt in range(10, 100, 10):
                rain_qt[qt] = rain.apply(np.percentile, 0, q=qt)

            rain_obs = compute_stats(data[:, 1], dates)

            errv = (rain_obs - rain_qt[30]) >= 0
            errv = errv & ((rain_qt[70] - rain_obs) >= 0)
            ee_value = (~errv).sum()

            ck = (ee_month < 5e-2)
            ck = ck & (np.max(ee_value) <= 3)

            # Plot stats
            import matplotlib.pyplot as plt
            plt.close('all')
            fig, axs = plt.subplots(nrows=2, ncols=2)
            axs = axs.flat[:]

            for k, nm in enumerate(['mean', 'std', 'plow', 'rho']):
                ax = axs[k]
                ax.set_title(nm)

                rain_obs[nm].plot(ax=ax, color='r', linewidth=3)

                rain_qt[50][nm].plot(ax=ax, linewidth=3, legend=False, color='b')
                for qt in [10, 90]:
                    rain_qt[qt][nm].plot(ax=ax, legend=False, color='b')

            fig.set_size_inches(12, 12)
            fig.tight_layout()
            fig.savefig(os.path.join(self.FHERE,
                ('test_pygme_knn_rainfall_stats_' +
                    'order[{0}]_nkk[{1}]_win[{2}].png').format(
                        lag, kn.config['halfwindow'], kn.config['nb_nn'])))

            if ck:
                print(('\t\tTEST KNN RAINFALL {0:02d} : ' +
                    'runtime = {1:0.5f}ms/10years').format(i+1, dta))
            else:
                print(('\t\tTEST KNN RAINFALL {0:02d} : ' +
                    'runtime = {1:0.5f}ms/10years\n\t\t' +
                    '  KNN FAILED TO BRACKET OBS (nb month): ' +
                    'ee_month={5:0.2f} ' +
                    'mean={2:0.0f} plow={3:0.0f} std={4:0.0f}').format(i+1, dta,
                        ee_value['mean'], ee_value['plow'],
                        ee_value['std'], ee_month))



if __name__ == '__main__':
    unittest.main()
