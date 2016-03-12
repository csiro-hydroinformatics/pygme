import os, sys, re
import unittest

from datetime import datetime
import time
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pygme.models.knndaily import KNNDaily
from pygme import calibration
from pygme.data import Matrix, set_seed

import c_pygme_models_utils

from hystat.linreg import Linreg
from hyplot import putils

set_seed(333)


# Utility function to compute rainfall stats
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

    tmp3b = pd.pivot_table(tmp2b.reset_index(),
                index='level_0', columns='level_1')[0]
    tmp3['rho'] = (tmp3b['mean2'] - tmp3['mean2']**2)/ tmp3['std']**2

    return tmp3.loc[:, ['mean', 'plow', 'std', 'rho']]


def plot_knndaily(kn, fp, cycle, nmax):
    plt.close('all')

    kk = np.arange(nmax)
    plt.plot(kn.knndaily_idx[kk] % cycle, '-o')
    plt.plot(kk, kk % cycle, '--')

    xx = np.arange(0, nmax, cycle)
    plt.plot(xx, np.zeros(len(xx)), 'k*', markersize=20)

    plt.savefig(fp)



class KNNDailyTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> KNNDailyTestCase')
        filename = os.path.abspath(__file__)
        self.FHERE = os.path.dirname(filename)

        FIMG = os.path.join(self.FHERE, 'images')
        if not os.path.exists(FIMG):
            os.mkdir(FIMG)

        self.FIMG = FIMG

    def test_print(self):
        nval = 5000
        nvar = 5
        var = np.random.uniform(0, 1, (nval, nvar))
        weights = np.ones(nval)
        kn = KNNDaily(var, weights)
        str_kn = '%s' % kn


    def test_knndaily_seasonality(self):
        halfwin = 10
        nb_nn = 5
        dini = datetime(2000, 4, 11)
        nyears = 30

        # Cyclical input data
        dt = pd.date_range(dini, datetime(dini.year + nyears - 1, 12, 31))
        knnvar_inputs0 = np.array([c_pygme_models_utils.dayofyear(dd.month, dd.day) for dd in dt])
        knnvar_inputs = knnvar_inputs0 + 2*np.random.uniform(-1, 1, len(knnvar_inputs0))

        # Output var
        knnvar_outputs = np.concatenate([knnvar_inputs0[:, None],
                                knnvar_inputs[:, None]], axis=1)
        kn = KNNDaily(knnvar_inputs, knnvar_outputs = knnvar_outputs)

        kn.config['halfwindow'] = halfwin
        kn.config['nb_nn'] = nb_nn
        kn.config['date_ini'] = dini.year * 1e4 + dini.month * 1e2 + dini.day

        nrand = kn.knnvar_outputs.shape[0]
        kn.allocate(np.random.uniform(0, 1, nrand),
                kn.knnvar_outputs.shape[1])

        states = [knnvar_inputs[0], kn.config['date_ini']]
        kn.initialise(states)
        kn.run()

        # Check that there is no drift
        res = pd.DataFrame({'knn_pos':kn.outputs[:, 0],
                                'data_pos':kn.knnvar_outputs[:,0],
                                'diff':kn.outputs[:, 0] - kn.knnvar_outputs[:,0]})


        kk = np.abs(res['diff']) < 300
        ck = np.max(np.abs(res.loc[kk, 'diff'])) < halfwin + 3

        med = res.loc[:, ['data_pos', 'knn_pos']].groupby('data_pos').median().values
        err = np.abs(med[halfwin+3:-halfwin-3, 0]-np.arange(halfwin+3, 365-halfwin-3))

        ck = np.all(err <= halfwin)
        self.assertTrue(ck)


    def test_knndaily_rainfall(self):
        ''' Test to check that KNNDaily can reproduce rainfall stats '''

        lf = [os.path.join(self.FHERE, 'data', f)
                for f in os.listdir(os.path.join(self.FHERE, 'data'))
                    if f.startswith('KNNTEST')]

        nsample = 30
        halfwin = 10
        nb_nn = 5
        lag = 0

        for i in range(len(lf)):

            fts = os.path.join(self.FHERE, 'data', lf[i])
            data = pd.read_csv(fts, comment='#', index_col=0, parse_dates=True)
            data = pd.DataFrame(data.iloc[:, 0])

            dates = data.index

            # Build lag matrix
            d = []
            nval = data.shape[0]
            for l in range(lag+1):
                d.append(data.shift(l))
            var_in = pd.concat(d, axis=1).values[lag:, :]

            var_out = var_in[:, 0]
            dates = dates[lag:]

            # Configure KNNDaily
            kn = KNNDaily(knnvar_inputs = var_in, knnvar_outputs = var_out)

            kn.config['halfwindow'] = halfwin
            kn.config['nb_nn'] = nb_nn
            kn.config['randomize_ties'] = 1
            kn.config['date_ini'] = dates[0].year * 1e4 + dates[0].month * 1e2 + dates[0].day

            nrand = var_in.shape[0]
            kn.allocate(np.ones(nrand), kn.knnvar_outputs.shape[1])

            # KNNDaily sample
            rain = {}
            idx = {}
            dta = 0
            states = var_in[0,:].copy().tolist() + [kn.config['date_ini']]

            for k in range(nsample):
                if k%5 == 0:
                    print('\t\t TEST {0:2d} - Run {1:3d}/{2:3d}'.format(i, k, nsample))
                t0 = time.time()

                kn.initialise(states)
                kn.inputs = np.random.uniform(0, 1, nrand)
                kn.run()

                t1 = time.time()
                dta += 1000 * (t1-t0)

                # Get rainfall stats
                nm = 'R{0:03d}'.format(k)
                rain[nm] = compute_stats(kn.outputs[:,0], dates)

            dta = dta/nsample/nrand * 365

            # Compute quantiles of rainfall stats
            rain = pd.Panel(rain)
            rain_qt = {}
            for qt in range(10, 100, 10):
                rain_qt[qt] = rain.apply(np.percentile, 0, q=qt)

            rain_obs = compute_stats(kn.knnvar_outputs[:, 0], dates)

            # Check simulated rainfall stats are bracketing obs stats
            errv = (rain_obs - rain_qt[10]) >= 0
            errv = errv & ((rain_qt[90] - rain_obs) >= 0)
            ee_value = (~errv).sum().mean()
            ck = ee_value < 2
            self.assertTrue(ck)

            # Plot stats
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
            fig.savefig(os.path.join(self.FIMG,
                ('test_pygme_knndaily_rainfall_stats_' +
                    'site[{3}]_order[{0}]_nkk[{1}]_win[{2}].png').format(
                        lag, kn.config['halfwindow'], kn.config['nb_nn'], i)))

            print(('\t\tTEST KNNDaily RAINFALL {0:02d} : ' +
                  'runtime = {1:0.5f}ms/10years').format(i+1, dta))


if __name__ == '__main__':
    unittest.main()
