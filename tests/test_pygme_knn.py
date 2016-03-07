import os
import re
import unittest

import datetime
import time
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pygme.models.knn import KNN, dayofyear
from pygme import calibration
from pygme.model import Matrix

from hystat.linreg import Linreg
from hyplot import putils


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


def plot_knn(kn, fp, cycle, nmax):
    plt.close('all')

    kk = np.arange(nmax)
    plt.plot(kn.knn_idx[kk] % cycle, '-o')
    plt.plot(kk, kk % cycle, '--')

    xx = np.arange(0, nmax, cycle)
    plt.plot(xx, np.zeros(len(xx)), 'k*', markersize=20)

    plt.savefig(fp)



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
        tmp = tmp + 0.1* np.arange(tmp.shape[1])
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


    def test_knn_seasonality(self):
        halfwin = 10
        nb_nn = 5
        cycle = 100

        ncycle = 30
        nval = ncycle * cycle
        nrand = 6 # nval
        nvar = 1
        cpi = cycle/3*2

        # Identical input data (so KNN will select neighbours randomly)
        input_var = np.ones(nval)

        # Cyclical input data
        input_var = np.sin((np.arange(nval)+cpi).astype(float)/cycle * 2 * math.pi)
        input_var = input_var*cycle/2
        #input_var[1:] += np.random.uniform(-2, 2, len(input_var)-1)

        idx = np.array([range(cpi, cycle) + range(cpi)]).reshape((cycle, 1))
        output_var = np.repeat(idx, ncycle, axis=1).T.flat[:]
        output_var = np.concatenate([output_var[:, None],
                                np.arange(nval)[:, None],
                                input_var[:, None]], axis=1)
        kn = KNN(input_var, output_var = output_var)

        kn.config['halfwindow'] = halfwin
        kn.config['nb_nn'] = nb_nn
        kn.config['cycle_length'] = cycle
        kn.config['cycle_position_ini'] = cpi

        kn.allocate(np.ones(nrand), output_var.shape[1])

        states = [input_var[0], cpi]
        kn.initialise(states)
        kn.run(seed=333)

        if nrand == nval:
            res = pd.DataFrame({'knn_pos':kn.outputs[:, 0],
                                    'knn_idx':kn.outputs[:,1],
                                    'knn_value':kn.outputs[:,2],
                                    'data_pos':output_var[:,0],
                                    'data_value':output_var[:,2]})
            res = res[:2*cycle]
            diff = res['knn_pos'] - res['data_pos']

            # Check that there is no drift in simulation
            kk = np.arange(nval)
            idx = np.abs(diff) < 2*halfwin
            lm = Linreg(diff[idx].index, diff[idx])
            lm.fit()
            p1, p2 = lm.params['Pr(>|t|)']
            #self.assertTrue(p1 > 0.5 and p2 > 0.5)

            plt.close('all')
            fig = plt.figure()
            gr = GridSpec(2, 2, width_ratios = [3, 1])

            ax = plt.subplot(gr[0, 0])
            res.loc[:, ['knn_pos', 'data_pos']].plot(ax=ax, legend=False)

            ax = plt.subplot(gr[1, 0])
            res.loc[:, ['knn_value', 'data_value']].plot(ax=ax, legend=False)

            ax = plt.subplot(gr[0, 1])
            diff.plot(ax=ax)
            putils.line(ax, p1, p2, lw=3)

            ax = plt.subplot(gr[1, 1])
            dd = res.loc[:, ['data_pos', 'knn_pos']]
            dd.boxplot(ax=ax, by='data_pos', sym='')
            ax.grid('off')
            ax.set_xticks(range(0,  cycle, 10))
            ax.set_xticklabels(range(0,  cycle, 10))
            putils.line(ax, 0, 1, 'k--', lw=2)

            fig.set_size_inches((24, 8))
            gr.tight_layout(fig)
            fp = os.path.join(self.FHERE, 'tmp.png')
            fig.savefig(fp)

        import pdb; pdb.set_trace()

    def test_knn_rainfall(self):
        ''' Test to check that KNN can reproduce rainfall stats '''

        return

        lf = [os.path.join(self.FHERE, 'data', f)
                for f in os.listdir(os.path.join(self.FHERE, 'data'))
                    if f.startswith('KNNTEST')]

        nsample = 100

        halfwin = 10
        nb_nn = 5
        lag = 0

        for i in [1]:

            fts = os.path.join(self.FHERE, 'data', lf[i])
            data = pd.read_csv(fts, comment='#', index_col=0, parse_dates=True)

            dates = data.index
            data = data.values

            # Build lag matrix
            d = []
            nval = data.shape[0]
            for l in range(lag+1):
                d.append(data[l:nval-lag+l, :])
            var_out = np.concatenate(d, axis=1)

            # Configure KNN
            var_in = var_out
            kn = KNN(input_var = var_in, output_var = var_out)

            kn.config['halfwindow'] = halfwin
            kn.config['nb_nn'] = nb_nn
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
