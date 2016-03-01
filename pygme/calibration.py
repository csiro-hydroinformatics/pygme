import math, time
from datetime import datetime

import numpy as np

from scipy.optimize import fmin_powell

from pygme.data import Vector, Matrix

now = datetime.now


def sse(obs, sim, errparams):
    err = obs-sim
    return np.sum(err*err)


def sseabs_bias(obs, sim, errparams):
    err = np.abs(obs-sim)
    E = np.sum(err*err)
    B = np.mean(obs-sim)
    return E*(1+abs(B))


def ssqe_bias(obs, sim, errparams):
    err = np.sqrt(obs)-np.sqrt(sim)
    E = np.sum(err*err)
    B = np.mean(obs-sim)
    return E*(1+abs(B))


def sls_llikelihood(obs, sim, errparams):
    err = obs-sim
    sigma = errparams[0]
    nval = len(obs)

    ll = np.sum(err*err)/(2*sigma*sigma) + nval * math.log(sigma)
    return ll



class Calibration(object):

    def __init__(self, model,
            ncalparams,
            nens_params = 1,
            errfun=None,
            minimize=True,
            optimizer=fmin_powell,
            initialise_model=True,
            timeit=False,
            nrepeat_fit=2):

        self._model = model
        self._minimize = minimize
        self._timeit = timeit
        self._ieval = 0
        self._iprint = 0
        self._runtime = np.nan
        self._initialise_model = initialise_model
        self._status = 'intialised'
        self._nrepeat_fit = nrepeat_fit

        self._obsdata = None
        self._ipos_cal = None

        # Create vector of calibrated parameters
        # (can be different from model parameters)
        self._calparams = Vector('calparams', ncalparams, nens_params)
        self._calparams.default = np.zeros(ncalparams, dtype=np.float64)
        self._calparams.means = np.zeros(ncalparams, dtype=np.float64)
        self._calparams.covar = np.eye(ncalparams, dtype=np.float64)

        self.errfun = sse

        # Wrapper around optimizer to
        # send the current calibration object
        def _optimizer(objfun, start, calib, disp, *args, **kwargs):
            kwargs['disp'] = disp
            final = optimizer(objfun, start, *args, **kwargs)
            return final

        self._optimizer = _optimizer

    def __str__(self):
        str = ('Calibration instance ' +
                'for model {0}\n').format(self._model.name)
        str += '  status     : {0}\n'.format(self._status)
        str += '  nrepeat_fit: {0}\n'.format(self._nrepeat_fit)
        str += '  ncalparams : {0}\n'.format(self.ncalparams)
        str += '  ieval      : {0}\n'.format(self._ieval)

        return str

    @property
    def ieval(self):
        return self._ieval


    @property
    def runtime(self):
        return self._runtime


    @property
    def ncalparams(self):
        return self._calparams.nval


    @property
    def calparams(self):
        return self._calparams.data


    @calparams.setter
    def calparams(self, value):
        self._calparams.data = value


    @property
    def model(self):
        return self._model


    @property
    def obsdata(self):
        return self._obsdata.data


    @property
    def errfun(self):
        return self._errfun

    @errfun.setter
    def errfun(self, value):
        self._errfun = value

        def objfun(calparams):

            if self._timeit:
                t0 = time.time()

            # Set model parameters
            params = self.cal2true(calparams)
            self._model.params = params

            # Exit objectif function if parameters hit bounds
            if self._model._params.hitbounds and self._status:
                return np.inf

            # Run model initialisation if needed
            if self._initialise_model:
                self._model.initialise()

            # run model
            self._model.run()
            self._ieval += 1

            if self._timeit:
                t1 = time.time()
                self._runtime = (t1-t0)*1000

            # Get error model parameters if they exist
            # example standard deviation of normal gaussian error model
            errparams = self.cal2err(calparams)

            # Locate iposes in the calibration period
            kk = np.in1d(self._model._inputs.ipos, self._ipos_cal)

            # Compute objectif function
            ofun = self._errfun(self.obsdata[kk, :], \
                    self._model.outputs[kk, :], errparams)

            if not self._minimize:
                ofun *= -1

            # Print output if needed
            if self._iprint>0:
                if self._ieval % self._iprint == 0:
                    self._calparams.data = calparams
                    print('{4} {0:3d} : {1:3.3e} {2} ~ {3:.3f} ms'.format( \
                        self._ieval, ofun, self._calparams.data, \
                        self._runtime, self._status))

            return ofun

        self._objfun = objfun


    @property
    def ipos_cal(self):
        return self._ipos_cal

    @ipos_cal.setter
    def ipos_cal(self, value):
        if value.dtype == np.dtype('bool'):
            _ipos_cal = np.where(value)[0]
        else:
            _ipos_cal = value

        if self._obsdata is None:
            raise ValueError('No obsdata data. Please allocate')

        # check value is within model inputs iposes
        ipos = self._model._inputs.ipos
        if np.any(~np.in1d(_ipos_cal, ipos)):
            raise ValueError(('Certain values in ipos_cal are not within '
                'input data ipos'))

        if np.min(_ipos_cal) < self.model.ipos_start:
            raise ValueError(('Min value in ipos_cal({0})' +
                ' exceeds model.ipos_start ({1})').format(np.max(_ipos_cal),
                                                        self.model.ipos_start))
        if np.max(_ipos_cal) > self.model.ipos_end:
            raise ValueError(('Max value in ipos_cal({0})' +
                ' exceeds model.ipos_end ({1})').format(np.max(_ipos_cal),
                                                        self.model.ipos_end))
        self._ipos_cal = _ipos_cal


    def check(self):
        ''' Performs check on calibrated model to ensure that all variables are
        properly allocated '''
        # Check ipos_cal is allocated
        if self._ipos_cal is None:
            raise ValueError('No ipos_cal data. Please allocate')

        # Check obsdata are allocated
        if self._obsdata is None:
            raise ValueError('No obsdata data. Please allocate')

        # Check inputs are allocated
        if self.model._inputs is None:
            raise ValueError(('No inputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check outputs are allocated
        if self.model._outputs is None:
            raise ValueError(('No outputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check inputs and obsdata have the right dimension
        nval, nvar, _, _ = self.model.get_dims('outputs')
        nval2 = self._obsdata.nval
        if nval != nval2:
            raise ValueError(('model inputs nval({0}) !=' + \
                ' obsdata nval({1})').format(nval, nval2))

        nvar2 = self._obsdata.nvar
        if nvar != nvar2:
            raise ValueError(('model outputs nvar({0}) !=' + \
                ' obsdata nvar({1})').format(nvar, nvar2))

        # Check params size
        calparams = np.zeros(self.ncalparams)
        params = self.cal2true(calparams)
        try:
            check = (params.ndim == 1)
        except Exception:
            check = False
        if not check:
            raise ValueError('cal2true does not return a 1D Numpy array')

        errparams = self.cal2err(calparams)
        if not errparams is None:
            try:
                check = (errparams.ndim == 1)
            except Exception:
                check = False
            if not check:
                raise ValueError('cal2err does not return a Numpy 1D Numpy array')


    def cal2true(self, calparams):
        ''' Convert calibrated parameters to true values '''
        return calparams


    def cal2err(self, calparams):
        ''' Get error model parameters from the list of calibrated parameters '''
        return None


    def setup(self, obsdata, inputs):

        if inputs.nval != obsdata.nval:
            raise ValueError(('Number of value in inputs({0}) different' +
                ' from obsdata ({1})').format(inputs.nval, obdata.nval))

        if not np.allclose(inputs.ipos, obsdata.ipos):
            raise ValueError('Different iposes in obsdata and inputs')

        if obsdata.nvar > self._model.noutputs_max:
            raise ValueError(('Number of variables in outputs({0}) greater' +
                ' than model can produce ({1})').format(obsdata.nvar,
                                                self._model.noutputs_max))

        self.model.allocate(inputs, obsdata.nvar)

        self._obsdata = obsdata

        # By default calibrate on everything
        self.ipos_cal = inputs.ipos.copy()


    def explore(self, nsamples = None, iprint=0,
            distribution = 'normal'):
        ''' Systematic exploration of parameter space and
        identification of best parameter set
        '''

        self.check()
        self._iprint = iprint
        self._ieval = 0
        self._status = 'explore'
        ncalparams = self.ncalparams

        # Set the number of samples
        if nsamples is None:
            nsamples = int(200 * math.sqrt(ncalparams))

        # Get random samples from parameter
        calparams_explore = self._calparams.clone(nsamples)
        calparams_explore.random()

        # Setup objective function
        ofun_explore = np.zeros(nsamples) * np.nan
        ofun_min = np.inf

        # Systematic exploration
        calparams_best = None

        for i in range(nsamples):
            # Get parameter sample
            calparams = calparams_explore._data[i, :]

            ofun = self._objfun(calparams)
            ofun_explore[i] = ofun

            if self._iprint>0:
                if self._ieval % self._iprint == 0:
                    self.calparams = calparams

            if ofun < ofun_min:
                ofun_min = ofun
                calparams_best = calparams

        if calparams_best is None:
            raise ValueError('Could not identify a suitable' + \
                '  parameter by exploration')

        self.calparams = calparams_best

        return calparams_best, calparams_explore, ofun_explore


    def fit(self, calparams_ini, iprint=0, *args, **kwargs):

        self.check()
        self._iprint = iprint
        self._status = 'fit'
        nrepeat_fit = self._nrepeat_fit

        if np.any(np.isnan(calparams_ini)):
            raise ValueError('calparams_ini contains NaN')

        if self._iprint>0:
            ofun_ini = self._objfun(calparams_ini)

            self._calparams.data = calparams_ini
            print('\n>> Fit start: {0:3.3e} {1} ~ {2:.2f} ms <<\n'.format( \
                    ofun_ini, calparams_ini, self._runtime))

        # Apply the optimizer several times to ensure convergence
        for k in range(nrepeat_fit):
            final = self._optimizer(self._objfun, \
                    calparams_ini, self, disp=self._iprint>0, \
                    *args, **kwargs)

            calparams_ini = final

        ofun_final = self._objfun(final)
        outputs_final = self.model.outputs
        self.calparams = final

        if self._iprint>0:
            print('\n>> Fit final: {0:3.3e} {1} ~ {2:.2f} ms <<\n'.format( \
                    ofun_final, self._calparams.data, self._runtime))

        self._status = 'fit completed'

        return final, outputs_final, ofun_final


    def run(self, obsdata, inputs, \
            ipos_cal=None, \
            iprint=0, \
            nsamples=None,
            *args, **kwargs):

        self.setup(obsdata, inputs)

        if ipos_cal is None:
            ipos_cal = np.arange(self._obsdata.nval)
        self.ipos_cal = ipos_cal

        try:
            start, explo, explo_ofun = self.explore(iprint=iprint, \
                nsamples=nsamples)
        except ValueError:
            start = self.model._params.default

        kwargs['iprint'] = iprint
        final, out, out_ofun = self.fit(start, *args, **kwargs)

        return final, out, out_ofun


    def sensitivity(self, calparams=None, dx=1e-3):

        if calparams is None:
            calparams = self._calparams.data

        ncalparams = self._calparams.nval
        sensitivity = np.zeros(ncalparams)

        ofun0 = self._objfun(calparams)

        for k in range(ncalparams):
            cp = calparams.copy()
            cp[k] += dx
            ofun1 = self._objfun(cp)
            sensitivity[k] = (ofun1-ofun0)/dx

        return sensitivity




class CrossValidation(object):

    def __init__(self, calib, explore=True):

        self._calib = calib
        self._explore = explore

        self._scheme = None
        self._calparams_ini = None
        self._calperiods = []

        self._obsdata = calib._obsdata.clone()
        nval = self._obsdata.nval
        self._mask = np.ones(nval, dtype=bool)


    def __str__(self):
        calib = self._calib
        str = ('Cross-validation instance for' +
                ' model {0}\n').format(calib._model.name)
        str += '  scheme     : {0}\n'.format(self._scheme)
        str += '  explore    : {0}\n'.format(self._explore)
        str += '  ncalparams : {0}\n'.format(calib.ncalparams)
        str += '  nperiods   : {0}\n'.format(len(self._calperiods))

        return str


    @property
    def calib(self):
        return self._calib


    @property
    def calperiods(self):
        return self._calperiods


    @property
    def calparams_ini(self):
        return self._calparams_ini

    @calparams_ini.setter
    def calparams_ini(self, value):
        self._calparams_ini = Vector('calparams_ini', \
                self._calib._calparams.nval)

        self._calparams_ini.data = value


    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        _value = np.atleast_1d(value, dtype=bool)

        nval = calib._obsdata.nval
        if _value.shape[0] != nval:
            raise ValueError(('Length of mask({0}) does not match ' +
                'nval of obsdata').format(_value.shape[0], nval))

        self._mask = mask


    def set_periods(self, scheme='split', nperiods=2, warmup=0,
            nleaveout=None):
        ''' Define calibration periods '''

        self._calib.check()
        nval, _, _, _ = self._calib.model.get_dims('inputs')
        nvalper = int((nval-warmup)/nperiods)

        # Exclude nan from calibration
        check = np.sum(np.isnan(self._obsdata.data), axis=1) == 0
        self.mask[np.where(check)[0]] = False

        # Set default nleaveout
        if nleaveout is None:
            nleaveout = nvalper

        # Build calibration sub-periods
        self._calperiods = []

        if scheme in ['split', 'leaveout']:

            for i in range(nperiods):

                if scheme == 'split':
                    ipos_cal_start = i*nvalper
                    ipos_cal_end = warmup + (i+1)*nvalper-1

                    if ipos_cal_start > ipos_cal_end:
                        raise ValueError('ipos_cal_start({0}) > ' +
                                'ipos_cal_end({1})'.format(ipos_cal_start,
                                    ipos_cal_end))

                    ipos_cal_leaveout = []

                    ipos_val_start = 0
                    ipos_val_end = nval-1
                    ipos_val_leaveout = np.arange(ipos_cal[0],
                            ipos_cal[-1]+1)

                else:
                    ipos_cal_start = 0
                    ipos_cal_end = nval-1

                    i1 = i*nvalper + warmup
                    i2 = i1 + nleaveout

                    if i1 > i2:
                        raise ValueError(('ipos_cal_leaveout[0]({0}) > ' +
                                'ipos_cal_leaveout[-1]({1})').format(i1, i2))

                    if i2 >= nval:
                        break
                    ipos_cal_leaveout = np.arange(i1, i2)
                    import pdb; pdb.set_trace()

                    idx_val = np.arange(i1, i2)
                    idx_val_leaveout = []


                per = {
                    'scheme':scheme,
                    'id':'CALPER{0}'.format(i+1),
                    'ipos_cal_start': ipos_cal_start,
                    'ipos_cal_end': ipos_cal_end,
                    'ipos_cal_leaveout':ipos_cal_leaveout,
                    'ipos_val_start': ipos_val_start,
                    'ipos_val_end': ipos_val_end,
                    'ipos_val_leaveout':idx_val_leaveout,
                    'warmup':warmup,
                    'log':{},
                    'completed':False
                }
                self._calperiods.append(per)
        else:
            raise ValueError('Cross validation scheme {0} ' +
                    'not recognised'.format(self._scheme))

        if len(self._calperiods) == 0:
            raise ValueError('No calibration period generated,' +
                    ' modify inputs')


    def run(self, *args, **kwargs):

        nper = len(self._calperiods)

        if nper == 0:
            raise ValueError('No calibration periods defined, please setup')

        for i in range(nper):
            # Set calibration range
            per = self._calperiods[i]
            per['log'][now()] = 'Calibration of subperiod {0} started'.format(i)

            self.calib.model.ipos_start = per['ipos_start']
            self.calib.model.ipos_end = per['ipos_end']
            self.calib.ipos_cal = per['ipos_cal']

            # Set leave out period
            if len(per['ipos_cal_leaveout'])>0:
                self.calib._obsdata = self._obsdata.clone()
                self.calib.obsdata[per['ipos_cal_leaveout']] = np.nan

            # Define starting point
            if self._explore:
                try:
                    calparams_ini, _, _ = self._calib.explore()
                    per['log'][now()] = 'Explore completed'
                except ValueError, e:
                    per['log'][now()] = 'Explore failed, {0}'.format(e)
                    calparams_ini = self.calib._calparams.default
            else:
                calparams_ini = self.calib._calparams.default

            # Perform fit for calibration period
            per['log'][now()] = 'Fit started'

            try:
                final, out, ofun = self._calib.fit(calparams_ini,
                                            *args, **kwargs)
                per['log'][now()] = 'Fit completed'

                per['completed'] = True
                per['calparams'] = final.copy()
                per['params'] = self._calib.model.params.copy()
                per['simulation'] = out
                per['objfun'] = ofun

            except Exception, e:
                per['log'][now()] = 'Fit failed, {0}'.format(e)




