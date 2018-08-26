import numpy as np
import time
import logging

from scipy.optimize import fmin

from hydrodiy.stat import transform, metrics
from hydrodiy.data.containers import Vector

# Setup login
LOGGER = logging.getLogger(__name__)

# Setup box cox transform
BC = transform.BoxCox2()


def format_array(x, fmt='3.3e'):
    return ' '.join([('{0:'+fmt+'}').format(u) for u in x])


class ObjFun(object):
    ''' Generic class to describe objective functions '''

    def __init__(self, name, orientation=1):

        self.name = name

        if not orientation in [-1, 1]:
            raise ValueError(('Expected orientation to be -1 or 1,'+\
                ' got {0}').format(orientation))

        self.orientation = orientation


    def __str__(self):
        return '{0} objective function, orientation {1}'.format(\
                    self.name, self.orientation)


    def compute(self, obs, sim, **kwargs):
        raise ValueError('Need to override this function')



class ObjFunSSE(ObjFun):
    ''' Sum of squared error objective function '''

    def __init__(self):
        super(ObjFunSSE, self).__init__('SSE', 1)


    def compute(self, obs, sim, **kwargs):
        err = obs.squeeze()-sim.squeeze()
        return np.nansum(err*err)



class ObjFunKGE(ObjFun):
    ''' KGE objective function '''

    def __init__(self):
        super(ObjFunKGE, self).__init__('KGE', -1)


    def compute(self, obs, sim, **kwargs):
        return metrics.kge(obs.squeeze(), sim.squeeze())



class ObjFunBCSSE(ObjFun):
    ''' Sum of squared error objective function
        for BC transformed flows.

        See transform class in package hydrodiy
        hydrodiy.stat.transforms.BoxCox1

    '''

    def __init__(self, lam=0.5, eta=1e-3):
        super(ObjFunBCSSE, self).__init__('BCSSE', 1)

        # Set Transform
        self.eta = eta
        BC.lam = lam
        BC.nu = 0.
        self.trans = BC

    def compute(self, obs, sim, **kwargs):
        # Mean shift factor
        BC.nu = np.nanmean(obs)*self.eta

        # Transform data
        tobs = self.trans.forward(obs)
        tsim = self.trans.forward(sim)

        # Compute errors
        t2 = time.time()
        err = tobs.squeeze()-tsim.squeeze()
        return np.nansum(err*err)


def check_vector(x, nval):
    ''' Check vector value '''
    if not isinstance(x, np.ndarray):
        raise ValueError('data is a not numpy array')

    if not x.ndim == 1:
        raise ValueError('data is not a 1d numpy array')

    if not len(x) == nval:
        raise ValueError('Expected vector of '+\
            'length {0}, got {1}'.format(nval, len(x)))

    if np.any(np.isnan(x)):
        raise ValueError('Expected no nan values in vector, '+\
            'got {0}'.format(xd))



# Overload Vector class to include parameter transform
class CalibParamsVector(Vector):

    def __init__(self, model, tparams=None, trans2true=None,\
            true2trans=None, fixed=None):
        ''' Object to handle calibrated parameters

        Parameters
        -----------
        model : pygme.model.Model
            Model to calibrate
        tparams : hydrodiy.containers.Vector
            Vector of calibrated model parameters
        trans2true : function
            Function to transform calibrated parameters into
            model parameters
        true2trans : function
            Function to transform model parameters into
            calibrated parameters
        fixed : dict
            Dictionary listing the fixed model parameters and
            their values. Example {'X1':100.}
        '''
        # Default values
        if tparams is None:
            tparams = model.params.clone()

        # Initialise Vector object
        super(CalibParamsVector, self).__init__(tparams.names, \
            tparams.defaults, tparams.mins, tparams.maxs, \
            tparams.hitbounds)

        # Check fixed
        if not fixed is None:
            names = model.params.names
            check = [k in names for k in fixed]
            if not np.all(check):
                raise ValueError('Expected names of fixed parameters '+\
                    'to be in {0}, got {1}'.format(names, \
                        list(fixed.keys())))

        self._fixed = fixed

        # Check mins and maxs are set
        if np.any(np.isinf(tparams.mins)):
            raise ValueError('Expected no infinite values in mins, '+
                'got {0}'.format(tparams.mins))

        if np.any(np.isinf(tparams.maxs)):
            raise ValueError('Expected no infinite values in maxs, '+
                'got {0}'.format(tparams.maxs))

        # Syntactic sugar for common transforms
        if trans2true == 'sinh':
            trans2true = np.sinh
            true2trans = np.arcsinh
        elif trans2true == 'exp':
            trans2true = np.exp
            true2trans = np.log

        # Set default transform as identity
        if (trans2true is None and not true2trans is None)\
            or (not trans2true is None and true2trans is None):
            raise ValueError('Expected both trans2true and true2trans '+\
                'to be either valid or None, got {0} and {1}'.format(\
                    trans2true, true2trans))

        if trans2true is None:
            trans2true = lambda x: x
            true2trans = lambda x: x

        # Check transforms applied to default values
        xd = trans2true(self.defaults)
        try:
            check_vector(xd, model.params.nval)
        except ValueError as err:
            raise ValueError(\
                'Problem with trans2true for default: {0}'.format(str(err)))

        # Check back transforms applied to default values
        xtd = true2trans(xd)
        try:
            check_vector(xtd, self.nval)
        except ValueError as err:
            raise ValueError('Problem with true2trans: {0}'.format(str(err)))

        # Check trans2true is one to one
        xmi = trans2true(self.mins)
        try:
            check_vector(xmi, model.params.nval)
        except ValueError as err:
            raise ValueError(\
                'Problem with trans2true for min: {0}'.format(str(err)))

        xma = trans2true(self.maxs)
        try:
            check_vector(xmi, model.params.nval)
        except ValueError as err:
            raise ValueError(\
                'Problem with trans2true for max: {0}'.format(str(err)))

        if np.any((xd-xmi)<0):
            raise ValueError('Expected transform of defaults to be greater'+\
                ' than transform of mins, got '+\
                't(defaults)={0} and t(mins)={1}'.format(xd, xmi))

        if np.any((xma-xd)<0):
            raise ValueError('Expected transform of maximum to be greater'+\
                ' than transform of defaults, got '+ \
                't(maxs)={0} and t(defaults)={1}'.format(xmax, xd))

        # Check transform followed by backtransform are neutral
        xd2 = trans2true(xtd)

        xtmi = true2trans(xmi)
        xmi2 = trans2true(xtmi)

        xtma = true2trans(xma)
        xma2 = trans2true(xtma)

        for v1, v2 in [[xd, xd2], [xmi, xmi2], [xma, xma2]]:
            if not np.allclose(v1, v2):
                raise ValueError('Expected trans2true followed by '+\
                    'true2trans to return the original vector, '+\
                    'got {0} -> {1}'.format(v1, v2))

        # Store data
        self._trans2true = trans2true
        self._true2trans = true2trans
        self._model = model
        model.params.values = trans2true(self.defaults).copy()


    def __setitem__(self, key, values):
        # Set item for the vector
        Vector.__setitem__(self, key, values)

        # transform values
        truev = self.trans2true(self.values)
        params = self._model.params
        params.values = truev

        # Set fixed
        fixe = self.fixed
        if not fixed is None:
            for pname, pvalue in fixed.items():
                params[pname] = pvalue


    @property
    def model(self):
        return self._model


    @property
    def fixed(self):
        return self._fixed


    @property
    def trans2true(self):
        return self._trans2true


    @property
    def true2trans(self):
        return self._true2trans


    @property
    def truevalues(self):
        return self._model.params.values

    @truevalues.setter
    def truevalues(self, values):
        # Set model params
        params = self._model.params
        params.values = values

        # Set fixed
        fixed = self.fixed
        if not fixed is None:
            for pname, pvalue in fixed.items():
                params[pname] = pvalue

        # Run the vector value setter
        tvalues = self.true2trans(self.model.params.values)
        Vector.values.fset(self, tvalues)


    @Vector.values.setter
    def values(self, values):
        # Run the vector value setter
        Vector.values.fset(self, values)

        # Set true values
        params = self._model.params
        params.values = self.trans2true(self.values)

        # Set fixed
        fixed = self.fixed
        if not fixed is None:
            for pname, pvalue in fixed.items():
                params[pname] = pvalue



def fitfun(values, calib, use_transformed_parameters):
    ''' Objective function wrapper to  be used by optimizer.
        Can be run with transformed or untransformed parameter values.
    '''
    # Get objects
    calparams = calib.calparams
    model = calib.model
    ical = calib.ical
    objfun = calib.objfun
    objfun_kwargs = calib.objfun_kwargs
    initial_kwargs = calib.initial_kwargs

    # Set model parameters
    # (note that parameters are passed to model within calparams object)
    if use_transformed_parameters:
        calparams.values = values
    else:
        calparams.truevalues = values

    # Exit objectif function if parameters hit bounds
    if model.params.hitbounds and calib.hitbounds:
        return np.inf

    # Initialise model
    model.initialise_fromdata(**initial_kwargs)

    # Run model with runtime assessment
    if calib.timeit:
        t0 = time.time()

    model.run()
    calib._nbeval += 1

    if calib.timeit:
        t1 = time.time()
        calib._runtime = (t1-t0)*1000

    # Compute objectif function during calibration period
    # Pass additional argument to obj fun
    ofun = objfun.compute(calib.obs[ical, :], \
                                    model.outputs[ical, :], \
                                    **objfun_kwargs)

    if np.isnan(ofun):
        ofun = np.inf

    # Apply function orientation
    # +1 = minimization
    # -1 = maximization
    ofun *= objfun.orientation

    # Check obj fun is not nan
    if np.isnan(ofun) or ofun == -np.inf:
        raise ValueError('Objective function is NaN or -inf for '+\
                'params={0}'.format(values))

    # Print output if needed
    if calib.iprint>0 and calib.nbeval % calib.iprint == 0:
        LOGGER.info(('Fitfun [{0}]: {1}({2}) = {3:3.3e} '+\
            '~ {4:.3f} ms').format( \
                calib.nbeval, objfun.name, format_array(\
                    calib.calparams.truevalues), \
                ofun, calib.runtime))

    return ofun



class Calibration(object):

    def __init__(self, calparams, \
            warmup=0, \
            objfun=ObjFunSSE(), \
            timeit=False, \
            hitbounds=True,
            objfun_kwargs={},\
            initial_kwargs={}):

        # Initialise calparams
        calparams.truevalues = calparams.model.params.defaults
        self._calparams = calparams

        self._objfun = objfun
        self._objfun_kwargs = objfun_kwargs
        self._initial_kwargs = initial_kwargs

        self.warmup = warmup
        self.timeit = timeit
        self.hitbounds = hitbounds
        self.iprint = 0

        self._nbeval = 0
        self._runtime = np.nan
        self._obs = None
        self._ical = None
        self._paramslib = None
        self._nparamslib = 0

    def __str__(self):
        str = ('Calibration instance ' +
                'for model {0}\n').format(self.model.name)
        str += '  ncalparams : {0}\n'.format(self.calparams.nval)
        str += '  objfun     : {0}\n'.format(self.objfun.name)
        str += '  warmup     : {0}\n'.format(self.warmup)
        str += '  hitbounds  : {0}\n'.format(self.hitbounds)
        str += '  nbeval     : {0}\n'.format(self.nbeval)
        str += '  nparamslib : {0}\n'.format(self.nparamslib)

        return str


    @property
    def runtime(self):
        return self._runtime


    @property
    def nbeval(self):
        return self._nbeval


    @property
    def objfun(self):
        return self._objfun


    @property
    def objfun_args(self):
        return self._objfun_args


    @property
    def objfun_kwargs(self):
        return self._objfun_kwargs


    @property
    def initial_kwargs(self):
        return self._initial_kwargs


    @property
    def fixed(self):
        return self._calparams.fixed


    def _check_paramslib(self):
        if self._paramslib is None:
            raise ValueError('Trying to access paramslib, but it is '+\
                        'not allocated.')

    @property
    def nparamslib(self):
        self._check_paramslib()
        return self._paramslib.shape[0]

    @property
    def paramslib(self):
        self._check_paramslib()
        return self._paramslib

    @paramslib.setter
    def paramslib(self, values):
        # Check dimensions
        values = np.atleast_2d(values).astype(np.float64)
        calparams = self.calparams
        model = calparams.model
        nlib, nparams = values.shape
        if nparams != model.params.nval:
            raise ValueError(('Expected {0} parameters in '+\
                'paramslib, got {1}').format(\
                    model.params.nval, \
                    nparams))

        if np.any(np.isnan(values)):
            raise ValueError('Expected no NaN in paramslib')

        # Clip paramslib within model parameter boundaries
        plib = values * np.nan
        mins = model.params.mins
        maxs = model.params.maxs

        for ip, row in enumerate(values):
            clipped = np.clip(row, mins, maxs)

            if not np.allclose(row, clipped):
                LOGGER.warning('Clipped parameter set '+\
                    '#{0} from paramslib: {1} -> {2}'.format(ip,\
                        row, clipped))

            # Check values is valid with parameter transform
            tclipped = calparams.true2trans(clipped)
            if np.any(np.isnan(tclipped)):
                raise ValueError('Parameter set '+\
                    '#{0} is invalid after transform: {1}'.format(ip, \
                        tclipped))

            clipped2 = calparams.trans2true(tclipped)
            if np.any(np.isnan(clipped2)):
                raise ValueError('Parameter set '+\
                    '#{0} is invalid after back transform: {1}'.format(\
                        ip,  clipped2))

            plib[ip, :] = clipped

        self._paramslib = plib


    @property
    def calparams(self):
        return self._calparams


    @property
    def model(self):
        return self._calparams.model


    @property
    def obs(self):
        if self._obs is None:
            raise ValueError('Trying to get obs, but it is '+\
                        'not allocated. Please allocate')

        return self._obs


    @property
    def ical(self):
        return self._ical

    @ical.setter
    def ical(self, values):
        nval = self.obs.shape[0]

        # Set to all indexes if None
        if values is None:
            ical = np.arange(nval)

        else:
            values = np.atleast_1d(values)

            if values.dtype == np.dtype('bool'):
                if values.shape[0] != nval:
                    raise ValueError(('Expected boolean ical of length {0},'+\
                            ' got {1}').format(nval, values.shape[0]))

                # Convert boolean to index
                ical = np.where(values)[0]

            else:
                ical = np.sort(values.astype(int))

        # check value is within obs indexes
        iout = (ical<0) | (ical>=nval)
        if np.sum(iout)>0:
            out = ical[iout]
            raise ValueError('Expected all values in ical to be '+
                'in [0, {0}], got {1} (first five only)'.format(nval-1,\
                out[:5]))

        # Check value leaves enough data for warmup
        if ical[0] < self.warmup:
            raise ValueError('Expected ical[0]>{1}, got {0}'.format(\
                    ical[0], self.warmup))

        self._ical = ical


    def allocate(self, obs, inputs):
        ''' Allocate model inputs and obs data '''

        # Convert outputs to numpy 2d array
        obs = np.atleast_2d(obs).astype(np.float64)
        if obs.shape[0] == 1 and obs.shape[1]>1:
            obs = obs.T

        nval, noutputs = obs.shape
        inputs = np.atleast_2d(inputs).astype(np.float64)

        # Check inputs and outputs size
        if inputs.shape[0] != nval:
            raise ValueError(('Expected same number of timestep '+\
                'in inputs({0}) and outputs({1})').format(\
                    inputs.shape[0], nval))

        if noutputs>self.model.noutputsmax:
            raise ValueError('Expected number of outputs to be '+\
                'lower than {0}, got {1}'.format(self.model.noutputsmax,
                    noutputs))

        # Allocate model
        self.model.allocate(inputs, noutputs)

        # Set obs data
        self._obs = obs

        # By default calibrate on everything excluding warmup
        self.ical = np.arange(self.warmup, nval)

        LOGGER.info('Calibration data allocated')


    def explore(self, iprint=0):
        ''' Systematic exploration of parameter space and
        identification of best parameter set
        '''
        self.iprint = iprint
        self._nbeval = 0
        LOGGER.info('Parameter exploration started')

        # Get data
        paramslib = self.paramslib
        nlib, _ = paramslib.shape

        # Initialise
        ofuns = np.zeros(nlib) * np.nan
        ofun_min = np.inf
        best = None

        # Systematic exploration of parameter library
        for i, values in enumerate(paramslib):
            # Run fitfun with untransformed parameters
            try:
                ofun = fitfun(values, calib=self, \
                        use_transformed_parameters=False)
            except ValueError as err:
                LOGGER.error(str(err))
                ofun = np.inf

            ofuns[i] = ofun

            # Store minimum of objfun
            if ofun < ofun_min:
                ofun_min = ofun

                # We use self.model.params.values
                # instead of values because
                # some param values may be changed by objfun
                # (e.g. fixed)
                best = self.model.params.values

        raise_it = True if best is None else False
        raise_it = True if not np.isfinite(ofun_min) else False
        if raise_it:
            raise ValueError('Could not identify a suitable' + \
                '  parameter set by exploration')

        # Set model parameters
        self.calparams.truevalues = best

        LOGGER.info(('End of explore [{0}]: '+\
                '{1}({2}) = {3:3.3e} ~ {4:.3f} ms').format( \
                self.nbeval, self.objfun.name, \
                format_array(self.calparams.truevalues), \
                ofun_min, self.runtime))

        LOGGER.info('Parameter exploration completed')

        return best, ofun_min, ofuns


    def fit(self, start=None, iprint=10, nrepeat=1, optimizer=fmin, \
                **kwargs):
        ''' Fit model using the supplied optmizer '''

        LOGGER.info('Parameter fit started')

        self.iprint = iprint
        self._nbeval = 0

        # check inputs
        if start is None:
            start = self.calparams.values

        start = np.atleast_1d(start).astype(np.float64)
        if np.any(np.isnan(start)):
            raise ValueError('Expected no NaN in start, got {0}'.format(\
                start))

        if not 'disp' in kwargs:
            kwargs['disp'] = 0

        # First run of fitfun
        fitfun_start = fitfun(start, calib=self, \
                            use_transformed_parameters=True)

        # Apply the optimizer several times to ensure convergence
        calparams = self.calparams

        for k in range(nrepeat):

            # Run optimizer using fitfun with transformed parameters
            tfinal = optimizer(fitfun, \
                        start, args=(self, True, ), \
                        **kwargs)

            calparams.values = tfinal
            fitfun_final = fitfun(tfinal, calib=self, \
                                use_transformed_parameters=True)

            # Loop
            start = tfinal

        # Get final model parameters
        # (certain parameters may be fixed)
        final = self.model.params.values
        outputs_final = self.model.outputs

        LOGGER.info(('End of fit [{0}]: {1}({2}) = '+\
            '{3:3.3e} ~ {4:.3f} ms').format( \
                self.nbeval, self.objfun.name, \
                    format_array(self.calparams.values), \
            fitfun_final, self.runtime))

        LOGGER.info('Parameter fit completed')

        return final, fitfun_final, outputs_final


    def workflow(self, obs, inputs, \
            ical=None, \
            iprint=0, \
            nrepeat=1, \
            optimizer=fmin, \
            **kwargs):

        LOGGER.info('Calibration workflow started')

        # 1. allocate data
        self.allocate(obs, inputs)

        # 2. set ical if needed
        if not ical is None:
            self.ical = ical

        # 3. Run exploration
        try:
            start, _, ofun_explore = self.explore(iprint=iprint)
        except ValueError as err:
            LOGGER.error('error in parameter exploration: {0}'.format(\
                            str(err)))
            start = self.model.params.defaults
            ofun_explore = None

        # 4. Run fit
        self.calparams.truevalues = start
        tstart = self.calparams.values

        final, ofun_final, outputs_final = self.fit(tstart, \
                                    iprint=iprint, nrepeat=nrepeat, \
                                    optimizer=optimizer, \
                                    **kwargs)

        LOGGER.info('Calibration workflow completed')

        return final, ofun_final, outputs_final, ofun_explore


