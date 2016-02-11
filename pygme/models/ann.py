
import numpy as np

from pygme.data import Matrix
from pygme.model import Model
from pygme.calibration import Calibration


def get_standardize_params(X, cst=None):

    U = X
    if not cst is None:
        U = np.log(X+cst)

    U = np.atleast_2d(U)
    if U.shape[0] == 1:
        U = U.T

    mu = np.nanmean(U, 0)
    sig = np.nanstd(U, 0)

    return {'mu':mu, 'sig':sig, 'cst':cst}


def standardize(X, params):
    mu = params['mu']
    sig = params['sig']

    U = X

    if 'cst' in params:
        cst = params['cst']
        if not cst is None:
            U = np.log(X+cst)

    U = np.atleast_2d(U)
    if U.shape[0] == 1:
        U = U.T

    #mu = np.nanmean(U, 0)
    #sig = np.nanstd(U, 0)
    Un = (U-mu)/sig

    return Un


def destandardize(Un, params):
    mu = params['mu']
    sig = params['sig']
    cst = params['cst']

    X = mu + Un * sig
    if not cst is None:
        X = np.exp(X) - cst

    return X


class ANN(Model):

    def __init__(self, ninputs, nneurons,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        self.nneurons = nneurons

        nparams = (ninputs + 2) * nneurons + 1

        noutputs_max = nneurons + 1

        Model.__init__(self, 'ann',
            nconfig=1,
            ninputs=ninputs,
            nparams=nparams,
            nstates=1,
            noutputs_max = noutputs_max,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self.config.names = ['dummy']
        self.config.units = ['-']

        self._params.units = ['-'] * nparams
        self._params.min = [-10.] * nparams
        self._params.max = [10.] * nparams
        self._params.default = [0.] * nparams

        self.reset()

        # Covariance matrix of parameters
        self._params_covar = Matrix.from_dims('params_covar', nparams, nparams)

        # Additional data to handle transformation
        self._inputs_trans = None
        self._inputs_trans_params = None

        self._outputs_trans = None
        self._outputs_trans_params = {'mu':0., 'sig':1., 'cst':None}


    @property
    def inputs_trans(self):
        return self._inputs_trans


    @property
    def outputs_trans(self):
        return self._outputs_trans


    @property
    def inputs_trans_params(self):
        return self._inputs_trans_params

    @inputs_trans_params.setter
    def inputs_trans_params(self, value):
        self._inputs_trans_params = { \
            'mu':value['mu'], \
            'sig':value['sig'], \
            'cst':value['cst'] \
        }


    @property
    def outputs_trans_params(self):
        return self._outputs_trans_params

    @outputs_trans_params.setter
    def outputs_trans_params(self, value):
        self._outputs_trans_params = { \
            'mu':value['mu'], \
            'sig':value['sig'], \
            'cst':value['cst'] \
        }


    def allocate(self, nval, noutputs=1, nens_inputs=1):
        if noutputs !=1:
            raise ValueError('Number of outputs defined for ANN model should be 1')

        super(ANN, self).allocate(nval, noutputs, nens_inputs)

        ninputs = self._inputs.nvar
        self._inputs_trans = Matrix.from_dims('inputs_trans', nval, ninputs)
        self._outputs_trans = Matrix.from_dims('outputs_trans', nval, 1)


    def standardize_inputs(self):
        params = self._inputs_trans_params
        self._inputs_trans.data = standardize(self._inputs.data, params)


    def standardize_outputs(self):
        params = self._outputs_trans_params
        self._outputs_trans.data = standardize(self._outputs.data, params)



    def params2idx(self):
        ''' Returns indices of parameters in the parameter vector '''
        nneurons = self.noutputs_max - 1
        ninputs = self.ninputs
        nparams, _ = self.get_dims('params')

        idx = np.arange(nparams)

        n1 = ninputs*nneurons

        # Parameter for first layer
        idxL1W = np.arange(n1)
        idxL1B = np.arange(n1, n1+nneurons)

        # Parameter for second layer
        idxL2W = np.arange(n1+nneurons, n1+2*nneurons)
        idxL2B = np.arange(n1+2*nneurons, n1+2*nneurons+1)

        return idxL1W, idxL1B, idxL2W, idxL2B


    def params2matrix(self):
        ''' Returns parameter matrices from the parameter vector '''
        nneurons = self.noutputs_max - 1
        ninputs = self.ninputs

        params = self._params.data
        idxL1W, idxL1B, idxL2W, idxL2B = self.params2idx()

        # Parameter for first layer
        L1W = params[idxL1W].reshape(ninputs, nneurons)
        L1B = params[idxL1B].reshape(1, nneurons)

        # Parameter for second layer
        L2W = params[idxL2W].reshape(nneurons, 1)
        L2B = params[idxL2B].reshape(1, 1)

        return L1W, L1B, L2W, L2B


    def run_layer1(self):
        ''' Compute outputs from first layer '''
        L1W, L1B, _, _ = self.params2matrix()
        return np.tanh(np.dot(self.inputs_trans.data, L1W) + L1B)


    def run_layer2(self, S):
        ''' Compute outputs from second layer '''
        _, _, L2W, L2B = self.params2matrix()
        return np.dot(S, L2W) + L2B


    def run(self):
        ''' Run model '''
        S = self.run_layer1()
        O = self.run_layer2(S)
        self.outputs_trans.data =  O

        params = self.outputs_trans_params
        self.outputs = destandardize(O, params)


    def jacobian(self):
        ''' Returns the jacobian of parameters '''
        idxL1W, idxL1B, idxL2W, idxL2B = self.params2idx()
        L1W, L1B, L2W, L2B = self.params2matrix()

        self.run()
        nval, _, _, _ = self.get_dims('inputs')
        nparams, _ = self.get_dims('params')
        jac = np.zeros((nval, nparams))
        inputs_trans = self.inputs_trans.data

        # Jacobian for first layer
        S = self.run_layer1()
        L1 = np.dot(inputs_trans, L1W) + L1B

        tanh_L1 = np.tanh(L1)
        mult = np.diagflat(L2W)
        X = np.dot(1-tanh_L1*tanh_L1, mult)
        jac[:, idxL1B] = X

        ninputs = inputs_trans.shape[1]
        nx = X.shape[1]
        for k in range(ninputs):
            for l in range(nx):
                col = idxL1W[k*nx + l]
                jac[:, col] = inputs_trans[:, k] * X[:, l]


        # Jacobian for second layer
        jac[:, idxL2W] = S
        jac[:, idxL2B] = np.ones((nval, 1))

        return jac



def ann_optimizer(objfun, start, calib, disp, *args, **kwargs):
    ''' Specific ANN optimizer implementing Levenberg-Mararquardt
    back propagation with Bayesian regularization '''

    # Set model parameters
    ann = calib.model
    ann.params.data = calib.cal2true(start)
    ann.run()
    out = ann.outputs_trans.data

    if out.shape[1] > 1:
        raise ValueError('Can only calibrate ANN with 1 output')

    # Initialise variables
    idx_cal = calib.idx_cal
    obs = calib._obsdata_trans.data[idx_cal, :]
    sse = calibration.sse(obs, out[idx_cal, :], None)

    nparams = ann.nparams
    gamk = float(nparams)
    nberr = float(obs.shape[0])

    # Output uncertainty
    beta = (nberr-gamk)/(2*sse)

    if beta<0:
        raise ValueError(('More parameters ({0}) than' + \
            ' obs({1})!'.format(nparams, nberr)))

    params = ann.params.reshape((nparams, 1))
    ssx = np.sum(params*params)

    # Parameter uncertainty
    alpha = gamk/(2*ssx)

    perf = beta*sse + alpha*ssx

    # Algorithm parameters
    MU = 0.005
    MU_NLOOPMAX = 100
    MU_MAX = 1e10
    MU_INC = 10
    MU_DEC = 10

    for epoch in range(1, calib.nepochs+1):
        # Compute Jacobian
        jac = ann.jacobian()
        jac = jac[idx_cal, :]

        # Compute errors updating matrices
        JJ = np.dot(jac.T, jac)
        err = obs-out[idx_cal,:]
        JE = np.dot(jac.T, err)
        II = np.eye(nparams)

        while MU <= MU_MAX:
            # Update parameters
            A = -(beta*JJ+II*(MU+alpha))
            B = beta*JE+alpha*params

            dX = np.linalg.solve(A, B)
            params2 = params+dX
            ssx2 = np.sum(params2*params2)

            # Run model with updated parameters
            ann.params = params2[:, 0]
            ann.run()
            out = ann.outputs_trans.data
            sse2 = calibration.sse(obs, out[idx_cal, :], None)
            perf2 = beta*sse2 + alpha*ssx2
            import pdb; pdb.set_trace()

            if perf2 < perf:
                sse = sse2
                ssx = ssx2
                perf = perf2
                params = params2
                ann.params.data = params2[:, 0]
                MU *= MU_DEC
                break

            MU *= MU_INC

        if disp & (epoch % calib._iprint == 0):
            print(format(('  EPOCH {0:2d}:\n' + \
                '\t\tmu   = {6:3.3e}\n' + \
                '\t\tsse   = {1:3.3e}\n' + \
                '\t\tssx   = {2:3.3e}\n' + \
                '\t\tgamk  = {3:3.3e}\n' + \
                '\t\talpha = {4:3.3e}\n' + \
                '\t\tbeta  = {5:3.3e}\n').format(epoch, sse, \
                    ssx, gamk, alpha, beta, MU)))

        if MU <= MU_MAX:
            A = beta*JJ+II*alpha
            Ainv = np.linalg.inv(A)
            gamk = float(nparams)-alpha*np.trace(Ainv)
            alpha = gamk/(2*ssx)
            beta = (nberr-gamk)/(2*sse)
            perf = beta*sse + alpha*ssx
        else:
            if disp:
                print('MU_MAX reached, stop at epoch {0}\n'.format(epoch))
            break

    ann.params = params.flat[:]

    A = beta*JJ+II*alpha
    Ainv = np.linalg.inv(A)
    ann._params.covar = Ainv

    return params


class CalibrationANN(Calibration):

    def __init__(self, ninputs, nneurons, \
        nepochs=30, \
        timeit=False):

        self.nepochs = nepochs

        ann = ANN(ninputs, nneurons)
        nparams = ann.nparams

        def dummy_optimizer(objfun, start, *args, **kwargs):
            return start

        Calibration.__init__(self,
            model = ann, \
            ncalparams = ann.nparams, \
            optimizer = dummy_optimizer, \
            timeit = timeit, \
            nrepeat_fit = 1)

        idxL1W, idxL1B, idxL2W, idxL2B = ann.params2idx()
        means = np.zeros(nparams)
        means[idxL1W] = 1.
        means[idxL2W] = 1.
        self._calparams.means =  means

        covar = np.eye(nparams)
        self._calparams.covar = covar

        # Setup errfun (not used in the fitting procedure, only for exploration)
        self.errfun = calibration.sse

        # Modify optimizer to accomodate for ANN procedure
        self._optimizer = ann_optimizer

        # Transformed obs
        self._obsdata_trans = None

        # Constants for transformation
        self.cst_inputs = 0.
        self.cst_outputs = 0.


    @property
    def obsdata_trans(self):
        return self._obsdata_trans


    def setup(self, obs, inputs):

        super(CalibrationANN, self).setup(obs, inputs)

        params = get_standardize_params(inputs.data, self.cst_inputs)
        self.model._inputs_trans_params = params
        self.model._inputs_trans.data = standardize(inputs.data, params)

        params = get_standardize_params(obs.data, self.cst_outputs)
        self.model._outputs_trans_params = params

        nval = self._obsdata.nval
        nvar = self._obsdata.nvar
        self._obsdata_trans = Matrix.from_dims('obsdata_trans', nval, nvar)
        self._obsdata_trans.data = standardize(obs.data, params)


