
import numpy as np

from useme.model import Model
from useme.calibration import Calibration
from useme import calibration


def standardize(X, cst=None):

    U = X
    if not cst is None:
        U = np.log(X+cst)

    U = np.atleast_2d(U)
    if U.shape[0] == 1:
        U = U.T

    mu = np.nanmean(U, 0)
    su = np.nanstd(U, 0)
    Un = (U-mu)/su

    return Un, mu, su


def destandardize(Un, mu, su, cst=None):
    U = mu + Un * su
    if not cst is None:
        X = np.exp(U) - cst

    return X


class ANN(Model):

    def __init__(self, ninputs, nneurons):

        self.nneurons = nneurons

        nparams = (ninputs + 2) * nneurons + 1

        noutputs_max = nneurons + 1

        Model.__init__(self, 'ann',
            nconfig=1, \
            ninputs=ninputs, \
            nparams=nparams, \
            nstates=1, \
            noutputs_max = noutputs_max,
            inputs_names = ['I{0}'.format(i) for i in range(ninputs)], \
            outputs_names = ['L2N1'] + \
                ['L1N{0}'.format(i) for i in range(1, nneurons+1)])

        self.config.names = ['dummy']
        self.config.units = ['-']

        self.states.names = ['dummy']
        self.states.units = ['-']

        self.params.units = ['-'] * nparams
        self.params.min = [-10.] * nparams
        self.params.max = [10.] * nparams
        self.params.default = [0.] * nparams

        self.params.reset()

    def params2idx(self):
        ''' Returns indices of parameters in the parameter vector '''
        nneurons = self._noutputs_max - 1
        ninputs = self.ninputs

        idx = np.arange(self.params.nval)

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
        nneurons = self._noutputs_max - 1
        ninputs = self.ninputs

        params = self.params.data
        idxL1W, idxL1B, idxL2W, idxL2B = self.params2idx()

        # Parameter for first layer
        L1W = params[idxL1W].reshape(ninputs, nneurons)
        L1B = params[idxL1B].reshape(1, nneurons)

        # Parameter for second layer
        L2W = params[idxL2W].reshape(nneurons, 1)
        L2B = params[idxL2B].reshape(1, 1)

        return L1W, L1B, L2W, L2B


    def jacobian(self):
        ''' Returns the jacobian of parameters '''
        idxL1W, idxL1B, idxL2W, idxL2B = self.params2idx()
        L1W, L1B, L2W, L2B = self.params2matrix()

        self.run()
        nval = self.outputs.nval
        nparams = self.params.nval
        jac = np.zeros((nval, nparams))
        inputs = self.inputs.data
        S = self.run_layer1()
        L1 = np.dot(inputs, L1W) + L1B

        # Jacobian for first layer
        tanh_L1 = np.tanh(L1)
        mult = np.diagflat(L2W)
        X = np.dot(1-tanh_L1*tanh_L1, mult)
        jac[:, idxL1B] = X

        ninputs = inputs.shape[1]
        nx = X.shape[1]
        for k in range(ninputs):
            for l in range(nx):
                col = idxL1W[k*nx + l]
                jac[:, col] = inputs[:, k] * X[:, l]


        # Jacobian for second layer
        jac[:, idxL2W] = S
        jac[:, idxL2B] = np.ones((nval, 1))

        return jac


    def run_layer1(self):
        ''' Compute outputs from first layer '''
        L1W, L1B, _, _ = self.params2matrix()
        return np.tanh(np.dot(self.inputs.data, L1W) + L1B)

    def run_layer2(self):
        ''' Compute outputs from second layer '''
        _, _, L2W, L2B = self.params2matrix()
        S = self.run_layer1()
        return np.dot(S, L2W) + L2B

    def run(self):

        # First layer
        S = self.run_layer1()

        # Second layer
        O = self.run_layer2()

        n3 = self.outputs.nvar
        self.outputs.data =  np.concatenate([O, S], axis=1)[:, :n3]



class CalibrationANN(Calibration):

    def __init__(self, ninputs, nneurons, \
        nepochs=30, \
        timeit=False):

        self.nepochs = nepochs

        ann = ANN(ninputs, nneurons)
        nparams = ann.params.nval

        def dummy_optimizer(objfun, start, *args, **kwargs):
            return start

        Calibration.__init__(self,
            model = ann, \
            ncalparams = nparams, \
            optimizer = dummy_optimizer, \
            timeit = timeit)

        idxL1W, idxL1B, idxL2W, idxL2B = ann.params2idx()
        means = np.zeros(nparams)
        means[idxL1W] = 1.
        means[idxL2W] = 1.
        self.calparams_means.data =  means

        stdevs = np.eye(nparams).flat[:]
        self.calparams_stdevs.data = stdevs

        self.errfun = calibration.sse


    def post_fit(self, calparams):

        # Set model parameters
        ann = self.model
        ann.params.data = self.cal2true(calparams)
        ann.run()
        out = ann.outputs.data

        if out.shape[1] > 1:
            raise ValueError('Can only calibrate ANN with 1 output')

        # Initialise variables
        idx_cal = self.idx_cal
        obs = self.observations.data[idx_cal, :]
        sse = calibration.sse(obs, out[idx_cal, :], None)
    
        nparams = ann.params.nval
        gamk = float(nparams)
        nberr = float(obs.shape[0])
        beta = (nberr-gamk)/(2*sse)

        params = ann.params.data
        ssx = np.sum(params*params)
        alpha = gamk/(2*ssx)

        perf = beta*sse + alpha*ssx

        MU = 0.005
        MU_NLOOPMAX = 100
        MU_MAX = 1e10

        jac = ann.jacobian()
        jac = jac[idx_cal, :]
        JJ = np.dot(jac.T, jac)
        err = obs-out[idx_cal,:]
        JE = np.dot(jac.T, err)
        II = np.eye(nparams)


        for epoch in range(1, self.nepochs+1):
            
            while MU <= MU_MAX:
                # Update parameters
                dX = -(beta*JJ+II*(MU+alpha))/(beta*JE+alpha*params)
                params2 = params+dX
                ssx2 = np.sum(params2*params2)

                # Run model with updated parameters
                ann.params.data = params2
                import pdb; pdb.set_trace()
                ann.run()
                out = ann.outputs.data
                sse2 = calibration.sse(obs, out[idx_cal, :], None)
                perf2 = beta*sse2 + alpha*ssx2


