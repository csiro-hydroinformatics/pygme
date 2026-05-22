import numpy as np

from hydrodiy.data.containers import Vector
from pygme.model import Model, UH, ParamsVector
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

from pygme import has_c_module
if has_c_module("models_hydromodels"):
    import c_pygme_models_hydromodels

HAYAMI_MAXUH = c_pygme_models_hydromodels.hayami_get_maxuh()
HAYAMI_UHEPS = c_pygme_models_hydromodels.hayami_get_uheps()

# Transformation functions for hayami parameters
def hayami_trans2true(x):
    return x


def hayami_true2trans(x):
    return x


def hayami_compute_theta(L, C, Z):
    return c_pygme_models_hydromodels.hayami_compute_theta(L, C, Z)


def hayami_compute_D(L, C, Z):
    return c_pygme_models_hydromodels.hayami_compute_D(L, C, Z)


# Hayami kernel function
def hayami_kernel(theta, z, t=None):
    if t is None:
        eps = 1e-3
        bounds = np.zeros(2)
        c_pygme_models_hydromodels.time_bounds_hayami(theta, z, eps, bounds)
        tlow, thigh = bounds
        t = np.linspace(tlow, thigh, 1000)

    theta = np.float64(theta)
    z = np.float64(z)
    t = np.array(t, dtype=np.float64)
    out = np.empty((t.shape[0], 4), dtype=np.float64)
    c_pygme_models_hydromodels.hayami_kernel_vect(theta, z, t, out)
    return out


class HayamiUH(UH):
    def __init__(self, config, niter=6, nordmax=HAYAMI_MAXUH):
        super(HayamiUH, self).__init__("lag", nordmax=nordmax)
        self._theta = config.timestep
        self._Z = 2.5
        self._config = config

        # Number of sub-timestep iterations
        # to compute hayami kernel
        self._niter = niter

    @property
    def config(self):
        return self._config

    @property
    def niter(self):
        return self._niter

    def set_uh(self, C, Z):
        L = self.config.length
        theta = hayami_compute_theta(L, C, Z)
        self._theta = theta
        self._Z = Z

        # Populate the uh ordinates
        ierr = c_pygme_models_hydromodels.uh_getuh_hayami(self.nordmax,
                                                          self.niter,
                                                          self.config.timestep,
                                                          theta, Z,
                                                          self._nord, self._ord)
        if ierr > 0:
            errmsg = f"When setting theta={theta:0.3f} and Z={Z:0.3f}"\
                     +" for Hayami UH, "\
                     + f"c_pygme_models_hydromodels.uh_getuh_hayami returns {ierr}"
            raise ValueError(errmsg)

        # Reset uh states to a vector of zeros
        # with length nord
        self._states[:self.nord] = 0

        # Set remaining ordinates to 0
        self._ord[self.nord:] = 0

    @property
    def theta(self):
        return self._theta

    @property
    def Z(self):
        return self._Z

    def clone(self):
        """ Generates a clone of the current UH """
        clone = HayamiUH(self.nordmax)
        clone.hayami_params = self.hayami_params
        clone._states = self.states.copy()
        clone._ord = self.ord.copy()

        return clone


class HayamiParamsVector(ParamsVector):
    def __init__(self, params, config, checkvalues=None):
        super(HayamiParamsVector, self).__init__(params)
        self._hayami_uh = HayamiUH(config)

    def _set_values(self):
        C, Z = self.values
        self._hayami_uh.set_uh(C, Z)
        super()._set_values()

    @property
    def nuh(self):
        return 0

    @property
    def uhs(self):
        return [(None, self._hayami_uh)]


class Hayami(Model):

    def __init__(self):

        # Config vector
        # default timestep in sec = daily (=86400 sec)
        # default reach length in m = 10km
        config = Vector(["timestep", "length",
                         "lateral"],
                        defaults=[86400, 1e4, 0],
                        mins=[1, 1, 0],
                        maxs=[np.inf, np.inf, 1])

        # params vector
        # C [m/s] -> celerity
        # Z [dimless] -> shape
        vect = Vector(["C", "Z"],
                      defaults=[1., 2.5],
                      mins=[1e-3, 1e-3],
                      maxs=[1e2, 1e2])
        params = HayamiParamsVector(vect, config)

        # State vector
        states = Vector(["Qsum", "Vr"])

        # Model
        super(Hayami, self).__init__("Hayami",
                                     config, params, states,
                                     ninputs=1,
                                     noutputsmax=2)

        self.inputs_names = ["Inflow"]
        self.outputs_names = ["Q", "VR"]


    def initialise(self, states=None, uhs=None):
        super(Hayami, self).initialise(states, uhs)
        self.states.Qsum = 0
        self.states.Vr = 0

    @property
    def ord(self):
        return self.params.uhs[0][1].ord

    def run(self):
        has_c_module("models_hydromodels")

        # Get uh object (not set_params function, see ParamsVector class)
        _, uh = self.params.uhs[0]

        ierr = c_pygme_models_hydromodels.hayami_run(uh.nord,
                                                     self.istart,
                                                     self.iend,
                                                     self.config.values,
                                                     self.params.values,
                                                     uh.ord,
                                                     self.inputs,
                                                     uh.states,
                                                     self.states.values,
                                                     self.outputs)

        if ierr > 0:
            errmsg = "c_pygme_models_hydromodels."\
                     + f"hayami_run returns {ierr}"
            raise ValueError(errmsg)


class CalibrationHayami(Calibration):

    def __init__(self, objfun=ObjFunBCSSE(0.5),
                 warmup=5*365,
                 timeit=False,
                 fixed=None,
                 nparamslib=400,
                 objfun_kwargs={}):

        # Input objects for Calibration class
        model = Hayami()
        params = model.params

        cp = Vector(["C", "Z"],
                    mins=params.mins,
                    maxs=params.maxs,
                    defaults=params.defaults)

        # no parameter transformation
        calparams = CalibParamsVector(model, cp,
                                      trans2true=hayami_trans2true,
                                      true2trans=hayami_true2trans,
                                      fixed=fixed)

        # Instanciate calibration
        super(CalibrationHayami, self).__init__(calparams,
                                                  objfun=objfun,
                                                  warmup=warmup,
                                                  timeit=timeit,
                                                  objfun_kwargs=objfun_kwargs)

        # Build parameter library from
        # systematic exploration of parameter space
        nn = 10
        C = np.linspace(0.1, 10, nn)
        Z = np.linspace(0.1, 10, nn)
        ee, zz = np.meshgrid(C, Z)
        self.paramslib = np.column_stack([ee.ravel(), zz.ravel()])
