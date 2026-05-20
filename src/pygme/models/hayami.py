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
        self._z = 2.5
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

    def set_uh(self, theta, z):
        self._theta = theta
        self._z = z

        # Populate the uh ordinates
        ierr = c_pygme_models_hydromodels.uh_getuh_hayami(self.nordmax,
                                                          self.niter,
                                                          self.config.timestep,
                                                          theta, z,
                                                          self._nord, self._ord)
        if ierr > 0:
            errmsg = f"When setting theta={theta:0.3f} and z={z:0.3f}"\
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
    def z(self):
        return self._z

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
        self._theta = None
        self._z = None
        self._C = None
        self._D = None

    def _set_values(self):
        eta, zeta = self.values

        config = self._hayami_uh.config
        L0 = config.length_ref
        L = config.length
        self._theta = c_pygme_models_hydromodels.hayami_compute_theta(L0, L,
                                                                      eta,
                                                                      zeta)
        self._z = c_pygme_models_hydromodels.hayami_compute_z(L0, L,
                                                              eta,
                                                              zeta)
        self._C = c_pygme_models_hydromodels.hayami_compute_C(L0, L,
                                                              eta,
                                                              zeta)
        self._D = c_pygme_models_hydromodels.hayami_compute_D(L0, L,
                                                              eta,
                                                              zeta)
        self._hayami_uh.set_uh(self._theta, self._z)
        super()._set_values()

    @property
    def nuh(self):
        return 0

    @property
    def theta(self):
        return self._theta

    @property
    def z(self):
        return self._z

    @property
    def C(self):
        return self._C

    @property
    def D(self):
        return self._D

    @property
    def uhs(self):
        return [(None, self._hayami_uh)]


class Hayami(Model):

    def __init__(self):

        # Config vector
        # default timestep in sec = daily (=86400 sec)
        # default reach length in m = 10km
        config = Vector(["timestep", "length_ref", "length",
                         "lateral"],
                        defaults=[86400, 1e4, 1e4, 0],
                        mins=[1, 1, 1, 0],
                        maxs=[np.inf, np.inf, np.inf, 1])

        # params vector
        # eta is measured in days and corresponds to length_ref
        # zeta is dimless and corresponds to length_ref
        vect = Vector(["eta", "zeta"],
                      defaults=[1., 1.],
                      mins=[1e-4, 1e-4],
                      maxs=[1e5, 100.])
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
    def theta(self):
        return self.params.theta

    @property
    def z(self):
        return self.params.z

    @property
    def C(self):
        return self.params.C

    @property
    def D(self):
        return self.params.D

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

        cp = Vector(["eta", "zeta"],
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
        eta = np.linspace(0.1, 10, nn)
        zeta = np.linspace(0.1, 10, nn)
        ee, zz = np.meshgrid(eta, zeta)
        self.paramslib = np.column_stack([ee.ravel(), zz.ravel()])
