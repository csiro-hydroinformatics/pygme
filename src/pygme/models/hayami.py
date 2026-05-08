import math
import numpy as np

from hydrodiy.data.containers import Vector
from pygme.model import Model, UH, ParamsVector, NORDMAXMAX
from pygme.calibration import Calibration, CalibParamsVector, ObjFunBCSSE

from pygme import has_c_module
if has_c_module("models_hydromodels"):
    import c_pygme_models_hydromodels


# Transformation functions for hayami parameters
def hayami_trans2true(x):
    return x


def hayami_true2trans(x):
    return x


class HayamiUH(UH):
    def __init__(self, config, nordmax=NORDMAXMAX):
        super(HayamiUH, self).__init__("lag")
        self._theta = config.timestep
        self._z = 2.5
        self._config = config

    @property
    def config(self):
        return self._config

    def set_uh(self, theta, z):
        theta = self.theta
        z = self.z

        # Populate the uh ordinates
        ierr = c_pygme_models_hydromodels.uh_getuh_hayami(self.nordmax,
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

    def _set_values(self):
        eta, z = self.values
        theta = eta * self.config.timestep
        self._hayami_uh.set_uh(theta, z)
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
        # default timestep = daily (=86400 sec)
        # default reach length = 10km
        config = Vector(["timestep", "length", "lateral"],
                        defaults=[86400, 1e4, 0],
                        mins=[1, 1, 0],
                        maxs=[np.inf, np.inf, 1])

        # params vector
        vect = Vector(["eta", "z"],
                      defaults=[1., 1.],
                      mins=[1e-4, 1e-4],
                      maxs=[2400, 100.])
        params = HayamiParamsVector(vect, config)

        # State vector
        states = Vector(["S"])

        # Model
        super(Hayami, self).__init__("Hayami",
                                     config, params, states,
                                     ninputs=1,
                                     noutputsmax=2)

        self.inputs_names = ["Inflow"]
        self.outputs_names = ["Q", "VR"]

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

        cp = Vector(["tU", "talpha"],
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
        nn = int(math.sqrt(nparamslib))
        uu, aa = np.meshgrid(np.linspace(0.1, 3, nn),
                             np.linspace(0, 1, nn))
        plib = np.column_stack([uu.ravel(), aa.ravel()])

        self.paramslib = plib
