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
    def __init__(self, nordmax=NORDMAXMAX):
        super(HayamiUH, self).__init__("lag")
        self._timebase = (1., 1.)

    @property
    def timebase(self):
        return self._timebase

    @timebase.setter
    def timebase(self, value):
        # Check value
        theta = np.float64(value[0])
        z = np.float64(value[0])

        # Populate the uh ordinates
        ierr = c_pygme_models_utils.uh_getuh(self.nordmax, self.uhid,
                                             value, self._nord, self._ord)
        if ierr > 0:
            errmsg = f"When setting param to {value} for UH {self.name}, "\
                     + f"c_pygme_models_utils.uh_getuh returns {ierr}"
            raise ValueError(errmsg)

        # Store parameter value
        self._timebase = (theta, z)

        # Reset uh states to a vector of zeros
        # with length nord
        self._states[:self.nord] = 0

        # Set remaining ordinates to 0
        self._ord[self.nord:] = 0


        pass

    def clone(self):
        """ Generates a clone of the current UH """
        clone = HayamiUH(self.nordmax)
        clone.timebase = self.timebase
        clone._states = self.states.copy()

        return clone


class HayamiParamsVector(ParamsVector):
    def __init__(self, params, checkvalues=None):
        super(HayamiParamsVector, self).__init__(params)
        self._uhs = [(None, hayami_uh)]


class Hayami(Model):

    def __init__(self):

        # Config vector
        config = Vector(["timestep", "length"],
                        defaults=[86400, 1e5],
                        mins=[1, 1],
                        maxs=[np.inf, np.inf])

        # params vector
        vect = Vector(["theta", "z"],
                      defaults=[1., 0.5],
                      mins=[1e-4, 1e-4],
                      maxs=[100 * 86400., 100.])
        params = HayamiParamsVector(vect)

        # Attach config to params to
        # retrieve data during UH computation
        params.config = config

        # State vector
        states = Vector(["S"])

        # Model
        super(Hayami, self).__init__("Hayami",
                                     config, params, states,
                                     ninputs=1,
                                     noutputsmax=4)

        self.inputs_names = ["Inflow"]
        self.outputs_names = ["Q", "VR"]

    def run(self):
        has_c_module("models_hydromodels")

        # Get uh object (not set_timebase function, see ParamsVector class)
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
