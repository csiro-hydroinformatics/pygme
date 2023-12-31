import math, itertools
import copy
import random
import pandas as pd
import numpy as np

from hydrodiy.data.containers import Vector

from pygme import has_c_module
if has_c_module("models_utils"):
    import c_pygme_models_utils
    NORDMAXMAX = c_pygme_models_utils.uh_getnuhmaxlength()
else:
    NORDMAXMAX = 300

UHNAMES = ["gr4j_ss1_daily", "gr4j_ss2_daily", \
            "gr4j_ss1_hourly", "gr4j_ss2_hourly", \
            "lag", "triangle", "flat"]

class ParameterCheckValueError(Exception):
    """ Error raised by a checkvalues function in ParamsVector """
    pass


class UH(object):

    def __init__(self, name, nordmax=NORDMAXMAX):
        """ Object handling unit hydrograph. The object does not run the
        convolution, just stores the unit hydrograph ordinates

        Parameters
        -----------
        name : str
            Name of the UH
        nordmax : int
            Maximum number of ordinates
        """
        # Set max length of uh
        if nordmax > NORDMAXMAX or nordmax<=0:
            raise ValueError(("Expected nuhmax in [1, {0}], "+\
                    "got {1}").format(NORDMAXMAX, nordmax))

        self._nordmax = nordmax

        # Check name
        self._uhid = 0
        for uid in range(len(UHNAMES)):
            if name == UHNAMES[uid]:
                self._uhid = uid+1
                break

        if self._uhid == 0:
            expected = "/".join(UHNAMES)
            raise ValueError("Expected UH name in {0}, got {1}".format(\
                expected, name))
        self.name = name

        # Initialise ordinates and states
        self._ord = np.zeros(nordmax, dtype=np.float64)
        self._ord[0] = 1.
        self._states = np.zeros(nordmax, dtype=np.float64)

        # set time base param value
        self._timebase = 0.

        # Number of ordinates
        # nuh is stored as an array to be passed to the C routine
        self._nord = np.array([1], dtype=np.int32)


    def __str__(self):
        str = "UH {0}: timebase={1} nord={2}".format(self.name, \
                self.timebase, self.nord)
        return str


    @property
    def uhid(self):
        return self._uhid


    @property
    def nordmax(self):
        return self._nordmax


    @property
    def nord(self):
        # There is a trick here, we return an
        # integer but the internal state is an array
        return self._nord[0]


    @property
    def timebase(self):
        return self._timebase

    @timebase.setter
    def timebase(self, value):
        has_c_module("models_utils")

        # Check value
        value = np.float64(value)

        # Populate the uh ordinates
        ierr = c_pygme_models_utils.uh_getuh(self.nordmax, self.uhid, \
                                        value, self._nord, self._ord)
        if ierr>0:
            raise ValueError(("When setting param to {0} for UH {1}, "+\
                "c_pygme_models_utils.uh_getuh returns {2}").format(\
                        value, self.name, ierr))

        # Store parameter value
        self._timebase = value

        # Reset uh states to a vector of zeros
        # with length nord
        self._states[:self.nord] = 0

        # Set remaining ordinates to 0
        self._ord[self.nord:] = 0


    @property
    def ord(self):
        return self._ord


    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, values):
        values = np.atleast_1d(values).astype(np.float64)

        nord = self.nord
        if values.shape[0]<nord:
            raise ValueError("Expected state vector to be of length"+\
                ">={0}, got {1}".format(nord, values.shape[0]))

        self.reset()
        self._states[:nord] = values[:nord]


    def reset(self):
        """ Set all UH states to zeros """
        self._states = np.zeros(self.nordmax)


    def clone(self):
        """ Generates a clone of the current UH """
        clone = UH(self.name, self.nordmax)
        clone.timebase = self.timebase
        clone._states = self.states.copy()

        return clone



# Overload Vector class to change UH and corresponding states
# when changing model parameters
class ParamsVector(Vector):

    def __init__(self, params, checkvalues=None):
        """ Object handling parameter vector. The object stores the unit
        hydrographs and the functions used to set the uh time base.

        The uhs are stoed in a list of tuples that can be accessed via the
        self.uhs. Each tuple contains two elements:
            (1) a function to setup the time base from parameter values
                the function must have a signature like set_timebase(params)
                and must return a float.
            (2) UH object (see pygme.model.UH)

        Parameters
        -----------
        params : hydrodiy.data.containers.Vector
            Vector of parameters including names, default values, min
            and max.
        checkvalues : function
            Function assessing if the parameter combination is valid. The
            function must have the following signature:

            def checkvalues(values)

            It must return a ParameterCheckValueError if the values are
            considered invalid

        """
        # check_hitbounds is turned on
        super(ParamsVector, self).__init__(params.names, \
                    params.defaults, params.mins, params.maxs, \
                    True)

        self._uhs = None

        # Check if the suitable function works for the default
        # parameter values
        self._checkvalues = None
        if not checkvalues is None:
            # Run the function for default values
            # this should not fail
            checkvalues(params.defaults)

            # Run the function for values lower than the minimum
            # this should fail and return a ValueError
            try:
                checkvalues(params.mins-1)
            except ParameterCheckValueError:
                pass
            else:
                raise ValueError("checkvalues function does not "+\
                    "generate a ParameterCheckValuesError when run "+\
                    "with params.mins-1")

            self._checkvalues = checkvalues

    def _set_values(self):
        # Check if the values suitable
        if not self._checkvalues is None:
            self._checkvalues(self.values)

        # Set UH parameter if needed
        if self.nuh>0:
            for iuh, (set_timebase, uh) in enumerate(self.uhs):
                uh.timebase = set_timebase(self)


    def __setattr__(self, name, value):
        # Set attribute for vector object
        super(ParamsVector, self).__setattr__(name, value)

        # Set UH parameter if needed
        if not hasattr(self, "names"):
            return

        if name in self.names:
            self._set_values()


    @property
    def checkvalues(self):
        return self._checkvalues


    @property
    def nuh(self):
        if self.uhs is None:
            return 0
        else:
            return len(self.uhs)


    @property
    def uhs(self):
        return self._uhs


    @property
    def iuhparams(self):
        return self._iuhparams


    @Vector.values.setter
    def values(self, val):
        # Run the vector value setter
        Vector.values.fset(self, val)

        # Set additional elements
        self._set_values()


    def add_uh(self, uh_name, set_timebase, nuhmax=NORDMAXMAX):
        """ Add uh using the uh name and a set_time base function

        Parameters
        -----------
        uh_name : str
            Name of a recognised uh.
        set_timebase: function
            Function with signature set_timebase(params)
            The function takes the parameter object as argument
            and returns a float corresponding to the UH
            time base
        nuhmax : int
            Maximum number of ordinates
        """

        if self._uhs is None:
            self._uhs = []

        test = set_timebase(self)
        if not isinstance(test, float):
            raise ValueError(("Expected set_timebase function to "+\
                "return a float, got {0}").format(test))

        # Create UH
        uh = UH(uh_name, nuhmax)

        # Set timebase to check it does not trigger any error
        uh.timebase = test

        # All test ok, appending uh to list of uhs
        self._uhs.append((set_timebase, uh))


    def clone(self):
        params = Vector(self.names, self.defaults, self.mins, \
                    self.maxs, self.check_hitbounds)

        clone = ParamsVector(params)
        clone.values = self.values.copy()
        if not self.uhs is None:
            clone._uhs = [(uht[0], uht[1].clone()) for uht in self.uhs]

        return clone


class Model(object):

    def __init__(self, name, config, params, states, \
            ninputs, noutputsmax):

        # Model name
        self.name = name

        # Config and params vectors
        self._config = config
        self._params = params
        self._states = states

        # Dimensions
        self._ninputs = int(ninputs)
        self._noutputsmax = int(noutputsmax)
        self._noutputs = 0 # will be set to >0 when outputs are allocated

        # data
        self._inputs = None
        self._inputs_names = [f"input{i+1:02d}" for i in range(self._ninputs)]
        self._outputs = None
        self._outputs_names = [f"output{i+1:02d}" for i in range(self._noutputsmax)]

        # Start/end index
        self._istart = None
        self._iend = None


    def __getattribute__(self, name):
        # Except certain names to avoid infinite recursion
        if name in ["name", "_config", "_params", "_states", "_ninputs", \
            "_noutputsmax", "_noutputs", "_inputs", "_outputs", "_istart", \
            "_iend", "_outputs_names"]:
            return super(Model, self).__getattribute__(name)

        if name in self._params.names:
            return getattr(self._params, name)

        if name in self._config.names:
            return getattr(self._config, name)

        if name in self._states.names:
            return getattr(self._states, name)

        return super(Model, self).__getattribute__(name)


    def __setattr__(self, name, value):
        # Except certain names to avoid infinite recursion
        if name in ["name", "_config", "_params", "_states", "_ninputs", \
            "_noutputsmax", "_noutputs", "_inputs", "_outputs", "_istart", \
            "_iend", "_outputs_names"]:
            super(Model, self).__setattr__(name, value)
            return

        if name in self._params.names:
            return setattr(self._params, name, value)

        elif name in self._config.names:
            return setattr(self._config, name, value)

        elif name in self._states.names:
            return setattr(self._states, name, value)

        else:
            super(Model, self).__setattr__(name, value)


    def __getitem__(self, name):
        if name in self._params.names or \
                name in self._config.names or \
                name in self._states.names:
            return getattr(self, name)
        else:
            raise ValueError(f"model {self.name}: '{name}' cannot be obtained from"+\
                        " the model attributes")


    def __setitem__(self, name, value):
        if name in self._params.names or \
                name in self._config.names or \
                name in self._states.names:
            return setattr(self, name, value)
        else:
            raise ValueError(f"model {self.name}: '{name}' cannot be set as a"+\
                        " model attributes")


    def __str__(self):
        str = ("\n{0} model implementation\n"+\
                "\tConfig: {1}\n\tParams: {2}\n\tStates: {3}"+\
                "\n\tNUH: {4}").format( \
                    self.name, self.config, self.params, \
                    self.states, self.params.nuh)
        return str


    @property
    def params(self):
        """ Get the parameter object """
        return self._params


    @property
    def config(self):
        """ Get the config vector """
        return self._config


    @property
    def states(self):
        """ Get the state vector """
        return self._states


    @property
    def ntimesteps(self):
        """ Get number of simulation timestep """
        if self._inputs is None:
            raise ValueError("Trying to get ntimesteps, but inputs "+\
                        "are not allocated. Please allocate")

        return self.inputs.shape[0]


    @property
    def istart(self):
        """ Get index of simulation start """
        if self._inputs is None:
            raise ValueError("Trying to get istart, "+\
                "but inputs are not allocated. Please allocate")

        if self._istart is None:
            raise ValueError("Trying to get istart, "+\
                "but it is not set. Please set value")

        return self._istart


    @istart.setter
    def istart(self, value):
        """ Set index of simulation start """
        value = np.int32(value)

        if self._inputs is None:
            raise ValueError("Trying to set istart, "+\
                "but inputs are not allocated. Please allocate")

        if value<0 or value>self.ntimesteps-1:
            raise ValueError("Expected istart in [0, {0}], got {2}".format(\
                self.ntimestep-1, value))

        self._istart = value


    @property
    def iend(self):
        """ Get index of simulation end """
        if self._inputs is None:
            raise ValueError("Trying to get iend, "+\
                "but inputs are not allocated. Please allocate")

        if self._iend is None:
            raise ValueError("Trying to get iend, "+\
                "but it is not set. Please set value")

        return self._iend


    @iend.setter
    def iend(self, value):
        """ Set index of simulation end """
        value = np.int32(value)

        if self._inputs is None:
            raise ValueError(("model {0}: Trying to set iend, "+\
                "but inputs are not allocated. Please allocate").format(self.name))

        # Syntactic sugar to get a simulation running for the whole period
        if value == -1:
            value = self.ntimesteps-1

        if value<0 or value>self.ntimesteps-1:
            raise ValueError("model {0}: Expected iend in [0, {1}], got {2}".format(\
                self.name, self.ntimesteps-1, value))

        self._iend = value


    @property
    def ninputs(self):
        """ Get number of model input variables """
        return self._ninputs


    @property
    def inputs_names(self):
        """ Get model inputs names """
        return self._inputs_names

    @inputs_names.setter
    def inputs_names(self, values):
        """ Set outputs names """
        if len(values) != self.ninputs:
            raise ValueError(f"model {self.name}: Trying to set inputs names, "+\
                f"a vector of length {self.ninputs} is expected, got {len(values)}")

        self._inputs_names = [str(nm) for nm in values]


    @property
    def inputs(self):
        """ Get model input array """
        if self._inputs is None:
            raise ValueError("Trying to access inputs, "+\
                "but they are not allocated. Please allocate")

        return self._inputs


    @inputs.setter
    def inputs(self, values):
        """ Set model input array """
        inputs = np.ascontiguousarray(np.atleast_2d(values).astype(np.float64))
        if inputs.shape[1] != self.ninputs:
            raise ValueError("model {0}: Expected {1} inputs, got {2}".format(\
                self.name, self.ninputs, values.shape[1]))

        self._inputs = inputs


    @property
    def noutputsmax(self):
        """ Get maximum number of model output variables """
        return self._noutputsmax


    @property
    def outputs_names(self):
        """ Get model outputs names """
        return self._outputs_names

    @outputs_names.setter
    def outputs_names(self, values):
        """ Set outputs names """
        if len(values) != self.noutputsmax:
            raise ValueError(f"model {self.name}: Trying to set outputs names, "+\
                f"a vector of length {self.noutputsmax} is expected, got {len(values)}")

        self._outputs_names = [str(nm) for nm in values]


    @property
    def noutputs(self):
        """ Get number of output variables """
        return self._noutputs


    @property
    def outputs(self):
        """ Get model output array """
        if self._outputs is None:
            raise ValueError(("model {0}: Trying to access outputs, "+\
                "but they are not allocated. Please allocate").format(\
                    self.name))

        return self._outputs


    @outputs.setter
    def outputs(self, values):
        """ Set model output array """
        outputs = np.ascontiguousarray(np.atleast_2d(values).astype(np.float64))
        noutputs = max(1, self.noutputs)

        if outputs.shape[1] != noutputs:
            raise ValueError("model {0}: Expected {1} outputs, got {2}".format(\
                self.name, noutputs, values.shape[1]))

        self._outputs = outputs


    def to_dataframe(self, index=None, include_inputs=False):
        """ Get model output in pandas dataframe format """
        cols = self.outputs_names[:self.noutputs]
        dfo = pd.DataFrame(self.outputs, columns=cols, index=index)

        if include_inputs:
            cols = self.inputs_names
            dfi = pd.DataFrame(self.inputs, columns=cols, index=index)
            return pd.concat([dfi, dfo], axis=1)
        else:
            return dfo


    def allocate(self, inputs, noutputs=1):
        """ Allocate inputs and outputs arrays.
        We define the number of outputs here to allow more
        flexible memory allocation """

        if noutputs <= 0 or noutputs > self.noutputsmax:
            raise ValueError(("model {0}: " +\
                "Expected noutputs in [1, {1}], got {2}").format(\
                    self.name, self.noutputsmax, noutputs))

        # Allocate inputs
        self.inputs = inputs

        # Allocate outputs
        self._noutputs = noutputs
        self.outputs = np.zeros((inputs.shape[0], noutputs))

        # Set istart/iend to default
        self.istart = 0
        self.iend = -1


    def initialise(self, states=None, uhs=None):
        """ Initialise state vector and potentially all UH states vectors """
        if states is None:
            self.states.reset()
        else:
            self.states.values = states

        if not uhs is None:
            # Set uhs states values to argument
            nuh = self.params.nuh
            if len(uhs) != nuh:
                raise ValueError(("Expected a list of {0} unit"+\
                    " hydrographs object for"+\
                    " initialisation, got {1}").format(\
                    nuh, len(uhs)))

            for iuh in range(nuh):
                # We extract the UH object
                # the set_timebase function is not needed here
                _, uh1 = self.params.uhs[iuh]

                # Compare with the uh supplied to initialise
                _, uh2 = uhs[iuh]

                if uh1.nord != uh2.nord:
                    raise ValueError(("Expected UH[{0}] nord to be {1}."+\
                            " Got {2}.").format(iuh, uh1.nord, uh2.nord))

                if abs(uh1.timebase-uh2.timebase)>1e-8:
                    raise ValueError(("Expected UH[{0}] timebase to be {1}."+\
                            " Got {2}.").format(iuh, uh1.timebase, \
                            uh2.timebase))

                uh1.reset()
                uh1.states[:uh1.nord] = uh2.states[:uh2.nord].copy()

        else:
            # Reset uhs states values
            nuh = self.params.nuh
            for iuh in range(nuh):
                # We extract the UH object
                # the set_timebase function is not needed here
                _, uh1 = self.params.uhs[iuh]
                uh1.reset()


    def initialise_fromdata(self):
        """ Initialise model from external data
            (e.g. steady state from parameter values)

            If not overridden, the function runs the initialise command without
            parameters.
        """
        self.initialise()


    def run(self):
        """ Run the model """
        raise NotImplementedError(("model {0}: "+\
            "Method run not implemented").format(self.name))


    def inisens(self, states0, states1, eps=1e-4, iout=0, ignore_error=False):
        """ Sensitivity on model initialisation

        Parameters
        -----------
        states0 : numpy.ndarray
            First set of states (ideally, empty storages for bucket models)
        states1 : numpy.ndarray
            Second set of states (ideally, full storages for bucket models)
        eps : float
            Tolerance
        iout : int
            Index of model output to use for sensitivity analysis

        Returns
        -----------
        wamrup : int
            Duration of warmup period
        sim0 : numpy.ndarray
            Simulation corresponding to first initialisation
        sim1 : numpy.ndarray
            Simulation corresponding to second initialisation
        """

        # First simulation
        self.states.values = states0
        self.run()
        sim1 = self.outputs[:, iout].copy()

        # Second simulation
        self.states.values = states1
        self.run()
        sim2 = self.outputs[:, iout].copy()

        # Difference
        idiff = np.abs(sim1-sim2) > eps
        nval = len(idiff)
        if np.sum(idiff) == 0:
            warmup = 0
        else:
            warmup = np.max(np.where(idiff)[0])

        if warmup == nval-1 and not ignore_error:
            raise ValueError("Warmup period is longer"+\
                    " than simulation duration")

        return warmup, sim1, sim2


    def clone(self):
        """ Clone the current model instance"""

        model = Model(self.name, \
            self.config.clone(), \
            self.params.clone(), \
            self.states.clone(), \
            self.ninputs, \
            self.noutputsmax)

        # Allocate data
        if not self._inputs is None:
            model.allocate(self.inputs, self.noutputs)
            model.istart = self.istart
            model.iend = self.iend

        return model


