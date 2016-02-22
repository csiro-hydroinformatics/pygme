import math
import numpy as np

from pygme.model import Model
from pygme.data import Matrix


class CompositeLink(object):
    def __init__(model, to_id,
        from_id=None,
        from_outputs_index=0,
        to_inputs_index=0,
        is_sync=True):

        self.model = model
        self.to_id = to_id
        self.from_id = from_id

class CompositeNetwork(object):

    def __init__():
        pass

    def add_link(self, model, id, to_id,
        from_id=None,
        from_outputs_index=0,
        to_inputs_index=0,
        is_sync=True):

        pass


    def compute_order(self):
        pass



class CompositeModel(Model):

    def __init__(self, name, network,
            ninputs, nparams, nstates,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        Model.__init__(self,
            name=name,
            nconfig=nconfig,
            ninputs=ninputs,
            nparams=nparams,
            nstates=nstates,
            noutputs_max = model.noutputs_max,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

