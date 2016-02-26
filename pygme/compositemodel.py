import math
import numpy as np

from pygme.model import Model
from pygme.data import Matrix

# Maximum outputs from a composite models (can be updated when loading package)
NOUTPUTSMAX = 100


class CompositeModel(Model):


    def __init__(self, name,
            ninputs, nparams, nstates,
            nens_params=1,
            nens_states=1,
            nens_outputs=1):

        Model.__init__(self,
            name=name,
            nconfig=0,
            ninputs=ninputs,
            nparams=nparams,
            nstates=nstates,
            noutputs_max = NOUTPUTSMAX,
            nens_params=nens_params,
            nens_states=nens_states,
            nens_outputs=nens_outputs)

        self._network = {}


    def add_link(self, id,
        model=None,
        composite_inputs_index=None,
        child_id=None,
        child_input_index=0,
        parent_output_index=0,
        run_after_parent=True):

        if id in self._network:
            link = self._network[id]

            # Check model is defined twice
            if not link['model'] is None and not model is None:
                raise ValueError('model cannot be defined twice' +
                    ' for an existing link, set model=None')

            # Check run_after_parent is defined twice
            if not links['run_after_parent'] == run_after_parent:
                raise ValueError('run_after_parent cannot be changed' +
                    ' for an existing link')

        else:
            link = {
                'model': model,
                'run_after_parent': run_after_parent,
                'run_order' : None,
                'composite_inputs_index' : composite_inputs_index,
                'children' : []
                'inputs_mapping': []
            }
            self._network[id] = link

        # Add connections
        if not child_id is None:
            link['children'].append((child_id, parent_output_index,
                                        child_output_index))



    def post_params_setter(self):
        raise ValueError('No params mapping. Please override')



    def allocate(self, inputs, noutputs=1):

        super(CompositeModel).allocate(inputs, noutputs)

        # Affect inputs to components
        for id, link in self._network.iteritems():
            pass


    def run(self, seed=None):

        pass


    def compute_run_order(self):
        ''' Compute component run order '''

        # Look for link with no parent
        top_nodes = []
        for id in self._network:
            link = self._network[id]
            if len(link['parents']) == 0:
                top_nodes.append(id)

        # Populate network by recurrence and check circularity
        def update(parents, done):
            # Find nodes
            children = []
            for id in parents:
                link = self._network[id]

                # Increase run order if relevant
                if link['run_after_parent']:
                    link['run_order'] += 1

                    if id in done:
                        raise ValueError('Circularity detected in network')

                done.append(id)

                # Add the children nodes
                children_list.append([k[0] for k in link['children']])

            if len(children)>0:
                update(children, done)

        update(top_nodes)



