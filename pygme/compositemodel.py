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
        child_id=None,
        child_input_index=0,
        parent_output_index=0,
        run_after_parent=True):

        if id in self._network:
            link = self._network[id]
        else:
            link = {
                'model': model,
                'run_after_parent': run_after_parent,
                'run_order' : None,
                'children' : []
            }
            self._network[id] = link

        # Check model is defined twice
        if link['model'] is None and not model is None:
            link['model'] = model
        elif not link['model'] is None and model is None:
            pass
        else:
            raise ValueError('model cannot be defined twice for an existing link, ' +
                    'set model=None')

        # Check run_after_parent is defined twice
        if not links['run_after_parent'] == run_after_parent:
            raise ValueError('run_after_parent cannot be changed for an existing link')

        # Add connections
        if not child_id is None:
            link['children'].append((child_id, parent_output_index, child_output_index))


    def params_mapping(self):
        raise ValueError('No params mapping. Please override')


    def compute_run_order(self):
        ''' Compute component run order '''

        # Look for link with no parent
        top_nodes = []
        for id in self._network:
            link = self._network[id]
            if len(link['parents']) == 0:
                top_nodes.append(id)

        # Populate network by recurrence and check circularity
        def update(parents):
            # Find nodes
            children = []
            for id in parents:
                link = self._network[id]

                # Increase run order if relevant
                if link['run_after_parent']:
                    link['run_order'] += 1

                # Add the children nodes
                children_list.append([k[0] for k in link['children']])

            if len(children)>0:
                update(children)

        update(top_nodes)



