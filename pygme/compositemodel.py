import math
import numpy as np

from pygme.model import Model
from pygme.data import Matrix

# Maximum outputs from a composite models (can be updated when loading package)
NOUTPUTSMAX = 100


class Node(object):

    def __init__(self, id, model):
        self._id = id
        self._var_mapping = {}
        self._model = model
        self._run_order = None
        self.children = {}

    @property
    def id(self):
        return self._id


    @property
    def model(self):
        return self._model


    @property
    def run_order(self):
        return self._run_order


    def add_child(self, child, run_after_parent=False):
        self._children[child] = run_after_parent


    def add_mapping(self,
            idx_node, idx_composite, type='inputs'):
        self._var_mapping[type].append({'node':idx_node,
                    'composite': idx_composite)


class Network(object):

    def __init__(self):
        self._nodes = {}
        self._max_runorder = 0


    def get_node(self, id):
        return self._nodes[id]


    def add_node(self, node):
        self._nodes[node['id']] = node


    @property
    def max_runorder(self):
        return self._max_runorder


    def compute_run_order(self):
        ''' Compute component run order '''

        # Look for head nodes (nodes with no parent)
        head = set(self._nodes.keys())
        for id in self._nodes:

            # Look for nodes that point to id
            for id2 in self._nodes:
                if id == id2:
                    continue

                # Remove the id from head nodes
                nd = self._nodes[id2]
                if id in nd['children'] and id in parents:
                    head.remove(id)

        # Populate network by recurrence and check circularity
        def update(parents, done, order):
            # Find nodes
            children = []
            for id in parents:
                link = self._nodes[id]

                # Increase run order if relevant
                if link['run_after_parent']:
                    link['run_order'] = order

                    if id in done:
                        raise ValueError('Circularity detected in network')

                done.append(id)

                # Add the children nodes
                children.extend(link['children'].keys())

            # Remove duplicates
            children = set(children)

            if len(children)>0:
                update(children, done, order+1)
                self._max_runorder += 1

        update(parents, [], 0)




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

        self._nodes = {}
        self._max_runorder = 0


    def post_params_setter(self):
        raise ValueError('No params mapping. Please override')



    def allocate(self, inputs, noutputs=1):

        super(CompositeModel, self).allocate(inputs, noutputs)

        # Affect inputs to components
        for id, link in self._nodes.iteritems():
            model = link['model']

            # Check inputs size
            nval1, nvar, _, _ = self.get_dims('inputs')
            nval2, _, _, _ = model.get_dims('inputs')
            if nval1 != nval2:
                raise ValueError(('With {0} model,'+
                    ' composite model inputs nval ({1}) different' +
                    ' from component {2} nval ({3})').format(
                        self.name, nval1, link['id'], nval2))

            # Loop through inputs indexes
            idx = link['composite_inputs_index']

            if idx is None:
                continue

            for i1, i2 in idx:
                if i2 >= nvar:
                    raise ValueError(('With {0} model,'+
                        ' component input index ({1}) is greater or equal' +
                        ' to number of inputs in composite model ({2})').format(
                            self.name, i2, nvar))

                model.inputs[:, i2] = self.inputs[:, i1]


    def run(self, seed=None):

        start, end = self.startend
        kk = np.arange(start, end + 1)


        for order in range(self._max_runorder+1):

            for id, link in self._nodes.iteritems():

                # Run models with appropriate run order
                if link['run_order'] == order:
                    model = link['model']
                    model.run(seed)

                    oidx = link['composite_outputs_index']
                    if not oidx is None:
                        for i1, i2 in oidx:
                            self.outputs[kk, i2] = model.outputs[kk, i1]

