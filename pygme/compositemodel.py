import math
import numpy as np

from pygme.model import Model
from pygme.data import Matrix

# Maximum outputs from a composite models (can be updated when loading package)
NOUTPUTSMAX = 100


class CompositeNode(object):

    def __init__(self, id, model, metadata=None):
        self._id = id
        self._var_mapping = {}
        self._model = model
        self._run_order = None
        self.children = {}
        self.metadata = metadata


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


    def add_mapping(self, idx_node, idx_composite, type='inputs'):

        _, nvar, _, _ = self.model.get_dims(type)
        if idx_node >= nvar:
            raise ValueError(('With {0} node, {1} mapping ' +
                'has index ({2}) greater or equal than ' +
                'number of variables ({3}) in model {4} {1}').format(self.id,
                    type, nvar, idx_node, nvar, self.model.name))

        self._var_mapping[type].append({'node':idx_node,
                    'composite': idx_composite})



class CompositeNetwork(object):

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
        for id, nd in self._nodes.iteritems():

            # Look for nodes that point to id
            for id2, nd2 in self._nodes.iteritems():
                if id == id2:
                    continue

                # Remove the id from head nodes
                if id in nd2.children and id in head:
                    head.remove(id)

        # Populate network by recurrence and check circularity
        def update(parents, done, order):
            # Find nodes
            children = []
            for id in parents:
                nd = self._nodes[id]

                # Increase run order if relevant
                if nd.run_after_parent:
                    nd.run_order = order

                    if id in done:
                        raise ValueError('Circularity detected in network')

                done.append(id)

                # Add the children nodes
                children.extend(nd.children.keys())

            # Remove duplicates
            children = set(children)

            if len(children)>0:
                update(children, done, order+1)
                self._max_runorder += 1

        update(parents, [], 0)



class CompositeModel(Model):


    def __init__(self, name,
            ninputs, nparams, nstates,
            network,
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

        self._network = network


    def post_params_setter(self):
        raise ValueError('No params mapping. Please override')


    def allocate(self, inputs, noutputs=1):

        super(CompositeModel, self).allocate(inputs, noutputs)

        # Affect inputs to components
        for id, node in self._network._nodes.iteritems():
            model = node.model

            # Check inputs size
            nval1, nvar, _, _ = self.get_dims('inputs')
            nval2, _, _, _ = model.get_dims('inputs')
            if nval1 != nval2:
                raise ValueError(('With {0} model,'+
                    ' composite model inputs nval ({1}) different' +
                    ' from component {2} nval ({3})').format(
                        self.name, nval1, link['id'], nval2))

            # Loop through inputs indexes
            mapping = node._var_mapping['inputs']

            for m in mapping:
                inode = m['node']
                icomposite = m['composite']

                if icomposite >= nvar:
                    raise ValueError(('With {0} model,'+
                        ' component input index ({1}) is greater or equal' +
                        ' to number of inputs in composite model ({2})').format(
                            self.name, icomposite, nvar))

                model.inputs[:, inode] = self.inputs[:, icomposite]


    def run(self, seed=None):

        start, end = self.startend
        kk = np.arange(start, end + 1)

        _, nvar, _, _ = self.get_dims('outputs')

        for order in range(self._max_runorder+1):

            for node in self._network._nodes.iteritems():

                # Run models with appropriate run order
                if node.run_order == order:
                    model = node.model
                    model.run(seed)

                    # Loop through outputs mapping
                    mapping = node._var_mapping['outputs']

                    for m in mapping:
                        inode = m['node']
                        icomposite = m['composite']

                        if icomposite >= nvar:
                            raise ValueError(('With {0} model,'+
                                ' component output index ({1}) is greater or equal' +
                                ' to number of outputs in composite model ({2})').format(
                                    self.name, icomposite, nvar))

                        self.outputs[:, icomposite] = model.outputs[:, inode]


