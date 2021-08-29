import math
import numpy as np

import matplotlib.pyplot as plt

from pygme.model import Model
from pygme.data import Matrix, Vector

# Maximum outputs from a composite models (can be updated when loading package)
NOUTPUTSMAX = 100


class CompositeNode(object):

    def __init__(self, model=None, id=None,
            x=np.random.uniform(),
            y=np.random.uniform()):
        self._id = id
        self._var_mapping = {"inputs":[], "outputs":[]}
        self._model = model
        self._runorder = 0
        self.children = {}
        self.x = x
        self.y = y


    def __str__(self):
        str = ("Node {0} : model {1}, run order {2}, xy " +
                "({3}, {4})").format(
                self.id, self.model.name, self.runorder,
                    self.x, self.y)
        return str


    @classmethod
    def from_dict(cls, data, model):
        """ Create CompositeNode object from dictionary """

        nd = CompositeNode(model=model,
            id=data["id"], x=data["id"], y=data["id"])

        for id, ch in data["children"].iteritems():
            nd.add_child(id, ch["idx_outputs_parent"],
                    ch["idx_inputs_child"])

        for type in ["inputs", "outputs"]:
            for m in data["var_mapping"][type]:
                nd.add_mapping(m["inode"], m["icomposite"], type)

        return nd


    @property
    def id(self):
        return self._id


    @property
    def model(self):
        return self._model


    @property
    def runorder(self):
        return self._runorder


    def add_child(self, child,
            idx_outputs_parent=0, idx_inputs_child=0):

        self.children[child] = {
                "idx_outputs_parent": idx_outputs_parent,
                "idx_inputs_child": idx_inputs_child
            }


    def add_mapping(self, inode, icomposite, type="inputs"):

        _, nvar, _, _ = self.model.get_dims(type)
        if inode >= nvar:
            raise ValueError(("With {0} node, {1} mapping " +
                "has index ({2}) greater or equal than " +
                "number of variables ({3}) in model {4} {1}").format(self.id,
                    type, nvar, inode, nvar, self.model.name))

        self._var_mapping[type].append({"inode":inode,
                    "icomposite": icomposite})


    def to_dict(self):
        data = {
            "id" : self._id,
            "var_mapping": self._var_mapping,
            "runorder" : self._runorder,
            "children" : self.children,
            "x" : self.x,
            "y" : self.y
        }
        return data



class CompositeNetwork(object):

    def __init__(self):
        self._nodes = {}
        self._max_runorder = 0


    def __getitem__(self, key):
        return self._nodes[key]


    def __setitem__(self, key, value):
        if key in self._nodes:
            raise ValueError(("Node {0} already" +
                " exists in network").format(key))

        value._id = key
        self._nodes[key] = value

    def __str__(self):
        str = "Network:\n"
        for id, nd in self._nodes.iteritems():
            str += "\t{0}\n".format(nd)

        return str

    @property
    def max_runorder(self):
        return self._max_runorder


    def compute_runorder(self):
        """ Compute component run order """

        # Look for head nodes (nodes with no parent)
        # And perform sanity check on network
        ids = self._nodes.keys()
        heads = set(ids)
        for id, nd in self._nodes.iteritems():

            _, nvaro, _, _ = nd.model.get_dims("outputs")

            # Check that all children are in network
            # and have valid inputs/outputs indexes
            for ch in nd.children:
                if not ch in ids:
                    raise ValueError(("Node {0} is a child of {1}, " +
                            "but is not a network node").format(ch,
                                id))

                # Check output index
                idxo = nd.children[ch]["idx_outputs_parent"]
                if idxo < 0 or idxo >= nvaro:
                    raise ValueError(("With the connection between node {0} and {1}, " +
                            "output index ({2}) is greater than "+
                            "the number of variables in model" +
                            " outputs data ({3})").format(id, ch, idxo, nvaro))

                idxi = nd.children[ch]["idx_inputs_child"]
                _, nvari, _, _ = self._nodes[ch].model.get_dims("inputs")
                if idxi < 0 or idxi >= nvari:
                    raise ValueError(("With the connection between node {0} and {1}, " +
                            "input index ({2}) is greater than "+
                            "the number of variables in model" +
                            " inputs data ({3})").format(id, ch, idxi, nvari))

            # Look for nodes that point to id
            for id2, nd2 in self._nodes.iteritems():
                if id == id2:
                    continue

                # Remove the id from head nodes
                if id in nd2.children and id in heads:
                    heads.remove(id)

        # Populate network by recurrence and check circularity
        def update(parents, done, order):
            # Find nodes
            children = []
            for id in parents:
                nd = self._nodes[id]

                # Increase run order
                nd._runorder = order

                if id in done:
                    raise ValueError("Circularity detected in network")

                done.append(id)

                # Add the children nodes
                children.extend(nd.children.keys())

            # Remove duplicates
            children = set(children)

            if len(children)>0:
                update(children, done, order+1)
                self._max_runorder += 1

        update(heads, [], 0)


    def draw(self, ax, arrow_factor=0.9,
            ptopts={}, arrowopts={}):

        xx = []
        yy = []

        for id, nd in self._nodes.iteritems():
            # Draw node
            ax.plot(nd.x, nd.y, "o", **ptopts)
            xx.append(nd.x)
            yy.append(nd.y)

            # Draw arrow
            for ch in nd.children:
                ndc = self._nodes[ch]
                ax.arrow(nd.x, nd.y,
                        (ndc.x-nd.x)*arrow_factor,
                        (ndc.y-nd.y)*arrow_factor,
                        **arrowopts)
                xx.append(ndc.x)
                yy.append(ndc.y)

        # Draw
        xx = np.array(xx)
        xlim = [np.min(xx), np.max(xx)]
        yy = np.array(yy)
        ylim = [np.min(yy), np.max(yy)]
        ax.plot(xlim, ylim, color="none")



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
        raise ValueError("No params mapping. Please override")


    def allocate(self, inputs, noutputs=1):

        super(CompositeModel, self).allocate(inputs, noutputs)

        # Affect inputs to components
        for id, node in self._network._nodes.iteritems():
            model = node.model

            # Check inputs size
            nval1, nvar, _, _ = self.get_dims("inputs")
            nval2, _, _, _ = model.get_dims("inputs")
            if nval1 != nval2:
                raise ValueError(("With {0} model,"+
                    " composite model inputs nval ({1}) different" +
                    " from component {2} nval ({3})").format(
                        self.name, nval1, link["id"], nval2))

            # Loop through inputs indexes
            mapping = node._var_mapping["inputs"]

            for m in mapping:
                inode = m["node"]
                icomposite = m["composite"]

                if icomposite >= nvar:
                    raise ValueError(("With {0} model,"+
                        " component input index ({1}) is greater or equal" +
                        " to number of inputs in composite model ({2})").format(
                            self.name, icomposite, nvar))

                model.inputs[:, inode] = self.inputs[:, icomposite]


    def run(self, seed=None):

        start, end = self.get_ipos_startend()
        kk = np.arange(start, end + 1)

        _, nvar, _, _ = self.get_dims("outputs")

        for order in range(self._network.max_runorder+1):

            for id, node in self._network._nodes.iteritems():

                # Run models with appropriate run order
                if node.runorder == order:
                    model = node.model
                    model.run(seed)

                    # Loop through outputs mapping
                    mapping = node._var_mapping["outputs"]

                    for m in mapping:
                        inode = m["inode"]
                        icomposite = m["icomposite"]

                        if icomposite >= nvar:
                            raise ValueError(("With {0} model,"+
                                " component output index ({1}) is greater or equal" +
                                " to number of outputs in composite model ({2})").format(
                                    self.name, icomposite, nvar))

                        self.outputs[:, icomposite] = model.outputs[:, inode]


