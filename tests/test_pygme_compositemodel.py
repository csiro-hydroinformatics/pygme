import os
import re
import math
import unittest

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from calendar import month_abbr as month

from pygme.compositemodel import CompositeModel, CompositeNetwork, CompositeNode
from pygme.models.basics import NodeModel


def get_model(nin, nout, nval=100):
    m = NodeModel(nin, nout)
    inputs = np.random.uniform(0, 1, (nval,nin))
    m.allocate(inputs)
    return m


class CompositeNodeTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CompositeNodeTestCase')

    def test_print(self):
        m = NodeModel(3, 4)
        nd = CompositeNode(m, 'nd1', 0., 1.)
        str = '{0}'.format(nd)


    def test_node1(self):
        m = NodeModel(3, 4)
        nd = CompositeNode(m, 'nd1')

        nd.add_mapping(2, 2)

        try:
            nd.add_mapping(4, 2)
        except ValueError, e:
            pass
        self.assertTrue(str(e).startswith('With nd1'))


class CompositeNetworkTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CompositeNetworkTestCase')
        source_file = os.path.abspath(__file__)
        self.FHERE = os.path.dirname(source_file)


    def test_network1(self):

        # Connections
        # nd1 ->             -> outputs[0]
        #        nd3  -> nd4        -> outputs[1]
        # nd2 ->             -> nd5
        #                           -> outputs[2]
        ntk = CompositeNetwork()

        ntk['nd1'] = CompositeNode(get_model(1, 1), x=0, y=2)
        ntk['nd1'].add_child('nd3')

        ntk['nd2'] = CompositeNode(get_model(1, 1), x=0, y=0)
        ntk['nd2'].add_child('nd3')

        ntk['nd3'] = CompositeNode(get_model(2, 1), x=1, y=1)
        ntk['nd3'].add_child('nd4')

        ntk['nd4'] = CompositeNode(get_model(1 , 2), x=2, y=1)
        ntk['nd4'].add_child('nd5')

        try:
            ntk.compute_runorder()
        except ValueError, e:
            str(e)
        self.assertTrue(str(e).startswith('Node nd5 is a child'))

        ntk['nd5'] = CompositeNode(get_model(1, 2), x=3, y=0.5)

        # Check printing
        ntks = str(ntk)

        # Check compute run order
        ntk.compute_runorder()

        ck = ntk['nd1'].runorder == 0
        ck = ck & (ntk['nd2'].runorder == 0)
        ck = ck & (ntk['nd3'].runorder == 1)
        ck = ck & (ntk['nd4'].runorder == 2)
        ck = ck & (ntk['nd5'].runorder == 3)

        self.assertTrue(ck)

        fig, ax = plt.subplots()
        ax.axis('off')
        ntk.draw(ax,
                ptopts={'mfc':'r', 'mec':'k', 'markersize':10},
                arrowopts={'fc':'k', 'lw':3})
        fp = os.path.join(self.FHERE, 'network.png')
        fig.savefig(fp)


class CompositeModelTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CompositeModelTestCase')

    def test_print(self):
        cm = CompositeModel('cm', 5, 3, 1)
        str_cm = '%s' % cm

    def test_compositemodel1(self):
        return

        nval = 100
        def get_model(nin, nout):
            m = NodeModel(nin, nout)
            inputs = np.random.uniform(0, 1, (nval,nin))
            m.allocate(inputs)
            return m

        # Connections
        # nd1 ->             -> outputs[0]
        #        nd3  -> nd4        -> outputs[1]
        # nd2 ->             -> nd5
        #                           -> outputs[2]
        ntk = CompositeNetwork()
        ntk['nd1'] = CompositeNode(get_model(1, 1))
        ntk['nd1'].add_child('nd3')

        ntk['nd2'] = CompositeNode(get_model(1, 1))
        ntk['nd2'].add_child('nd3', 0, 1)

        ntk['nd3'] = CompositeNode(get_model(2, 1))
        ntk['nd3'].add_child('nd4')

        ntk['nd4'] = CompositeNode(get_model(1, 2))
        ntk['nd4'].add_child('nd5', 1, 0)

        ntk['nd5'] = CompositeNode(get_model(1, 2))
        ntk.compute_runorder()

        #cm = CompositeModel('cm', 2, 3, 1)

        ## Add links
        #for (i, c) in enumerate(components):
        #    if c[0] in ['nd4', 'nd5']:
        #        model = get_node(1, 2)
        #    elif c[0] == 'nd3':
        #        model = get_node(2, 1)
        #    else:
        #        model = get_node(1, 1)

        #    if i > 0:
        #        if c[0] == components[i-1][0]:
        #            model = None

        #    cm.add_link(id=c[0],
        #        model=model,
        #        child_id=c[1],
        #        child_input_index=c[3],
        #        parent_output_index=c[2],
        #        composite_inputs_index=c[4],
        #        composite_outputs_index=c[5])

        ## Compute run order
        #cm.compute_run_order()

        ## Allocate
        #inputs = np.random.uniform(0, 1, (nval,2))
        #cm.allocate(inputs, 2)

        ## run
        #cm.run()

        #o1 = cm.outputs
        #o2 = np.dot(np.repeat(np.sum(inputs, axis=1).reshape((nval, 1)), 2, 1),
        #        np.diag([0.5, 0.5]))
        #ck = np.allclose(o1, o2)


