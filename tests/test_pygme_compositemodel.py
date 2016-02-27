import os
import re
import math
import unittest

import time

import numpy as np
import pandas as pd

from calendar import month_abbr as month

from pygme.compositemodel import CompositeModel, CompositeNetwork, CompositeNode
from pygme.models.basics import NodeModel

class CompositeNodeTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CompositeNodeTestCase')

    def test_node1(self):



class CompositeNetworkTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CompositeNetworkTestCase')

    def test_network1(self):
        pass


class CompositeModelTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CompositeModelTestCase')

    def test_print(self):
        pass
        #cm = CompositeModel('cm', 5, 3, 1)
        #str_cm = '%s' % cm

    def test_compositemodel1(self):
        return

        nval = 100
        def get_node(nin, nout):
            nd = Node(nin, nout)
            inputs = np.random.uniform(0, 1, (nval,nin))
            nd.allocate(inputs)
            return nd

        # Connections
        # nd1 ->             -> outputs[0]
        #        nd3  -> nd4        -> outputs[1]
        # nd2 ->             -> nd5
        #                           -> outputs[2]
        components = [
            ['nd1', 'nd3', 0, 0, [(0, 0)], None],
            ['nd2', 'nd3', 0, 1, [(1, 0)], None],
            ['nd3', 'nd4', 0, 0, None, None],
            ['nd4', None, 0, 0, None, [(0, 0)]],
            ['nd4', 'nd5', 1, 0, None, None],
            ['nd5', None, 0, 0, None, [(0, 1), (1, 2)]]
        ]

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


