import os
import re
import math
import unittest

import time

import numpy as np
import pandas as pd

from calendar import month_abbr as month

from pygme.compositemodel import CompositeModel
from pygme.models.basics import Node


class CompositeModelTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CompositeModelTestCase')

    def test_print(self):
        cm = CompositeModel('cm', 5, 3, 1)
        str_cm = '%s' % cm

    def test_compositemodel1(self):

        nd1 = Node(1, 1)

        nval = 100
        inputs = np.random.uniform(0, 1, (nval,1))
        nd1.allocate(inputs)

        nd2 = Node(2, 1)
        inputs = np.random.uniform(0, 1, (nval,2))
        nd2.allocate(inputs)

        components = [
            ['nd1', 'nd3', 0, 0],
            ['nd2', 'nd3', 1, 0],
            ['nd3', 'nd4', 0, 0],
            ['nd4', None, 0, 0]
        ]

        cm = CompositeModel('cm', 5, 3, 1)

        for c in components:
            if c[0] == 'nd3':
                model = nd2.clone()
            else:
                model = nd1.clone()

            cm.add_link(id=c[0], model=model,
                child_id=c[1],
                child_input_index=c[2],
                parent_output_index=c[3])

