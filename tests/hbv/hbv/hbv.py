import math
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle


from hbvmodel import adaptor

def run(prec, ep, incon, param):
    ''' Run hbv '''
    # Model inputs
    prec = np.array(prec, dtype=np.float64, copy=False, ndmin=1)
    ep = np.array(ep, dtype=np.float64, copy=False, ndmin=1)

    itsteps = prec.shape[0]
    if itsteps != ep.shape[0]:
        raise ValueError('prec and ep do not have the same length')

    # Additional inputs
    airt = 30 + ep*0
    incon = np.array([50, 2.5, 2.5], dtype=np.float64)
    area = np.array([1.], dtype=np.float64)

    # Model parameters
    param = np.zeros((15, 1), dtype=np.float64)
    # .. snow melt params ..
    param[:2, :] = 1.
    param[2:4, :] = 0.
    # .. rr parameters ..
    param[4:, 0] = param

    # Preparing outputs
    output = np.zeros(P.shape, dtype=np.float64)

    adaptor.run(itsteps, 1, area, param, incon,\
                            prec, airt, ep, output)

    return output

