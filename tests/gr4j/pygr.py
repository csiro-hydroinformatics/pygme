from math import tanh, sqrt
import numpy as np
import pandas as pd
import logging

# Setup login
LOGGER = logging.getLogger(__name__)

def ss1(ordinate, lag):
    lag = max(0.5, lag)
    if ordinate < 0:
        return 0
    elif ordinate <= lag:
        return (ordinate/lag)**2.5
    else:
        return 1

def ss2(ordinate, lag):
    lag = max(0.5, lag)
    if ordinate < 0:
        return 0
    elif ordinate <= lag:
        return 0.5*(ordinate/lag)**2.5
    elif ordinate <= 2*lag:
        return 1-0.5*(2-ordinate/lag)**2.5
    else:
        return 1

def uh_ord(nord, lag):
    ''' Compute uh ordinates '''
    ord1 = np.zeros(nord)
    ord2 = np.zeros(nord)
    suh1, suh2 = 0., 0.
    nuh1, nuh2 = 0, 0
    EPS = 1e-10

    for i in range(nord):
        if suh1 < 1-EPS:
            nuh1 += 1
        else:
            break

        ord1[i] = ss1(i+1, lag)-ss1(i, lag)
        suh1 += ord1[i]

    for i in range(nord):
        if suh2 < 1-EPS:
            nuh2 += 1
        else:
            break

        ord2[i] = ss2(i+1, lag)-ss2(i, lag)
        suh2 += ord2[i]

    return nuh1, ord1, nuh2, ord2


def run(LOGGER, rain, evap, params, states, ord1, ord2, uh1, uh2, nuh1, nuh2):
    ''' Run gr4j model with no uh '''
    # constants
    partition1 = 0.9

    # inputs
    nval = len(rain)
    columns = ['P', 'E', 'ES', 'PS', 'PR', 'EN', \
            'AE', 'PERC', 'S', 'UH1', 'UH2', 'R', \
            'Q9', 'QR', 'QD', 'Q', 'ECH']
    outputs = pd.DataFrame(np.zeros((nval, len(columns))), \
                    columns=columns)

    # Time loop
    for i in range(nval):
        if i%50 == 0 and i> 0:
            LOGGER.info('Processing timestep {0}/{1}'.format(i+1, nval))

        # Get states
        S, R = states

        # Get inputs
        P = rain[i]
        E = evap[i]

        # production store with maximum filling level of 100%
        Scapacity = params[0]
        SR = S/Scapacity
        SR = min(1, SR)

        if P>E:
            WS =(P-E)/Scapacity
            TWS = tanh(WS)

            ES = 0
            PS = Scapacity*(1-SR*SR)*TWS
            PS /= (1+SR*TWS)
            PR = P-E-PS
            EN = 0
            AE = E
        else:
            WS = (E-P)/Scapacity
            TWS = tanh(WS)

            ES = S*(2-SR)*TWS
            ES /= (1+(1-SR)*TWS)
            PS = 0
            PR = 0
            EN = E-P
            AE = ES+P

        S += PS-ES

        # percolation
        SR = S/Scapacity/2.25
        S2 = S/sqrt(sqrt(1.+SR*SR*SR*SR))

        PERC = S-S2
        S = S2
        PR += PERC
        states[0] = S

        #/* UH */
        for k in range(nuh1-1):
            uh1[k] = uh1[k+1]+ord1[k]*PR

        uh1[nuh1-1] = ord1[nuh1-1]*PR

        for k in range(nuh2-1):
            uh2[k] = uh2[k+1]+ord2[k]*PR

        uh2[nuh1-1] = ord2[nuh2-1]*PR

        uhoutput1 = uh1[0]
        uhoutput2 = uh2[0]

        # Potential Water exchange
        RR = states[1]/params[2]
        ECH = params[1]*RR*RR*RR*sqrt(RR)

        #/* Routing store calculation */
        Q9 = uhoutput1 * partition1
        TP = states[1] + Q9 + ECH

        #/* Case where Reservoir content is not sufficient */
        ech1 = ECH-TP
        states[1] = 0

        if TP>=0:
            states[1] = TP
            ech1 = ECH
        RR = states[1]/params[2]
        RR4 = RR*RR
        RR4 *= RR4
        R2 = states[1]/sqrt(sqrt(1.+RR4))
        QR = states[1]-R2
        states[1] = R2

        #/* Direct runoff calculation */
        QD = 0

        #/* Case where the UH cannot provide enough water */
        Q1 = uhoutput2 * (1-partition1)
        TP = Q1 + ECH
        ech2 = ECH-TP
        QD = 0

        if TP>0:
            QD = TP
            ech2 = ECH

        #/* TOTAL STREAMFLOW */
        Q = QD + QR

        # Store
        outputs.loc[i, 'P'] = P
        outputs.loc[i, 'E'] = E
        outputs.loc[i, 'ES'] = ES
        outputs.loc[i, 'PS'] = PS
        outputs.loc[i, 'PR'] = PR
        outputs.loc[i, 'EN'] = EN
        outputs.loc[i, 'AE'] = AE
        outputs.loc[i, 'PERC'] = PERC
        outputs.loc[i, 'S'] = S
        outputs.loc[i, 'UH1'] = uh1[0]
        outputs.loc[i, 'UH2'] = uh2[0]
        outputs.loc[i, 'R'] = R2
        outputs.loc[i, 'Q9'] = Q9
        outputs.loc[i, 'QR'] = QR
        outputs.loc[i, 'QD'] = QD
        outputs.loc[i, 'Q'] = Q
        outputs.loc[i, 'ECH'] = ECH

    return outputs
