#include "c_gr4j.h"
#include "c_uh.h"


int gr4j_minmaxparams(int nparams, double * params)
{
    if(nparams<4)
        return GR4J_ERROR + __LINE__;

	params[0] = c_minmax(1e-2, 1e5, params[0]); 	// S
	params[1] = c_minmax(-100, 100, params[1]);	// IGF
	params[2] = c_minmax(1e-2, 1e5, params[2]); 	// R
	params[3] = c_minmax(0.5, 50, params[3]); // TB

	return 0;
}


int gr4j_production(double P, double E,
        double Scapacity,
        double S,
        double * prod)
{
    double SR, TWS, WS, PS, ES, EN=0, PR, PERC, S2;
    double AE;

    /* production store */
    SR = S/Scapacity;
    SR = SR > 1. ? 1. : SR;

    if(P>E)
    {
        WS =(P-E)/Scapacity;
        TWS = c_tanh(WS);

        ES = 0;
        PS = Scapacity*(1-SR*SR)*TWS;
        PS /= (1+SR*TWS);
    	PR = P-E-PS;
    	EN = 0;
        AE = E;
    }
    else
    {
    	WS = (E-P)/Scapacity;
        TWS = c_tanh(WS);

    	ES = S*(2-SR)*TWS;
        ES /= (1+(1-SR)*TWS);
    	PS = 0;
    	PR = 0;
    	EN = E-P;
        AE = ES+P;
    }
    S += PS-ES;

    /* percolation */
    SR = S/Scapacity/2.25;
    S2 = S/sqrt(sqrt(1.+SR*SR*SR*SR));

    PERC = S-S2;
    S = S2;
    PR += PERC;

    prod[0] = EN;
    prod[1] = PS;
    prod[2] = ES;
    prod[3] = AE;
    prod[4] = PERC;
    prod[5] = PR;
    prod[6] = S;

    return 0;
}


int gr4j_runtimestep(int nparams,
    int nuh1, int nuh2,
    int ninputs,
    int nstates,
    int noutputs,
    double * params,
    double * uh1,
    double * uh2,
    double * inputs,
    double * statesuh1,
    double * statesuh2,
    double * states,
    double * outputs)
{
    int ierr=0;

    double Q, P, E, Q1, Q9;
    double prod[7];
    double ES, PS, PR, AE;
    double PERC,ECH,TP,R2,QR,QD;
    double EN, ech1,ech2, RR, RR4;
    double uhoutput1[1], uhoutput2[1];

    double partition1 = 0.9;

    /* inputs */
    P = inputs[0];
    P = P < 0 ? 0 : P;

    E = inputs[1];
    E = E < 0 ? 0 : E;

    /* Production */
    gr4j_production(P, E, params[0], states[0], prod);

    EN = prod[0];
    PS = prod[1];
    ES = prod[2];
    AE = prod[3];
    PERC = prod[4];
    PR = prod[5];
    states[0] = prod[6];

    /* UH */
    uh_runtimestep(nuh1, PR, uh1, statesuh1, uhoutput1);
    uh_runtimestep(nuh2, PR, uh2, statesuh2, uhoutput2);

    /* Potential Water exchange */
    RR = states[1]/params[2];
    ECH = params[1]*RR*RR*RR*sqrt(RR);

    /* Routing store calculation */
    Q9 = *uhoutput1 * partition1;
    TP = states[1] + Q9 + ECH;

    /* Case where Reservoir content is not sufficient */
    ech1 = ECH-TP;
    states[1] = 0;

    if(TP>=0)
    {
        states[1]=TP;
        ech1=ECH;
    }

    RR = states[1]/params[2];
    RR4 = RR*RR;
    RR4 *= RR4;
    R2 = states[1]/sqrt(sqrt(1.+RR4));
    QR = states[1]-R2;
    states[1] = R2;

    /* Direct runoff calculation */
    QD = 0;

    /* Case where the UH cannot provide enough water */
    Q1 = *uhoutput2 * (1-partition1);
    TP = Q1 + ECH;
    ech2 = ECH-TP;
    QD = 0;

    if(TP>0)
    {
        QD = TP;
        ech2 = ECH;
    }

    /* TOTAL STREAMFLOW */
    Q = QD + QR;

    /* RESULTS */
    outputs[0] = Q;

    if(noutputs>1)
        outputs[1] = ech1+ech2;
    else
	return ierr;

    if(noutputs>2)
	    outputs[2] = AE;
    else
	    return ierr;

    if(noutputs>3)
	    outputs[3] = PR;
    else
	    return ierr;

    if(noutputs>4)
        outputs[4] = QD;
    else
        return ierr;

    if(noutputs>5)
        outputs[5] = QR;
    else
        return ierr;

    if(noutputs>6)
	outputs[6] = PERC;
    else
	    return ierr;

    if(noutputs>7)
	    outputs[7] = states[0];
	else
		return ierr;

    if(noutputs>8)
	    outputs[8] = states[1];
	else
		return ierr;

	return ierr;
}


/* --------- Model runner ----------*/
int c_gr4j_run(int nval, int nparams,
    int nuh1, int nuh2,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
    double * params,
    double * uh1,
    double * uh2,
    double * inputs,
    double * statesuh1,
    double * statesuh2,
    double * states,
    double * outputs)
{
    int ierr=0, i;

    /* Check dimensions */
    if(noutputs > GR4J_NOUTPUTS)
        return GR4J_ERROR + __LINE__;

    if(nstates > GR4J_NSTATES)
        return GR4J_ERROR + __LINE__;

    if(nuh1+nuh2 > NUHMAXLENGTH)
        return GR4J_ERROR + __LINE__;

    if(nuh1 <= 0 || nuh2 <= 0)
        return GR4J_ERROR + __LINE__;

    if(start < 0)
        return GR4J_ERROR + __LINE__;

    if(end >= nval)
        return GR4J_ERROR + __LINE__;

    /* Check parameters */
    ierr = gr4j_minmaxparams(nparams, params);

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
        /* Run timestep model and update states */
    	ierr = gr4j_runtimestep(nparams,
                nuh1, nuh2,
                ninputs,
                nstates,
                noutputs,
                params,
                uh1, uh2,
                &(inputs[ninputs*i]),
                statesuh1,
                statesuh2,
                states,
                &(outputs[noutputs*i]));

		if(ierr>0)
			return ierr;
    }

    return ierr;
}

