#include "c_gr6j.h"
#include "c_uh.h"


int gr6j_minmaxparams(int nparams, double * params)
{
    if(nparams<4)
        return GR6J_ERROR + __LINE__;

	params[0] = c_minmax(1, 1e5, params[0]); 	// S
	params[1] = c_minmax(-50, 50, params[1]);	// IGF
	params[2] = c_minmax(1, 1e5, params[2]); 	// R
	params[3] = c_minmax(0.5, 50, params[3]); // TB

    /* TODO */
	params[4] = c_minmax(0.5, 50, params[3]); // TB
	params[5] = c_minmax(0.5, 50, params[3]); // TB

	return 0;
}


int gr6j_runtimestep(int nparams,
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

    double Q, P, E, Q9, Q1;
    double prod[6];
    double ES, PS, PR;
    double PERC,ECH,TP,R2,QR,QD;
    double EN, ech1,ech2, RR;
    double uhoutput1[1], uhoutput2[1];

    /* inputs */
    P = inputs[0];
    P = P < 0 ? 0 : P;

    E = inputs[1];
    E = E < 0 ? 0 : E;

    /* GR4J Production */
    gr4j_production(P, E, params[0], states[0], prod);

    EN = prod[0];
    PS = prod[1];
    ES = prod[2];
    PERC = prod[3];
    PR = prod[4];
    states[0] = prod[5];

    /* UH */
    uh_runtimestep(nuh1, PR, uh1, statesuh1, uhoutput1);
    uh_runtimestep(nuh2, PR, uh2, statesuh2, uhoutput2);

    /* Potential Water exchange */
    RR = states[1]/params[2];
    ECH = params[1]*RR*RR*RR*sqrt(RR);
	//double rdivx3 = R / x3 - x5;
	//F = x2*pow(rdivx3, 1.0);

    /* Routing store calculation */
    Q9 = *uhoutput1 * 0.9;
    TP = states[1] + Q9 * 0.6 + ECH;

    /* Case where Reservoir content is not sufficient */
    ech1 = ECH-TP;
    states[1] = 0;

    if(TP>=0)
    {
        states[1]=TP;
        ech1=ECH;
    }

    RR = states[1]/params[2];
    R2 = states[1]/sqrt(sqrt(1.+RR*RR*RR*RR));
    QR = states[1]-R2;
    states[1] = R2;

    /* Direct runoff calculation */
    QD = 0;

    /* Case where the UH cannot provide enough water */
    Q1 = *uhoutput2 * 0.1
    TP = Q1 + ECH;
    ech2 = ECH-TP;
    QD = 0;

    if(TP>0)
    {
        QD = TP;
        ech2 = ECH;
    }

    /* Exponential reservoir */
    states[2] =  states[2] + Q9 * 0.4;

    /* TOTAL STREAMFLOW */
    Q = QD + QR;

    /* RESULTS */
    outputs[0] = Q;

    if(noutputs>1)
        outputs[1] = ech1+ech2;
    else
	return ierr;

    if(noutputs>2)
	    outputs[2] = ES+EN;
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

    if(noutputs>9)
	    outputs[9] = states[2];
	else
		return ierr;

	return ierr;
}


/* --------- Model runner ----------*/
int c_gr6j_run(int nval, int nparams,
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
    if(noutputs > GR6J_NOUTPUTS)
        return GR6J_ERROR + __LINE__;

    if(nstates > GR6J_NSTATES)
        return GR6J_ERROR + __LINE__;

    if(nuh1+nuh2 > NUHMAXLENGTH)
        return GR6J_ERROR + __LINE__;

    if(nuh1 <= 0 || nuh2 <= 0)
        return GR6J_ERROR + __LINE__;

    if(start < 0)
        return GR6J_ERROR + __LINE__;

    if(end >= nval)
        return GR6J_ERROR + __LINE__;

    /* Check parameters */
    ierr = gr6j_minmaxparams(nparams, params);

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
        /* Run timestep model and update states */
    	ierr = gr6j_runtimestep(nparams,
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

