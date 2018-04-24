#include "c_gr6j.h"
#include "c_uh.h"


int gr6j_minmaxparams(int nparams, double * params)
{
    if(nparams < 6)
    {
        return GR6J_ERROR + __LINE__;
    }

	params[0] = c_minmax(1, 1e4, params[0]); 	// S
	params[1] = c_minmax(-50, 50, params[1]);	// IGF
	params[2] = c_minmax(1, 1e4, params[2]); 	// R
	params[3] = c_minmax(0.5, 50, params[3]); // TB
	params[4] = c_minmax(-50, 50, params[4]); // IGF 2
	params[5] = c_minmax(1, 1e4, params[5]); // A

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
    double prod[7];
    double ES, PS, PR, AE;
    double PERC,ECH,TP,R2,QR,QD;
    double EN, ech1,ech2, RR, RR4, AR, QRExp;
    double uhoutput1[1], uhoutput2[1];

    double partition1 = 0.9;
    double partition2 = 0.6;

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
    AE = prod[3];
    PERC = prod[4];
    PR = prod[5];
    states[0] = prod[6];

    /* UH */
    uh_runtimestep(nuh1, PR, uh1, statesuh1, uhoutput1);
    uh_runtimestep(nuh2, PR, uh2, statesuh2, uhoutput2);

    /* Potential Water exchange */
    RR = states[1]/params[2];
    ECH = params[1]*(RR-params[4]);

    /* Routing store calculation */
    Q9 = *uhoutput1 * partition1;
    TP = states[1] + Q9*partition2 + ECH;

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

    /* Exponential reservoir */
    states[2] +=  ECH + Q9*(1-partition2);
    AR = c_minmax(-33, 33, states[2]/params[5]);

    if(AR > 7) QRExp = states[2]+params[5]/exp(AR);
    else if (AR < -7) QRExp = params[5]*exp(AR);
    else QRExp = params[5]*log(exp(AR)+1);

    states[2] -= QRExp;

    /* TOTAL STREAMFLOW */
    Q = QD + QR + QRExp;
    Q = Q < 0 ? 0 : Q;

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
	outputs[7] = QRExp;
    else
	    return ierr;

    if(noutputs>8)
	    outputs[8] = states[0];
	else
		return ierr;

    if(noutputs>9)
	    outputs[9] = states[1];
	else
		return ierr;

    if(noutputs>10)
	    outputs[10] = states[2];
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

