#include "c_gr2m.h"


int gr2m_minmaxparams(int nparams, double * params)
{
    if(nparams<4)
    {
        return GR2M_ERROR + __LINE__;
    }

	params[0] = c_minmax(1,1e5,params[0]); 	// S
	params[1] = c_minmax(0,3,params[1]);	// IGF

	return 0;
}


/*******************************************************************************
* Run time step code for the GR2M rainfall-runoff model
*
* --- Inputs
* ierr			Error message
* nconfig		Number of configuration elements (1)
* nparams			Number of paramsameters (4)
* ninputs		Number of inputs (2)
* nstates		Number of states (1 output + 2 model states + 8 variables = 11)
* nuh			Number of uh ordinates (2 uh)
*
* params			Model paramsameters. 1D Array nparams(4)x1
*					params[0] = S
*					params[1] = IGF
*
* uh			uh ordinates. 1D Array nuhx1
*
* inputs		Model inputs. 1D Array ninputs(2)x1
*
* statesuh		uh content. 1D Array nuhx1
*
* states		Output and states variables. 1D Array nstates(11)x1
*
*******************************************************************************/

int c_gr2m_runtimestep(int nconfig, int nparams, int ninputs,
        int nstates, int noutputs,
	    double * config,
	    double * params,
        double * inputs,
        double * states,
        double * outputs)
{
    int ierr=0;

    /* parameters */
    double Scapacity = params[0];
    double IGFcoef = params[1];
    double Rcapacity = config[0];

    /* model variables */
    double P, E, WS;
    double Sr, S, R, S1, S2, PHI, PSI, P1, P2, P3;
    double R1, R2, F, Q, AE;

    /* inputs */
    P = inputs[0] < 0 ? 0 : inputs[0];
    E = inputs[1] < 0 ? 0 : inputs[1];

    S = c_minmax(0, params[0], states[0]);
    R = states[1];
    R = R < 0. ? 0. : R;

    /* main GR2M procedure */

    /* production */
    WS = P/Scapacity;
    WS = WS > 13 ? 13 : WS;
    PHI = tanh(WS);
    S1 = (S+Scapacity*PHI)/(1+PHI*S/Scapacity);
    P1 = P+S-S1;

    WS = E/Scapacity;
    WS = WS > 13 ? 13 : WS;
    PSI = tanh(WS);
    S2 = S1*(1-PSI)/(1+PSI*(1-S1/Scapacity));
    AE = S1-S2;

    Sr = S2/Scapacity;
    S = S2/pow(1.+Sr*Sr*Sr, 1./3);
    P2 = S2-S;
    P3 = P1 + P2;

    /* routing */
    R1 = R + P3;
    R2 = IGFcoef * R1;
    F = R2-R1;
    Q = R2*R2/(R2+Rcapacity);
    R = R2-Q;

    /* states */
    states[0] = S;
    states[1] = R;

    /* output */
    outputs[0] = Q;

    if(noutputs>1)
        outputs[1] = S;
    else
        return ierr;

    if(noutputs>2)
        outputs[2] = R;

    if(noutputs>3)
        outputs[3] = F;
    else
        return ierr;

    if(noutputs>4)
        outputs[4] = P1;
    else
        return ierr;

    if(noutputs>5)
        outputs[5] = P2;
    else
        return ierr;

    if(noutputs>6)
        outputs[6] = P3;
    else
        return ierr;

    if(noutputs>7)
        outputs[7] = R1;
    else
        return ierr;

    if(noutputs>8)
        outputs[8] = R2;
    else
        return ierr;

    if(noutputs>9)
        outputs[9] = AE;
    else
        return ierr;

    if(noutputs>10)
        outputs[10] = S1;
    else
        return ierr;

    if(noutputs>11)
        outputs[11] = S2;
    else
        return ierr;


    return ierr;
}


// --------- Component runner --------------------------------------------------
int c_gr2m_run(int nval,
    int nconfig,
    int nparams,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
    double * config,
    double * params,
    double * inputs,
    double * statesini,
    double * outputs)
{
    int ierr, i;

    /* Check dimensions */
    if(nconfig != GR2M_NCONFIG)
        return GR2M_ERROR + __LINE__;

    if(nparams != GR2M_NPARAMS)
        return GR2M_ERROR + __LINE__;

    if(nstates != GR2M_NSTATES)
        return GR2M_ERROR + __LINE__;

    if(ninputs != GR2M_NINPUTS)
        return GR2M_ERROR + __LINE__;

    if(noutputs > GR2M_NOUTPUTS)
        return GR2M_ERROR + __LINE__;

    if(start < 0)
        return GR2M_ERROR + __LINE__;

    if(end >=nval)
        return GR2M_ERROR + __LINE__;

    /* Check parameters */
    ierr = gr2m_minmaxparams(nparams, params);

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
       /* Run timestep model and update states */
    	ierr = c_gr2m_runtimestep(nconfig, nparams,
                ninputs,
                nstates,
                noutputs,
                config,
    		    params,
                &(inputs[ninputs*i]),
                statesini,
                &(outputs[noutputs*i]));

        if(ierr > 0 )
            return ierr;
    }

    return ierr;
}

