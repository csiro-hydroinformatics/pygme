#include "c_gr2mdiff.h"

int c_gr2mdiff_runtimestep(int nconfig, int nparams, int ninputs,
        int nstates, int noutputs,
	    double * config,
	    double * params,
	    double * dparams,
        double * inputs,
        double * states,
        double * dstates,
        double * outputs,
        double * doutputs)
{
    int ierr=0;

    /* parameters */
    dn_double Scapacity = dn_init(params[0]);
    Scapacity.dot[0] = dparams[0]

    dn_double IGFcoef = dn_init(params[1]);
    IGFcoef.dot[1] = dparams[1]

    dn_double Rcapacity = dn_init(config[0]);

    /* model variables */
    dn_double P, E, WS;
    dn_double Sr, S, R, S1, S2, PHI, PSI, P1, P2, P3;
    dn_double R1, R2, F, Q, AE;

    /* inputs */
    P.val = inputs[0] < 0 ? 0 : inputs[0];
    E.val = inputs[1] < 0 ? 0 : inputs[1];

    S = c_minmax(0, params[0], states[0]);
    R = states[1];
    R = R < 0. ? 0. : R;

    /* main GR2M procedure */

    /* production */

    /* .. integration of dv/dt = [1-(v/X1)^2] P ..
       see Edijatno (1991) page 45.
    */
    WS = P/Scapacity;
    //WS = WS > 13 ? 13 : WS;
    PHI = c_tanh(WS); // Use fast tanh function. Cap WS at 4.9 though.
    S1 = (S+Scapacity*PHI)/(1+PHI*S/Scapacity);
    P1 = P+S-S1;

    /* .. integration of dv/dt = -v/X1 (2-v/X1) E ..*/
    WS = E/Scapacity;
    //WS = WS > 13 ? 13 : WS;
    PSI = c_tanh(WS); // Use fast tanh function. Cap WS at 4.9 though.
    S2 = S1*(1-PSI)/(1+PSI*(1-S1/Scapacity));
    AE = S1-S2;

    Sr = S2/Scapacity;
    S = S2/cbrt(1.+Sr*Sr*Sr);
    P2 = S2-S;
    P3 = P1 + P2;

    /* routing */
    R1 = R + P3;
    R2 = IGFcoef * R1;
    F = R2-R1;
    Q = R2*R2/(R2+Rcapacity);
    R = R2-Q;

    /* states */
    states[0] = S.val;
    states[1] = R.val;

    /* output */
    outputs[0] = Q.val;
    doutputs[0] = Q.dot[0];
    doutputs[1] = Q.dot[1];

    if(noutputs>1)
        outputs[1] = S;
    else
        return ierr;

    if(noutputs>2)
        outputs[2] = R;
    else
        return ierr;

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
int c_gr2mdiff_run(int nval,
    int nconfig,
    int nparams,
    int ninputs,
    int nstates,
    int noutputs,
    int ndoutputs,
    int start, int end,
    double * config,
    double * params,
    double * dparams,
    double * inputs,
    double * statesini,
    double * dstatesini,
    double * outputs,
    double * doutputs)
{
    int ierr, i;

    /* Check dimensions */
    if(nconfig != GR2MDIFF_NCONFIG)
        return GR2MDIFF_ERROR + __LINE__;

    if(nparams != GR2MDIFF_NPARAMS)
        return GR2MDIFF_ERROR + __LINE__;

    if(nstates != GR2MDIFF_NSTATES)
        return GR2MDIFF_ERROR + __LINE__;

    if(ninputs != GR2MDIFF_NINPUTS)
        return GR2MDIFF_ERROR + __LINE__;

    if(noutputs > GR2MDIFF_NOUTPUTS)
        return GR2MDIFF_ERROR + __LINE__;

    if(ndoutputs > GR2MDIFF_NDOUTPUTS)
        return GR2MDIFF_ERROR + __LINE__;

    if(start < 0)
        return GR2MDIFF_ERROR + __LINE__;

    if(end >=nval)
        return GR2MDIFF_ERROR + __LINE__;

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
    		    dparams,
                &(inputs[ninputs*i]),
                statesini,
                dstatesini,
                &(outputs[noutputs*i]),
                &(doutputs[ndoutputs*i]));

        if(ierr > 0 )
            return ierr;
    }

    return ierr;
}

