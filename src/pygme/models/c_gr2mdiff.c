#include "c_gr2mdiff.h"

dn_double dn_init(double val) {
    dn_double dn;
    dn.val = val;
    for(int i = 0; i < GR2MDIFF_NDOT; i ++ )
        dn.dot[i] = 0.;
    return dn;
}

dn_double dn_add(dn_double dn1, dn_double dn2) {
    dn_double dn;
    dn.val = dn1.val + dn2.val;
    for(int i = 0; i < GR2MDIFF_NDOT; i ++ )
        dn.dot[i] = dn1.dot[i] + dn2.dot[i];
    return dn;
}

dn_double dn_prod(dn_double dn1, dn_double dn2) {
    dn_double dn;
    dn.val = dn1.val * dn2.val;
    for(int i = 0; i < GR2MDIFF_NDOT; i ++ )
        dn.dot[i] = dn1.dot[i] * dn2.val + dn1.val * dn2.dot[i];
    return dn;
}

dn_double dn_div(dn_double dn1, dn_double dn2) {
    dn_double dn;
    dn.val = dn1.val / dn2.val;
    for(int i = 0; i < GR2MDIFF_NDOT; i ++ )
        dn.dot[i] = (dn1.dot[i] - dn2.dot[i] * dn.val) / dn2.val;
    return dn;
}

dn_double dn_tanh(dn_double dn1) {
    dn_double dn;
    dn.val = tanh(dn1.val);
    for(int i = 0; i < GR2MDIFF_NDOT; i ++ )
        dn.dot[i] = dn1.dot[i] * (1 + dn.val * dn.val);
    return dn;
}

dn_double dn_intpow(dn_double dn1, int n) {
    dn_double dn;
    double val = 1.;
    /* val power n-1 to be used later in derivative */
    for(int k = 0; k < n - 1; k ++ )
        val *= dn1.val;

    dn.val = val * dn1.val;

    for(int i = 0; i < GR2MDIFF_NDOT; i ++ )
        dn.dot[i] = (n - 1) * dn1.dot[i] * val;
    return dn;
}


dn_double dn_cbrt(dn_double dn1) {
    dn_double dn;
    dn.val = cbrt(dn1.val);
    for(int i = 0; i < GR2MDIFF_NDOT; i ++ )
        dn.dot[i] = 1./3. * dn1.dot[i] / (dn1.val * dn1.val);
    return dn;
}


int c_gr2mdiff_runtimestep(int nconfig, int nparams, int ninputs,
        int nstates, int noutputs,
	    double * config,
	    double * params,
	    double * dparams,
        double * inputs,
        double * states,
        double * outputs,
        double * doutputs)
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
    double dQ;

    /* inputs */
    P = inputs[0] < 0 ? 0 : inputs[0];
    E = inputs[1] < 0 ? 0 : inputs[1];

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
    states[0] = S;
    states[1] = R;

    /* output */
    outputs[0] = Q;
    doutputs[0] = dQ;

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
                &(outputs[noutputs*i]),
                &(doutputs[ndoutputs*i]));

        if(ierr > 0 )
            return ierr;
    }

    return ierr;
}

