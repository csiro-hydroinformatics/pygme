#include "c_basics.h"

int monthlypattern_runtimestep(int nconfig, 
    int nstates,
    int noutputs,
    double * config,
    double * states,
    double * outputs)
{
    int ierr=0, nbday, date[3];
    double day, Dmonth, D; 

    /* inputs  and states */
    day = states[0];

    /* Get month and day number */
    ierr = c_utils_getdate(day, date);
    if(ierr > 0)
        return ierr;

    /* Get number of day in month */
    nbday = c_utils_daysinmonth(date[0], date[1]);
    if(nbday < 0)
        return BASICS_ERROR + __LINE__;

    /* Compute extraction */
    Dmonth = config[date[1]-1];
    D = Dmonth/nbday;

    /* Add one day */
    ierr = c_utils_add1day(date);
    if(ierr > 0)
        return ierr;

    /* states */
    states[0] = (double)(date[0] * 10000 + date[1] * 100 + date[2]);

    /* RESULTS */
    outputs[0] = D;

    return ierr;
}


/* --------- Model runner ----------*/

int c_monthlypattern_run(int nval,
    int nconfig,
    int nstates,
    int noutputs,
    int start, int end,
    double * config,
    double * statesini,
    double * outputs)
{
    int ierr=0, i;

    /* Check dimensions */
    if(nconfig < 12)
        return BASICS_ERROR + __LINE__;

    if(start < 0)
        return BASICS_ERROR + __LINE__;

    if(end >= nval)
        return BASICS_ERROR + __LINE__;

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
        /* Run timestep model and update states */
    	ierr = monthlypattern_runtimestep(nconfig,
                nstates,
                noutputs,
                config,
                statesini,
                &(outputs[noutputs*i]));

	if(ierr>0)
	    return ierr;
    }

    return ierr;
}


int sinuspattern_runtimestep(int is_cumulative, int nparams, 
    int nstates,
    int noutputs,
    double * params,
    double * states,
    double * outputs)
{
    int ierr=0, day, date[3];
    double doy, lower, upper, S, CS, phi, nu; 

    /* inputs  and states */
    day = states[0];
    CS = states[1];

    /* Get month and day number */
    ierr = c_utils_getdate(day, date);
    if(ierr > 0)
        return ierr;

    /* Get day of year */
    doy = (double) c_utils_dayofyear(date[1], date[2]);
    if(doy < 0)
        return BASICS_ERROR + __LINE__;

    /* Reset cumulative */
    if(doy < 2)
        CS = 0;

    /* model parameters */
    lower = params[0];
    upper = params[1];
    phi = params[2];
    nu = params[3];

    /* Run sinus component */
    S = (sin((doy/365-phi)*2*UTILS_PI)+1)/2;

    /* Run shape component */
    if(abs(nu) > SINUSPATTERN_NUMIN)
        S = (exp(S*sinh(nu))-1)/(exp(sinh(nu))-1);

    /* Run scaling component */
    S = lower+(upper-lower)*S;

    /* Cumulative */
    if(is_cumulative == 1)
        CS += S;
    else
        CS = S;

    /* Add one day */
    ierr = c_utils_add1day(date);
    if(ierr > 0)
        return ierr;

    /* states */
    states[0] = (double)(date[0] * 10000 + date[1] * 100 + date[2]);
    states[1] = CS;

    /* RESULTS */
    outputs[0] = CS;

    return ierr;
}


/* --------- Model runner ----------*/

int c_sinuspattern_run(int nval,
    int nconfig,
    int nparams,
    int nstates,
    int noutputs,
    int start, int end,
    double * config,
    double * params,
    double * statesini,
    double * outputs)
{
    int ierr=0, i, is_cumulative;
    double lower, upper, phi, nu;

    /* Check dimensions */
    if(nconfig < 1)
        return BASICS_ERROR + __LINE__;

    if(nparams < 4)
        return BASICS_ERROR + __LINE__;

    if(nstates < 0)
        return BASICS_ERROR + __LINE__;

    if(start < 0)
        return BASICS_ERROR + __LINE__;

    if(end >= nval)
        return BASICS_ERROR + __LINE__;

    /* check model parameters */
    lower = params[0];
    if(isnan(lower))
        return BASICS_ERROR + __LINE__;

    upper = params[1];
    if(isnan(upper))
        return BASICS_ERROR + __LINE__;

    phi = params[2];
    nu = params[3];

    upper = upper < lower ? lower : upper;
    phi = phi < 0. ? 0. : phi > 1. ? 1. : phi;
    nu = nu < -6. ? -6. : nu > 6. ? 6. : nu;

    params[0] = lower;
    params[1] = upper;
    params[2] = phi;
    params[3] = nu;

    /* define config */
    is_cumulative = rint(config[0]);
    if(is_cumulative != 0 && is_cumulative != 1)
        return BASICS_ERROR + __LINE__;

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
        /* Run timestep model and update states */
    	ierr = sinuspattern_runtimestep(is_cumulative, nparams,
                nstates,
                noutputs,
                params,
                statesini, 
                &(outputs[noutputs*i]));

	if(ierr>0)
	    return ierr;
    }

    return ierr;
}

