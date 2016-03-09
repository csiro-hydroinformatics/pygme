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

    if(noutputs > MONTHLYPATTERN_NOUTPUTS)
        return BASICS_ERROR + __LINE__;

    if(nstates > MONTHLYPATTERN_NSTATES)
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

