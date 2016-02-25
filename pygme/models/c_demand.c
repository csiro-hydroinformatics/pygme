
#include "c_demand.h"


int demand_runtimestep(int nconfig, int ninputs,
    int nstates,
    int noutputs,
    double * config,
    double * inputs,
    double * states,
    double * outputs)
{
    int ierr=0, month;
    double day, nday, nbday, D, Dsum, E, Q;

    /* inputs  and states */
    Q = inputs[0];
    day = states[0];
    Dsum = states[1];

    /* Get month and day number */
    month = rint(day*1e-2 - floor(day*1e-4)*1e2);
    nday = rint(day - floor(day*1e-6)*1e4 - nmonth*1e2);

    /* Get number of day in month */
    nbday = c_utils_dayinmonth(;

    /* Get demand */
    D = config[month-1];


    /* Add one day */



    /* RESULTS */
    outputs[0] = E;

    if(noutputs>1)
        outputs[1] = D;
    else
	    return ierr;

	return ierr;
}


/* --------- Model runner ----------*/

int c_demand_run(int nval,
    int nconfig,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
	double * config,
	double * inputs,
	double * statesini,
    double * outputs)
{
    int ierr=0, i;

    /* Check dimensions */
    if(noutputs > DEMAND_NOUTPUTS)
        return ESIZE_OUTPUTS;

    if(nstates > DEMAND_NSTATES)
        return ESIZE_STATES;

    if(start < 0)
        return ESIZE_OUTPUTS;

    if(end >= nval)
        return ESIZE_OUTPUTS;

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
        /* Run timestep model and update states */
    	ierr = demand_runtimestep(
                ninputs,
                nstates,
                noutputs,
                &(inputs[ninputs*i]),
                states,
                &(outputs[noutputs*i]));

		if(ierr>0)
			return ierr;
    }

    return ierr;
}

