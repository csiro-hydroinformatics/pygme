#include "c_demand.h"

int demand_runtimestep(int nconfig, int ninputs,
    int nstates,
    int noutputs,
    double * config,
    double * inputs,
    double * states,
    double * outputs)
{
    int ierr=0, nbday, date[3];
    double day, Dmonth, Esum, E, Q, Qdown, Qmin;

    /* inputs  and states */
    Q = inputs[0];
    Qmin = inputs[1];
    day = states[0];
    Esum = states[1];

    /* Get month and day number */
    ierr = c_utils_getdate(day, date);
    if(ierr > 0)
        return ierr;

    /* Reset sum of extracted volume */
    if(date[2] == 1)
        Esum = 0;

    /* Get number of day in month */
    nbday = c_utils_daysinmonth(date[0], date[1]);
    if(nbday < 0)
        return 6000 + __LINE__;

    /* Compute extraction */
    Dmonth = config[date[1]-1];
    Qmin = Qmin < 0 ? 0 : Qmin;

    if(Esum >= Dmonth || Q <= Qmin || Dmonth <= 0.)
    {
        E = 0;
    }
    else
    {
        /* Compute remaining demand for the month */
        E = (Dmonth-Esum)/(nbday-date[2]+1);

        /* Correct E with Q and Qmin to have Qdown > Qmin */
        E = E > Q-Qmin ? Q-Qmin : E;
    }
    Esum += E;
    Qdown = Q-E;

    /* Add one day */
    ierr = c_utils_add1day(date);
    if(ierr > 0)
        return ierr;

    /* states */
    states[0] = (double)(date[0] * 10000 + date[1] * 100 + date[2]);
    states[1] = Esum;

    /* RESULTS */
    outputs[0] = E;

    if(noutputs>1)
        outputs[1] = Qdown;
    else
	    return ierr;

    if(noutputs>2)
        outputs[2] = Esum/Dmonth;
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
    if(nconfig < 12)
        return ESIZE_CONFIG;

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
    	ierr = demand_runtimestep(nconfig, ninputs,
                nstates,
                noutputs,
                config,
                &(inputs[ninputs*i]),
                statesini,
                &(outputs[noutputs*i]));

		if(ierr>0)
			return ierr;
    }

    return ierr;
}

