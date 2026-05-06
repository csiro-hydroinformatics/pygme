#ifndef __HAYAMI__
#define __HAYAMI__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Number of inputs required by LAG ROUTE run */
#define HAYAMI_NINPUTS 5

/* Number of params required by LAG ROUTE run */
#define HAYAMI_NPARAMS 5

/* Number of states returned by LAG ROUTE run */
#define HAYAMI_NSTATES 5

/* Number of outputs returned by LAG ROUTE run */
#define HAYAMI_NOUTPUTS 10

int c_hayami_run(int nval,
        int nparams,
        int nuh,
        int ninputs,
        int nconfig,
        int nstates,
        int noutputs,
        int start, int end,
        double * config,
	    double * params,
        double * uh,
	    double * inputs,
        double * statesuh,
	    double * states,
        double * outputs);

#endif
