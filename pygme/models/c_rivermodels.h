
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Number of inputs required by DEMAND run */
#define DEMAND_NINPUTS 5

/* Number of states returned by DEMAND run */
#define DEMAND_NSTATES 2

/* Number of outputs returned by DEMAND run */
#define DEMAND_NOUTPUTS 3

int c_demand_runtimestep(int nconfig, int nstates, int noutputs,
    double * config,
    double * inputs,
    double * states,
    double * outputs);

int c_demand_run(int nval,
    int nconfig,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
	double * config,
	double * inputs,
	double * statesini,
    double * outputs);

