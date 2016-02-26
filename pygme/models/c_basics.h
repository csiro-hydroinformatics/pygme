
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Number of states returned by MONTHLYPATTERN run */
#define MONTHLYPATTERN_NSTATES 1

/* Number of outputs returned by MONTHLYPATTERN run */
#define MONTHLYPATTERN_NOUTPUTS 1

int monthlypattern_runtimestep(int nconfig, 
    int nstates,
    int noutputs,
    double * config,
    double * states,
    double * outputs);

int c_monthlypattern_run(int nval,
    int nconfig,
    int nstates,
    int noutputs,
    int start, int end,
    double * config,
    double * statesini,
    double * outputs);

