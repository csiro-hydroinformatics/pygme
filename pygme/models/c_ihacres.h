#ifndef __IHACRES__
#define __IHACRES__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Number of inputs required by IHACRES run */
#define IHACRES_NCONFIG 2

/* Number of inputs required by IHACRES run */
#define IHACRES_NINPUTS 2

/* Number of params required by IHACRES run */
#define IHACRES_NPARAMS 2

/* Number of states returned by IHACRES run */
#define IHACRES_NSTATES 1

/* Number of outputs returned by IHACRES run */
#define IHACRES_NOUTPUTS 10

int c_ihacres_run(int nval,
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
    double * outputs);

#endif
