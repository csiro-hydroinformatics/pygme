#ifndef __WAPABA__
#define __WAPABA__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Number of inputs required by WAPABA run */
#define WAPABA_NINPUTS 2

/* Number of params required by WAPABA run */
#define WAPABA_NPARAMS 5

/* Number of states returned by WAPABA run */
#define WAPABA_NSTATES 2

/* Number of outputs returned by WAPABA run */
#define WAPABA_NOUTPUTS 10

int c_wapaba_run(int nval, int nparams, int ninputs,
    int nstates, int noutputs,
    int start, int  end,
	double * params,
	double * inputs,
	double * statesini,
    double * outputs);

#endif
