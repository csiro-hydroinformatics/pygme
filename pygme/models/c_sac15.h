
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"
#include "c_uh.h"

/* Number of inputs required by SAC15 run */
#define SAC15_NINPUTS 5

/* Number of params required by SAC15 run */
#define SAC15_NPARAMS 15

/* Number of states returned by SAC15 run */
#define SAC15_NSTATES 6

/* Number of outputs returned by SAC15 run */
#define SAC15_NOUTPUTS 20

int sac15_runtimestep(int nparams,
    int nuh, int ninputs,
    int nstates, int noutputs,
	double * params,
    double * uh,
    double * inputs,
	double * statesuh,
    double * states,
    double * outputs);

int c_sac15_run(int nval,
    int nparams,
    int nuh,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
	double * params,
    double * uh,
	double * inputs,
    double * statesuhini,
	double * statesini,
    double * outputs);

