
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"
#include "c_gr4j.h"

/* Number of inputs required by GR6J run */
#define GR6J_NINPUTS 5

/* Number of params required by GR6J run */
#define GR6J_NPARAMS 6

/* Number of states returned by GR6J run */
#define GR6J_NSTATES 6

/* Number of outputs returned by GR6J run */
#define GR6J_NOUTPUTS 20

int gr6j_runtimestep(int nparams,
    int nuh1, int nuh2, int ninputs,
    int nstates, int noutputs,
	double * params,
    double * uh1,
    double * uh2,
    double * inputs,
	double * statesuh1,
	double * statesuh2,
    double * states,
    double * outputs);

int c_gr6j_run(int nval,
    int nparams,
    int nuh1,
    int nuh2,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
	double * params,
    double * uh1,
    double * uh2,
	double * inputs,
    double * statesuh1,
    double * statesuh2,
	double * statesini,
    double * outputs);

