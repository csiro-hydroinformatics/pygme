
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Max number of uh */
#define HBV_MAXUH 10000

/* Number of inputs required by HBV run */
#define HBV_NINPUTS 2

/* Number of params required by HBV run */
#define HBV_NPARAMS 4

/* Number of states returned by HBV run */
#define HBV_NSTATES 2

/* Number of outputs returned by HBV run */
#define HBV_NOUTPUTS 9

int hbv_production(double P, double E,
        double S,
        double state0,
        double * prod);

int hbv_runtimestep(int nparams,
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

int c_hbv_run(int nval,
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

