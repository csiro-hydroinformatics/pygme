
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Max number of uh */
#define HBV_MAXUH 500

/* Number of inputs required by HBV run */
#define HBV_NINPUTS 2

/* Number of params required by HBV run */
#define HBV_NPARAMS 10

/* Number of states returned by HBV run */
#define HBV_NSTATES 3

/* Number of outputs returned by HBV run */
#define HBV_NOUTPUTS 13

int hbv_soilmoisture(double rain, double etp, double moist,
        double LP, double FC, double BETA, double *prod);

int hbv_respfunc(double dq, double K0, double LSUZ,
        double k1, double K2, double CPERC, double BMAX, double CROUTE,
        double suz, double slz,
        double *resp, int *bql, double *dqh);

int hbv_runtimestep(int nparams,
    int ninputs,
    int nstates,
    int noutputs,
    double * params,
    double * inputs,
    double * states,
    double * outputs,
    int * bql,
    double * dquh);

int c_hbv_run(int nval, int nparams,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
    double * params,
    double * inputs,
    double * statesini,
    double * outputs);

