#ifndef __HAYAMI__
#define __HAYAMI__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"
#include "c_uh.h"

/* Max number of uh -> 100 * 24 = 100 days at hourly timestep */
#define HAYAMI_MAXUH 2400

/* Number of inputs required by LAG ROUTE run */
#define HAYAMI_NINPUTS 5

/* Number of params required by LAG ROUTE run */
#define HAYAMI_NPARAMS 5

/* Number of states returned by LAG ROUTE run */
#define HAYAMI_NSTATES 5

/* Number of outputs returned by LAG ROUTE run */
#define HAYAMI_NOUTPUTS 10

/* Maximum argument to exponential in hayami kernel */
#define HAYAMI_EXP_ARGMAX 100

/* Minimum time in hayami kernel (sec) */
#define HAYAMI_TMIN 1e-3


int c_hayami_get_maxuh();

double hayami_kernel(double theta, double z, double t);

double hayami_kernel_diff(double theta, double z, double t);

double hayami_kernel_tmax(double theta, double z);

double integrate_hayami_kernel(double a, double b, double theta, double z);

int hayami_kernel_tbounds(double theta, double z, double eps, double tbounds[2]);

int c_uh_getuh_hayami(int nuhlengthmax,
                      double timestep,
                      double theta,
                      double z,
                      int * nuh,
                      double * uh);

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
