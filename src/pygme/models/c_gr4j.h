#ifndef __GR4J__
#define __GR4J__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Percolation factor :
   daily = 2.25
   hourly = 4
*/
#define GR4J_PERCFACTOR 2.25

/* Number of inputs required by GR4J run */
#define GR4J_NINPUTS 2

/* Number of params required by GR4J run */
#define GR4J_NPARAMS 4

/* Number of states returned by GR4J run */
#define GR4J_NSTATES 2

/* Number of outputs returned by GR4J run */
#define GR4J_NOUTPUTS 11

int c_compute_PmEm(int nval,double * rain, double * evap, double* PmEm);

double c_gr4j_X1_initial_objfun(double Pm,double Em, double X1, double Sini);

int c_gr4j_X1_initial(double Pm, double Em, double X1, double * solution);

int gr4j_production(double P, double E,
        double S,
        double state0,
        double * prod);

int gr4j_runtimestep(int nparams,
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

int c_gr4j_run(int nval,
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

#endif
