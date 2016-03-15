
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

#define SINUSPATTERN_NUMIN 1e-30

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

int sinuspattern_runtimestep(int is_cumulative, 
    int nparams, 
    int nstates,
    int noutputs,
    double * params,
    double * states,
    double * outputs);

int c_sinuspattern_run(int nval,
    int nconfig, 
    int nparams,
    int nstates,
    int noutputs,
    int start, int end,
    double * config,
    double * params,
    double * statesini,
    double * outputs);
