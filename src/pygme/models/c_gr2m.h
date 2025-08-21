#ifndef __GR2M__
#define __GR2M__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Number of config required by GR2M run */
#define GR2M_NCONFIG 1

/* Number of inputs required by GR2M run */
#define GR2M_NINPUTS 2

/* Number of params required by GR2M run */
#define GR2M_NPARAMS 2

/* Number of states returned by GR2M run */
#define GR2M_NSTATES 2

/* Number of outputs returned by GR2M run */
#define GR2M_NOUTPUTS 12

int c_gr2m_run(int nval, int nconfig, int nparams, int ninputs,
    int nstates, int noutputs,
    int start, int  end,
	double * config,
	double * params,
	double * inputs,
	double * statesini,
    double * outputs);

#endif
