
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"


/* Maximum size of resampling kernel */
#define KNN_NKERNEL_MAX 150


int c_knndaily_run(int nparams, int nval, int nvar, int nrand, 
    double * params,
    double * weights,
    double * var,
    double * rand,
    double * outputs)

