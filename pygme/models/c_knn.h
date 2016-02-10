
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"


/* Maximum size of resampling kernel */
#define KNN_NKERNEL_MAX 150

/* Maximum number of variables to evaluate distance */
#define KNN_NVAR_MAX 10


int c_knn_run(int nparams, int nval, int nvar, int nrand,
    double * params,
    double * weights,
    double * var,
    double * rand,
    double * outputs)

