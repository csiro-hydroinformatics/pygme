
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "c_utils.h"

/* Maximum size of resampling kernel */
#define KNNDAILY_NKERNEL_MAX 50

/* Maximum number of variables to evaluate distance */
#define KNNDAILY_NVAR_MAX 20

/* Minimum value of positive weight */
#define KNNDAILY_WEIGHT_MIN 1e-20

/* Value used to initialise KNN distance */
#define KNNDAILY_DIST_MAX 1e100

int c_knndaily_run(int nconfig, int nval, int nvar, int nrand,
    int start, int end,
    double * config,
    double * rand,
    double * var,
    double * states,
    int * knn_ipos);

