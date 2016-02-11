
#include <math.h>
#include <stdlib.h>
#include <stdio.h>


/* Maximum size of resampling kernel */
#define KNN_NKERNEL_MAX 50

/* Maximum number of variables to evaluate distance */
#define KNN_NVAR_MAX 20

/* Minimum value of positive weight */
#define KNN_WEIGHT_MIN 1e-30

int c_knn_run(int nparams, int nval, int nvar, int nrand,
    int idx_select,
    double * params,
    double * weights,
    double * var,
    double * rand,
    int * knn_idx);

