#ifndef __AUTODIFF__
#define __AUTODIFF__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#define AUTODIFF_NDIFFMAX 10

typedef struct  {
    int ndiff;
    double value;
    double diff[AUTODIFF_NDIFFMAX];
} DualNumber;

DualNumber dualnumber_initialise(int ndiff, double value, double diff);

DualNumber dualnumber_add(DualNumber dn1, DualNumber dn2);
DualNumber dualnumber_diff(DualNumber dn1, DualNumber dn2);
DualNumber dualnumber_mult(DualNumber dn1, DualNumber dn2);
DualNumber dualnumber_div(DualNumber dn1, DualNumber dn2);

DualNumber dualnumber_exp(DualNumber dn1);
DualNumber dualnumber_log(DualNumber dn1);
DualNumber dualnumber_sqrt(DualNumber dn1);

#endif
