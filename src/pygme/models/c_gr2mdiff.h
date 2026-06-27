#ifndef __GR2MDIFF__
#define __GR2MDIFF__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"
#include "c_gr2m.h"

/* Number of config required by GR2MDIFF run */
#define GR2MDIFF_NCONFIG 1

/* Number of inputs required by GR2MDIFF run */
#define GR2MDIFF_NINPUTS 2

/* Number of params required by GR2MDIFF run */
#define GR2MDIFF_NPARAMS 2

/* Number of states returned by GR2MDIFF run */
#define GR2MDIFF_NSTATES 2

/* Number of outputs returned by GR2MDIFF run */
#define GR2MDIFF_NOUTPUTS 12
#define GR2MDIFF_NDOUTPUTS 2

#define GR2MDIFF_NDOT 2

struct dn_double {
    double val;
    double dot[GR2MDIFF_NDOT];
};
typedef struct dn_double dn_double;

dn_double dn_init(double val);
dn_double dn_add(dn_double dn1, dn_double dn2);
dn_double dn_prod(dn_double dn1, dn_double dn2);
dn_double dn_div(dn_double dn1, dn_double dn2);
dn_double dn_intpow(dn_double dn1, int n);
dn_double dn_tanh(dn_double dn1);
dn_double dn_cbrt(dn_double dn1);


int c_gr2mdiff_run(int nval, int nconfig, int nparams, int ninputs,
    int nstates, int noutputs, int ndoutputs,
    int start, int  end,
	double * config,
	double * params,
	double * dparams,
	double * inputs,
	double * statesini,
    double * outputs,
    double * doutputs);

#endif
