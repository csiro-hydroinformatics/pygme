
#ifndef __UTILS__
#define __UTILS__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message */
#define UTILS_ERROR 10000
#define BASICS_ERROR 11000
#define UH_ERROR 12000

#define GR2M_ERROR 20000
#define GR4J_ERROR 21000
#define GR6J_ERROR 22000
#define HBV_ERROR 23000

#define LAGROUTE_ERROR 30000

#define KNNDAILY_ERROR 40000

#define RIVERMODELS_ERROR 50000

#define UTILS_PI 3.14159265358979

/* utility functions */
double c_minmax(double min,double max,double input);


double c_tanh(double x);


int c_isleapyear(int year);


int c_daysinmonth(int year, int month);


int c_dayofyear(int month, int day);


int c_add1month(int * date);


int c_add1day(int * date);


int c_getdate(double day, int * date);


int c_accumulate(int nval, double start,
        int year_monthstart,
        double * inputs, double * outputs);


int c_rootfind(double (*fun)(double, int, double *),
        int *niter, int * status, double epsx, double epsfun, int nitermax,
        double * roots, int nargs, double * args);


int c_rootfind_test(int ntest, int *niter, int *status,
        double epsx, double epsfun, int nitermax, double * roots,
        int nargs, double * args);

#endif
