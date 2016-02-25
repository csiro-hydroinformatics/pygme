
#ifndef __UTILS__
#define __UTILS__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message for vector size errors */
#define ESIZE_OUTPUTS 5000+__LINE__
#define ESIZE_INPUTS 5000+__LINE__
#define ESIZE_PARAMS 5000+__LINE__
#define ESIZE_STATES 5000+__LINE__
#define ESIZE_STATESUH 5000+__LINE__
#define ESIZE_CONFIG 5000+__LINE__

#define EMODEL_RUN 5000+__LINE__

int c_utils_getesize(int * esize);

double c_utils_minmax(double min,double max,double input);

double c_utils_tanh(double x);

int c_utils_isleapyear(int year);

int c_utils_daysinmonth(int year, int month);

int c_utils_add1day(int date[3]);

#endif
