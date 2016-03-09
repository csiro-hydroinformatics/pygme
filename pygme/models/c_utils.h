
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
#define GR2M_ERROR 13000
#define GR4J_ERROR 14000
#define LAGROUTE_ERROR 15000
#define KNNDAILY_ERROR 16000
#define RIVERMODELS_ERROR 17000

/* utility functions */
double c_utils_minmax(double min,double max,double input);

double c_utils_tanh(double x);

int c_utils_isleapyear(int year);

int c_utils_daysinmonth(int year, int month);

int c_utils_dayofyear(int month, int day);

int c_utils_add1day(int * date);

int c_utils_getdate(double day, int * date);

#endif
