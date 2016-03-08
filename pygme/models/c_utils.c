#include "c_utils.h"


int c_utils_geterror(int * error)
{
    error[0] = ESIZE_INPUTS;
    error[1] = ESIZE_OUTPUTS;
    error[2] = ESIZE_PARAMS;
    error[3] = ESIZE_STATES;
    error[4] = ESIZE_STATESUH;
    error[5] = ESIZE_STATESUH;

    error[10] = EINVAL;
    error[11] = EMODEL_RUN;

    return 0;
}

double c_utils_minmax(double min, double max, double input)
{
    return input < min ? min :
            input > max ? max : input;
}

double c_utils_tanh(double x)
{
    double a, b, xsq;
    x = x > 4.9 ? 4.9 : x;
    xsq = x*x;
    a = (((36.*xsq+6930.)*xsq+270270.)*xsq+2027025.)*x;
    b = (((xsq+630.)*xsq+51975.)*xsq+945945.)*xsq+2027025.;
    return a/b;
}


int c_utils_isleapyear(int year)
{
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

int c_utils_daysinmonth(int year, int month)
{
    int n;
    int days_in_month[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

    if(month < 1 || month > 12)
        return -1;

	n = days_in_month[month];

    return c_utils_isleapyear(year) == 1 && month == 2 ? n+1 : n;
}


int c_utils_dayofyear(int month, int day)
{
    int n;
    int day_of_year[13] = {0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};

    if(month < 1 || month > 12)
        return -1;

    if(day < 1 || day > 31)
        return -1;

    /* No need to take leap years into account. This confuses other algorithms */

    return day_of_year[month] + day;
}


int c_utils_add1day(int * date)
{
    int nbday;

    nbday = c_utils_daysinmonth(date[0], date[1]);
    if(nbday < 0)
        return 6000+__LINE__;

    if(date[2] < nbday)
    {
        date[2] += 1;
        return 0;
    }
    else if(date[2] == nbday) {
        /* change month */
        date[2] = 1;

        if(date[1] < 12)
        {
            date[1] += 1;
            return 0;
        }
        else
        {
            /* change year */
            date[1] = 1;
            date[0] += 1;
            return 0;
        }
    }
    else {
        return 2000 + __LINE__;
    }

    return 0;
}

int c_utils_getdate(double day, int * date)
{
    int year, month, nday, nbday;

    year = (int)(day * 1e-4);
    month = (int)(day * 1e-2) - year * 100;
    nday = (int)(day) - year * 10000 - month * 100;

    if(month < 0 || month > 12)
        return 2000 + __LINE__;

    nbday = c_utils_daysinmonth(year, month);
    if(nday < 0 || nday > nbday)
        return 2000 + __LINE__;

    date[0] = year;
    date[1] = month;
    date[2] = nday;

    return 0;
}
