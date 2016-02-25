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

	n = days_in_month[month];

    return c_utils_isleapyear(year) == 1 && month == 2 ? n+1 : n;
}

int c_utils_add1day(int date[3])
{
    int nbday;

    nbday = c_utils_dayinmonth(year, month);
    if(date[2] < nbday)
    {
        date[2] += 1;
        return;
    }
    else {
        /* change month */
        date[2] = 1;

        if(date[1] < 12)
        {
            date[1] += 1;
            return;
        }
        else
        {
            /* change year */
            date[1] = 1;
            date[0] += 1;
            return;
        }
    }

    return 0;
}

