#include "c_utils.h"


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


int c_utils_add1month(int * date)
{
    int nbday;

    /* change month */
    if(date[1] < 12)
    {
        date[1] += 1;
    }
    else
    {
        /* change year */
        date[1] = 1;
        date[0] += 1;
    }

    /* Check that day is not greater than
     * number of days in month */
    nbday = c_utils_daysinmonth(date[0], date[1]);
    if(nbday < 0)
        return UTILS_ERROR + __LINE__;

    if(date[2] > nbday)
        date[2] = nbday;

   return 0;
}

int c_utils_add1day(int * date)
{
    int nbday;
    nbday = c_utils_daysinmonth(date[0], date[1]);

    if(nbday < 0)
        return UTILS_ERROR + __LINE__;

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
        return UTILS_ERROR + __LINE__;
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
        return UTILS_ERROR + __LINE__;

    nbday = c_utils_daysinmonth(year, month);
    if(nday < 0 || nday > nbday)
        return UTILS_ERROR + __LINE__;

    date[0] = year;
    date[1] = month;
    date[2] = nday;

    return 0;
}

int c_utils_accumulate(int nval, double start,
        int year_monthstart,
        double * inputs, double * outputs)
{
    int i, ierr, date[3];
    double CS, I1, I2;

    ierr = c_utils_getdate(start, date);
    if(ierr < 0)
        return UTILS_ERROR + __LINE__;

    CS = 0;
    I2 = 0;
    for(i=0; i<nval; i++)
    {
        if(date[1] == year_monthstart && date[2] == 1)
            CS = 0;

        ierr = c_utils_add1day(date);

        I1 = inputs[i];
        if(!isnan(I1))
            I2 = I1;

        CS += I2;
        outputs[i] = CS;
    }

    return 0;
}


int c_utils_root_square(double (*fun)(double, int, *double),
        int *niter, int *status,
        double * roots,
        int nargs, double * args){

    int i, nitermax;
    double values[3];
    double ca, cb, D, E, F, H, x0, eps, df;

    *niter = 0;
    *status = 0;
    nitermax = 100;
    eps = 1e-7;

    /* Check roots are bracketing 0 */
    for(i=0; i<3; i++)
        values[i]= fun(roots[i], nargs, args);

    if(values[0]*values[2]>0)
            return UTILS_ERROR + __LINE__;

    df = fabs(values[2]-values[0]);

    /* Convergence loop */
    while(*niter < nitermax){

        for(i=0; i<3; i++)
            values[i]= fun(roots[i], nargs, args);

        if(fabs(values[2]-values[0]) < eps * df){
            *status = 1;
            return 0;
        }

        /* Square interpolation
        * g(x) = f(a)(x-b) + f(b)(x-a) +
        *           (x-a)(x-b) * [f(c)-f(a)*(c-b) - f(b)*(c-a)]/(c-a)/(c-b)
        * As a result:
        *   g(a) = f(a)
        *   g(b) = f(b)
        *   g(c) = f(c)
        *
        * with D = f(c)/(c-a)/(c-b) + f(a)/(c-a) + f(b)/(c-b)
        * we have
        * g(x) = a*b*D-f(a)*b-f(b)*a + [f(a)+f(b)-(a+b)*D] * x + D * x^2
        *      = E + F x + D x^2
        *
        * Finally g(x0) = 0 leads to
        *  H = F^2-4*E*D
        *  x0 = (-F+sqrt(H))/2D
        */

        ca = roots[2]-roots[0];
        cb = roots[2]-roots[1];

        D = values[2]/ca/cb+values[0]/ca+values[1]/cb;
        E = roots[0]*roots[1]*D-values[0]*roots[1]-values[1]*roots[0];
        F = values[0]+values[1]-(roots[0]+roots[1])*D;

        H = F*F-4*E*D;
        if(H<0)
            return UTILS_ERROR + __LINE__;

        /* iteration */
        x0 = (-F+sqrt(H))/2/D;
        if(roots[1]<x0)
            roots[0] = roots[1];
        else
            roots[2] = roots[1];

        roots[1] = x0;

        *niter++;
    }

    *status = 2;

    return 0;
}


double funtest1(double x, int nargs, double * args){
    double a, b, y, x4;
    a = args[0];
    b = args[1];
    x4 = (x/a)*(x/a);
    x4 = x4*x4;
    y = -b+x-x/sqrt(sqrt(1+x4));
    return y;
}

int c_utils_root_square_test(int ntest, int *niter, int *status,
        double * roots, int nargs, double * args){

    if(ntest == 0)
        return c_utils_root_square(funtest1, niter,
            status, roots, nargs, args);
}
