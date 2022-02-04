#include "c_utils.h"


double c_minmax(double min, double max, double input)
{
    return input < min ? min :
            input > max ? max : input;
}

double c_min(double x, double x0)
{
    return x<x0 ? x : x0;
}

double c_max(double x, double x0)
{
    return x<x0 ? x0 : x;
}

double c_tanh(double x)
{
    double a, b, xsq;
    x = x > 4.9 ? 4.9 : x;
    xsq = x*x;
    a = (((36.*xsq+6930.)*xsq+270270.)*xsq+2027025.)*x;
    b = (((xsq+630.)*xsq+51975.)*xsq+945945.)*xsq+2027025.;
    return a/b;
}


int c_isleapyear(int year)
{
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

int c_daysinmonth(int year, int month)
{
    int n;
    int days_in_month[13] = {0, 31, 28, 31, 30, 31, 30,
                    31, 31, 30, 31, 30, 31};

    if(month < 1 || month > 12)
    {
        return -1;
    }

	n = days_in_month[month];

    return c_isleapyear(year) == 1 && month == 2 ? n+1 : n;
}


int c_dayofyear(int month, int day)
{
    int day_of_year[13] = {0, 0, 31, 59, 90, 120,
                        151, 181, 212, 243, 273, 304, 334};

    if(month < 1 || month > 12)
        return -1;

    if(day < 1 || day > 31)
        return -1;

    /* No need to take leap years into account. This confuses other algorithms */

    return day_of_year[month] + day;
}


int c_add1month(int * date)
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
    nbday = c_daysinmonth(date[0], date[1]);
    if(nbday < 0)
        return UTILS_ERROR + __LINE__;

    if(date[2] > nbday)
        date[2] = nbday;

   return 0;
}

int c_add1day(int * date)
{
    int nbday;
    nbday = c_daysinmonth(date[0], date[1]);

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

int c_getdate(double day, int * date)
{
    int year, month, nday, nbday;

    year = (int)(day * 1e-4);
    month = (int)(day * 1e-2) - year * 100;
    nday = (int)(day) - year * 10000 - month * 100;

    if(month < 0 || month > 12)
        return UTILS_ERROR + __LINE__;

    nbday = c_daysinmonth(year, month);
    if(nday < 0 || nday > nbday)
        return UTILS_ERROR + __LINE__;

    date[0] = year;
    date[1] = month;
    date[2] = nday;

    return 0;
}

/* The accumulate function compute the cumulative sum
    of input vector with a reset at the beginning of
    each water year. The function also computes the water year
    and the day of the water year */
int c_accumulate(int nval, double start,
        int year_monthstart,
        double * inputs, double * outputs)
{
    int i, ierr, date[3];
    double CS, I1, I2, WY, DOY;

    ierr = c_getdate(start, date);
    if(ierr < 0)
        return UTILS_ERROR + __LINE__;

    I2 = 0;
    /* Accumulated value */
    CS = 0;
    /* Water year */
    WY = date[0];
    /* Day of year */
    DOY = NAN;

    for(i=0; i<nval; i++)
    {
        DOY += 1;

        if(date[1] == year_monthstart && date[2] == 1)
        {
            CS = 0;
            WY += 1;
            DOY = 1;
        }

        /* Skip day of year for leap years so that all years have
            a max doy of 365 */
        if(date[1] == 2 && date[2] == 29)
            DOY -= 1;

        ierr = c_add1day(date);

        I1 = inputs[i];
        if(!isnan(I1))
            I2 = I1;

        CS += I2;
        outputs[3*i] = CS;
        outputs[3*i+1] = WY;
        outputs[3*i+2] = DOY;
    }

    return 0;
}

/* ************************
 * Bisection root finding algorithm.
 * See https://en.wikipedia.org/wiki/Root-finding_algorithm#Bisection_method
 *
 * fun : function to find the root from. Function signature must be
 *          double fun(double x, int nargs, double * args)
 *
 * niter : number of iterations
 *
 * status : convergence status
 *     -1 = Error: Function values for roots are not bracketing zero
 *     -2 = Error: Function value is NaN
 *     -3 = Error: No convergence after nitermax iterations
 *
 *      0 = Nothing done. One of the initial points is a root
 *      1 = Convergence achieved after nitermax iterations
 *      2 = Convergence achived, stopped algorithm because
 *              function does not change
 *      3 = Convergence achieved, stopped algorithm because
 *              parameters do no change
 *
 * nitermax : maximum number of iterations
 *
 * eps : Tolerance threshold for parameter and function changes
 *
 * roots : initial and final values (3 values, the root is the middle one)
 *
 * nargs : number of function arguments
 *
 * args : function arguments
 *
 * */
int c_rootfind(double (*fun)(double, int, double *),
        int *niter, int *status, double epsx, double epsfun,
        int nitermax, double *roots, int nargs, double * args)
{
    int iroot;
    double tmp, values[2];
    double x0, f0, dx, maxf, fa, fb;

    *niter = 0;
    *status = 0;

    /* Sort initial roots */
    if(roots[1]<roots[0])
    {
        tmp = roots[0];
        roots[0] = roots[1];
        roots[1] = tmp;
    }

    /* Find largest function value */
    values[0] = fun(roots[0], nargs, args);
    values[1] = fun(roots[1], nargs, args);

    fa = values[0];
    fb = values[1];
    maxf = fabs(fa) < fabs(fb) ? fabs(fb) : fabs(fa);

    /* Root already found, nothing to do */
    if(maxf < epsfun)
        return 0;

    /* Check roots are bracketing 0 */
    if(values[0]*values[1]>=0)
    {
        *status = -1;
        fprintf(stdout, "\n\nRoots not bracketing 0: f(%f)=%f and f(%f)=%f\n",
                    roots[0], values[0], roots[1], values[1]);
        return UTILS_ERROR + __LINE__;
    }

    /* Convergence loop */
    while(*niter < nitermax)
    {
        /* x range */
        dx = fabs(roots[0]-roots[1]);

        /* Check convergence of fun */
        if(maxf < epsfun)
            *status = 2;

        /* Check convergence of parameters */
        if(dx < epsx)
            *status = 3;

        /* Continues if no convergence reached */
        if(*status == 0)
        {
            /* Compute mid-interval values */
            x0 = (roots[0]+roots[1])/2;
            f0 = fun(x0, nargs, args);

            if(isnan(f0))
            {
                *status = -2;
                fprintf(stdout, "\n\nIter [%d]: f(x0) is NaN\n", *niter);
                return UTILS_ERROR + __LINE__;
            }

            /* Choose which values to replace depending on the sign of f0 */
            iroot = f0*values[0] > 0 ? 0 : 1;

            /* Replace value */
            roots[iroot] = x0;
            values[iroot] = f0;

            /* Iterate */
            maxf = fabs(f0) < maxf ? fabs(f0) : maxf;
            *niter = *niter + 1;

        } else
            break;
    }

    /* One more step to ensure convergence */
    x0 = (roots[0]+roots[1])/2;
    f0 = fun(x0, nargs, args);
    iroot = f0*values[0] > 0 ? 0 : 1;
    roots[iroot] = x0;
    values[iroot] = f0;

    /* Sort roots to ensure best root is the first one */
    if(fabs(values[0]) > fabs(values[1]))
    {
        tmp = roots[0];
        roots[0] = roots[1];
        roots[1] = tmp;
    }

    /* check final convergence */
    if(maxf > epsfun)
        *status = -3;
    else
        *status = 1;

    return 0;
}


/*
 * Root finding function to be used in the test
 * y = -a+x-x/(1+(x/b)^c)^(1/c)
 */
double funtest1(double x, int nargs, double * args)
{
    double a, b, c, y;
    a = args[0];
    b = args[1];
    c = args[2];
    y = -a+x-x/pow(1+pow(x/b, c), 1./c);
    return y;
}

/*
 * Testing root finding algorithm
 */
int c_rootfind_test(int ntest, int *niter, int *status,
        double epsx, double epsfun, int nitermax,
        double * roots, int nargs, double * args)
{
    if(ntest == 1)
        return c_rootfind(funtest1, niter,
            status, epsx, epsfun, nitermax, roots, nargs, args);
    else
        return UTILS_ERROR + __LINE__;

    return 0;
}


