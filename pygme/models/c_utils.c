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
    int days_in_month[13] = {0, 31, 28, 31, 30, 31, 30,
                    31, 31, 30, 31, 30, 31};

    if(month < 1 || month > 12)
        return -1;

	n = days_in_month[month];

    return c_utils_isleapyear(year) == 1 && month == 2 ? n+1 : n;
}


int c_utils_dayofyear(int month, int day)
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
    double CS, I1, I2, WY;

    ierr = c_utils_getdate(start, date);
    if(ierr < 0)
        return UTILS_ERROR + __LINE__;

    CS = 0;
    I2 = 0;
    WY = date[0];

    for(i=0; i<nval; i++)
    {
        if(date[1] == year_monthstart && date[2] == 1){
            CS = 0;
            WY += 1;
        }

        ierr = c_utils_add1day(date);

        I1 = inputs[i];
        if(!isnan(I1))
            I2 = I1;

        CS += I2;
        outputs[2*i] = CS;
        outputs[2*i+1] = WY;
    }

    return 0;
}

/* This is a simplistic root finder for function fun */
int c_utils_root_square(double (*fun)(double, int, double *),
        int *niter, int *status, double eps,
        double * roots,
        int nargs, double * args){

    int i, nitermax;
    double values[3];
    double a, b, c, D, E, F, G, H, I, J, x0;
    double dx, dx0, df, df0;

    *niter = 0;
    *status = 0;

    /* Maximum number of iteration */
    nitermax = 20;

    /* Check roots are increating */
    if(roots[1]<roots[0] || roots[2]<roots[1])
        return UTILS_ERROR + __LINE__;

    df0 = 1e30;
    for(i=0; i<3; i++){
        values[i]= fun(roots[i], nargs, args);

        /* Computes reference for convergence test */
        df = fabs(values[i]);
        if(df<eps){
            roots[1] = roots[i];
            return 0;
        }

        if(df < df0)
            df0 = df*eps;
    }

    /* Check roots are bracketing 0 */
    if(values[0]*values[2]>=0)
            return UTILS_ERROR + __LINE__;

    /* Computes reference for convergence test */
    dx0 = fabs(roots[1]-roots[0]);
    if(fabs(roots[2]-roots[1]) < dx0)
        dx0 = fabs(roots[2]-roots[1]);
    dx0 *= eps;

    /* Convergence loop */
    while(*niter < nitermax){

        /* Check convergence */
        if(df < df0 || df<eps){
            *status = 2;
            return 0;
        }

        dx = fabs(roots[0]-roots[2]);
        if(dx < dx0 || dx<eps){
            *status = 3;
            return 0;
        }


        /* Square interpolation
        * g(x) = (x-b)*(x-c) * f(a)/(a-b)/(a-c)
                 (x-a)*(x-c) * f(b)/(b-a)/(b-c)
        *        (x-a)(x-b) * f(c)/(c-a)/(c-b)
        * As a result:
        *   g(a) = f(a)
        *   g(b) = f(b)
        *   g(c) = f(c)
        *
        * with D = f(a)/(a-b)/(a-c)
        *      E = f(b)/(b-a)/(b-c)
        *      F = f(c)/(c-a)/(c-b)
        * we have
        * g(x) = D*b*c + E*a*c + F*a*b
        *        - [D*(b+c)+E*(a+c)+F*(a+b)]*x
        *        + (D+E+F)*x^2
        *
        * we pose
        * G = D*b*c + E*a*c + F*a*b
        * H = D*(b+c)+E*(a+c)+F*(a+b)
        * I = D+E+F
        *
        * Finally g(x0) = 0 leads to
        *  J = H^2-4*G*I
        *  x0 = (H+sqrt(J))/2I
        */

        a = roots[0];
        b = roots[1];
        c = roots[2];

        D = values[0]/(a-b)/(a-c);
        E = values[1]/(b-a)/(b-c);
        F = values[2]/(c-b)/(c-a);

        G = D*b*c+E*a*c+F*a*b;
        H = D*(b+c)+E*(a+c)+F*(a+b);
        I = D+E+F;

        J = H*H-4*G*I;
        if(J<0){

            fprintf(stdout, "\n\nIter [%d]:\n", *niter);
            for(i=0; i<3; i++)
                fprintf(stdout, "f(%f) = %0.10f\n", roots[i], values[i]);

            return UTILS_ERROR + __LINE__;
        }

        /* iteration */
        x0 = (H+sqrt(J))/2/I;
        if(roots[1]<x0)
            roots[0] = roots[1];
        else
            roots[2] = roots[1];

        roots[1] = x0;

        *niter = *niter + 1;

        df = 1e30;
        for(i=0; i<3; i++){
            values[i]= fun(roots[i], nargs, args);

            if(fabs(values[i]) < df)
                df = fabs(values[i])*eps;
        }
    }

    *status = 1;

    return 0;
}


double funtest1(double x, int nargs, double * args){
    double a, b, c, y;
    a = args[0];
    b = args[1];
    c = args[2];
    y = -a+x-x/pow(1+pow(x/b, c), 1./c);
    return y;
}

int c_utils_root_square_test(int ntest, int *niter, int *status,
        double eps,
        double * roots, int nargs, double * args){

    if(ntest == 1)
        return c_utils_root_square(funtest1, niter,
            status, eps, roots, nargs, args);
    else
        return UTILS_ERROR + __LINE__;

    return 0;
}
