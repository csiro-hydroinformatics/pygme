#include "c_uh.h"

double uh_gr4j_ss1_daily(double ordinate, double lag)
{
    lag = lag < 0.5 ? 0.5 : lag;
    double s = ordinate < 0 ? 0 :
        ordinate <= lag ? pow(ordinate/lag, 2.5) : 1;

    return s;
}

double uh_gr4j_ss2_daily(double ordinate,double lag)
{
    lag = lag < 0.5 ? 0.5 : lag;
    double s = ordinate <0 ? 0 :
        ordinate <= lag ? 0.5*pow(ordinate/lag, 2.5) :
        ordinate < 2*lag ? 1-0.5*pow(2-ordinate/lag, 2.5) : 1;

    return s;
}

double uh_gr4j_ss1_hourly(double ordinate, double lag)
{
    lag = lag < 0.5 ? 0.5 : lag;
    double s = ordinate < 0 ? 0 :
        ordinate < lag ? pow(ordinate/lag, 1.25) : 1;

    return s;
}

double uh_gr4j_ss2_hourly(double ordinate,double lag)
{
    lag = lag < 0.5 ? 0.5 : lag;
    double s = ordinate <0 ? 0 :
        ordinate < lag ? 0.5*pow(ordinate/lag, 1.25) :
        ordinate < 2*lag ? 1-0.5*pow(2-ordinate/lag, 1.25) : 1;

    return s;
}


double uh_lag(double ordinate, double lag)
{
    double s = ordinate < lag ? 0. :
        ordinate > lag+1 ? 1. :
        ordinate-lag;

    return s;
}


double uh_triangle(double ordinate, double lag)
{
    double u, s;

    /* Prevent lag from being exactly equal to 0 */
    lag = lag<UHEPS ? UHEPS : lag;

    /* compute ordinate difference */
    u = ordinate/lag;
    s = ordinate < 0. ? 0. :
        ordinate < lag ?  u*u/2 :
        ordinate < 2*lag ? 1/2+(1-u)*(1-u)/2 : 1;

    return s;
}

double uh_flat(double ordinate, double lag)
{
    double s;

    /* Prevent lag from being exactly equal to 0 */
    lag = lag<UHEPS ? UHEPS : lag;

    /* compute ordinate difference */
    s = ordinate < lag ?  ordinate/lag : 1;

    return s;
}

double uh_delta(int uhid, double lag, double ordinate)
{
    if(uhid == 1)
        return uh_gr4j_ss1_daily(ordinate+1, lag)
            - uh_gr4j_ss1_daily(ordinate, lag);

    if(uhid == 2)
        return uh_gr4j_ss2_daily(ordinate+1, lag)
            - uh_gr4j_ss2_daily(ordinate, lag);

    if(uhid == 3)
        return uh_gr4j_ss1_hourly(ordinate+1, lag)
            - uh_gr4j_ss1_hourly(ordinate, lag);

    if(uhid == 4)
        return uh_gr4j_ss2_hourly(ordinate+1, lag)
            - uh_gr4j_ss2_hourly(ordinate, lag);

    if(uhid == 5)
        return uh_lag(ordinate+1, lag)
            - uh_lag(ordinate, lag);

    if(uhid == 6)
        return uh_triangle(ordinate+1, lag)
            - uh_triangle(ordinate, lag);

    if(uhid == 7)
        return uh_flat(ordinate+1, lag)
            - uh_flat(ordinate, lag);

    return 0;
}


int c_uh_getnuhmaxlength(void)
{
    return NUHMAXLENGTH;
}

double c_uh_getuheps(void)
{
    return UHEPS;
}


int c_uh_getuh(int nuhlengthmax,
        int uhid,
        double lag,
        int * nuh,
        double * uh)
{
    int i;
    double suh;

    lag = lag < 0 ? 0 : lag;

    /* UH ordinates */
    *nuh = 0;
    suh = 0;
    for(i=0; i < nuhlengthmax-1; i++)
    {
        if(suh < 1-UHEPS)
            *nuh += 1;
        else
            break;

        uh[i] = uh_delta(uhid, lag, (double)i);
        suh += uh[i];
    }

    /* NUH is not big enough */
    if(1-suh > UHEPS || *nuh > nuhlengthmax)
    {
        fprintf(stdout, "suh=%0.4f\n", suh);
        return UH_ERROR + __LINE__;
    }

    /* Small correction of first ordinate to remove any bias */
    uh[0] += 1-suh;

    return 0;
}


int uh_runtimestep(int nuh,
        double input,
        double * uh,
        double * states,
        double * outputs)
{
    int ierr=0, k;

    for (k=0;k<nuh-1;k++)
        states[k] = states[1+k]+uh[k]*input;

    states[nuh-1] = uh[nuh-1]*input;
    *outputs = states[0];

    return ierr;
}


