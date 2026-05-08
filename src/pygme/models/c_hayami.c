#include "c_hayami.h"

int c_hayami_get_maxuh() {
    return HAYAMI_MAXUH;
}


int hayami_minmaxparams(int nparams, double * params)
{
        double p1, p2;

        if(nparams<2)
            return HAYAMI_ERROR + __LINE__;

        p1 = params[0];
        params[0] = p1 < 1e-2 ? 1e-2 :
            p1 > 20 ? 20 : p1;

        p2 = params[1];
        params[1] = p2 < 0. ? 0. :
            p2 > 1. ? 1. : p2;

	return 0;
}


double hayami_kernel(double theta, double z, double t) {
    /*
     * See Moussa, R. (1996).
     * https://doi.org/10.1002/(SICI)1099-1085(199609)10:9%253C1209::AID-HYP380%253E3.0.CO;2-2
     */
    double u = t / theta;
    double A = sqrt(z / theta / theta / UTILS_PI);
    return A * exp(z * (2 - 1. / u - u)) / sqrt(u * u * u);
}


double uh_hayami(double a, double b, double theta, double z)
{
    if(b <= a)
        return 0;

    /* Gaussian quadature abscissae and weights */
    double x[5] = {0.1488743389816312,0.4333953941292472,
                  0.6794095682990244,0.8650633666889845,
                  0.9739065285171717};
    double w[5] = {0.2955242247147529,0.2692667193099963,
                   0.2190863625159821,0.1494513491505806,
                   0.0666713443086881};

    /* Initialise */
    double mid = 0.5 * (b + a);
    double rg = 0.5 * (b - a);
    double s = 0;

    /* 10 pt Gaussian Quadrature */
    for (int j = 0; j < 5; j++) {
        double dx = rg * x[j];
        s += w[j] * (hayami_kernel(theta, z, mid + dx) + hayami_kernel(theta, z, mid - dx));
    }
    return s *= rg;
}


int c_uh_getuh_hayami(int nuhlengthmax,
                      double timestep,
                      double theta,
                      double z,
                      int * nuh,
                      double * uh)
{
    int i, j;
    double du, u, suh;
    double t0, t1;

    /* Define an ordinate below which the bulk of the integral
     * is. This is defined as M1 + sqrt(M2) with M1 and M2
     * the first and second moment of the kernel.
     * M1 = theta
     * M2 = theta^2 (1 + 1/2z)
     * See Moussa (1996)
     */
    double tmax = theta * (1 + sqrt(1 + 1. / 2 / c_max(1e-100, z)));

    int ndiv = 2;
    double a[2];
    double b[2];

    /* UH ordinates */
    *nuh = 0;
    suh = 0;
    for(i = 0; i < nuhlengthmax - 1; i++)
    {
        if(suh < 1 - UHEPS)
            *nuh += 1;
        else
            break;

        /* Split interval if kernel concentrated towards 0 */
        t0 = timestep * (double)i;
        t1 = t0 + timestep;

        a[0] = t0;
        b[0] = i == 0 ? c_min(tmax, t1) : t1;

        a[1] = b[0];
        b[1] = t1;

        /* Compute kernel for each subdivision */
        u = 0;
        for(j = 0; j < ndiv; j++) {
            du = uh_hayami(a[j], b[j], theta, z);
            u += du;
            if(u > 1) {
                u = 1;
                break;
            }
            fprintf(stdout, "a=%0.2e b=%0.2e du=%0.2e u=%0.2e\n",
                    a[j], b[j], du, u);
        }

        uh[i] = u;
        suh += u;
    }

    /* NUH is not big enough */
    if(1 - suh > UHEPS)
    {
        *nuh = nuhlengthmax - 1;
        for(i = 0; i < nuhlengthmax - 1; i++)
            uh[i] /= suh;
    }

    /* Small correction of first ordinate to remove any bias */
    //uh[0] += 1 - suh;

    return 0;
}


int c_hayami_runtimestep(
        int nuh, int ninputs,
        int nstates, int noutputs,
	    double dt, double lateral,
        double theta,
        double * uh,
        double * inputs,
	    double * statesuh,
        double * states,
        double * outputs)
{
    int k, ierr = 0;
    double is_lateral = lateral > 0 ? 1. : 0.;
    double qin, qsum, qconvol, qout;
    double vr;

    /* Parameters */

    /* input */
    qin = inputs[0];
    qin = qin < 0 ? 0. : qin;

    /* if lateral flow, use cumulative inflow
     * See Equation 49 in Moussa (1996)
     * */
    qsum = (states[0] + qin) * dt / theta;
    qconvol = qin - is_lateral * qsum;

    /* Hayami uh */
    vr = 0;
    for (k = 0; k < nuh - 1; k++)
    {
        statesuh[k] = statesuh[1 + k] + uh[k] * qconvol;

        /* Volume in transit in the river reach */
        if(k > 0)
            vr += statesuh[k] * dt;
    }
    statesuh[nuh - 1] = uh[nuh - 1] * qconvol;

    if(nuh > 1)
        vr += statesuh[nuh - 1] * dt;

    /* States */
    states[0] += qin;

    /* flow outputs */
    /* See Equation 49 in Moussa (1996) */
    outputs[0] = statesuh[0] + is_lateral * qsum;

    /* Storage outputs */
    if(noutputs > 1)
        outputs[1] = vr;
    else
        return ierr;

    return ierr;
}

// --------- Component runner --------------------------------------------------
int c_hayami_run(int nval,
        int nparams,
        int nuh,
        int ninputs,
        int nconfig,
        int nstates,
        int noutputs,
        int start, int end,
        double * config,
        double * params,
        double * uh,
        double * inputs,
        double * statesuh,
        double * states,
        double * outputs)
{
    int ierr=0, i;
    double dt, lateral, theta;

    /* Check dimensions */
    if(nparams < 2)
        return HAYAMI_ERROR + __LINE__;

    if(nconfig < 2)
        return HAYAMI_ERROR + __LINE__;

    if(nstates < 1)
        return HAYAMI_ERROR + __LINE__;

    if(nuh > HAYAMI_MAXUH || nuh <= 0)
        return HAYAMI_ERROR + __LINE__;

    if(noutputs > HAYAMI_NOUTPUTS)
        return HAYAMI_ERROR + __LINE__;

    if(start < 0)
        return HAYAMI_ERROR + __LINE__;

    if(end >= nval)
        return HAYAMI_ERROR + __LINE__;

    /* Config data */
    dt = config[0];
    dt = dt < 1 ? 1 : dt;

    lateral = config[2];

    /* Check parameters */
    ierr = hayami_minmaxparams(nparams, params);
    theta = params[0];

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
       /* Run timestep model and update states */
    	ierr = c_hayami_runtimestep(nuh, ninputs,
                                    nstates, noutputs,
                                    dt, lateral, theta, uh,
                                    &(inputs[ninputs*i]),
                                    statesuh,
                                    states,
                                    &(outputs[noutputs*i]));
    }

    return ierr;
}

