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
    double A = sqrt(theta * z / UTILS_PI);
    t = c_max(HAYAMI_TMIN, t);
    double arg = z * (2 - theta / t - t / theta);
    arg = c_minmax(-HAYAMI_EXP_ARGMAX, HAYAMI_EXP_ARGMAX, arg);
    return A * exp(arg) / sqrt(t * t * t);
}


double uh_hayami(double ordinate, double theta, double z, double timestep)
{
    /* Gaussian quadature abscissae and weights */
    double x[5] = {0.1488743389816312,0.4333953941292472,
                  0.6794095682990244,0.8650633666889845,
                  0.9739065285171717};
    double w[5] = {0.2955242247147529,0.2692667193099963,
                   0.2190863625159821,0.1494513491505806,
                   0.0666713443086881};

    /* Boundaries of integration */
    double a = ordinate * timestep;
    double b = (ordinate + 1) * timestep;

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
    int i;
    double u;
    double suh;

    /* UH ordinates */
    *nuh = 0;
    suh = 0;
    for(i = 0; i < nuhlengthmax - 1; i++)
    {
        if(suh < 1 - UHEPS)
            *nuh += 1;
        else
            break;

        /* Integration can be very inaccurate sometimes */
        u = c_min(1., uh_hayami((double)i, theta, z, timestep));
        uh[i] = u;
        suh += u;
    }

    /* NUH is not big enough */
    //if(1 - suh > UHEPS || *nuh > nuhlengthmax)
    //{
    //    fprintf(stdout, "suh=%0.4f\n", suh);
    //    return HAYAMI_ERROR + __LINE__;
    //}

    /* Small correction of first ordinate to remove any bias */
    uh[0] += 1 - suh;

    return 0;
}


int c_hayami_runtimestep(
        int nuh, int ninputs,
        int nstates, int noutputs,
	    double dt,
        double * uh,
        double * inputs,
	    double * statesuh,
        double * states,
        double * outputs)
{
    int k, ierr = 0;
    double qin;
    double vr;

    /* input */
    qin = inputs[0];
    qin = qin < 0 ? 0. : qin;

    /* Hayami uh */
    vr = 0;
    for (k = 0; k < nuh - 1; k++)
    {
        statesuh[k] = statesuh[1 + k] + uh[k] * qin;

        /* Volume in transit in the river reach */
        if(k > 0)
            vr += statesuh[k] * dt;
    }
    statesuh[nuh - 1] = uh[nuh - 1] * qin;

    if(nuh > 1)
        vr += statesuh[nuh - 1] * dt;

    /* flow outputs */
    outputs[0] = statesuh[0];

    /* Storage outputs */
    if(noutputs > 2)
        outputs[2] = vr;
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
    double dt;

    /* Check dimensions */
    if(nconfig < 1)
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

    /* Check parameters */
    ierr = hayami_minmaxparams(nparams, params);

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
       /* Run timestep model and update states */
    	ierr = c_hayami_runtimestep(nuh, ninputs,
                                    nstates, noutputs,
                                    dt, uh,
                                    &(inputs[ninputs*i]),
                                    statesuh,
                                    states,
                                    &(outputs[noutputs*i]));
    }

    return ierr;
}

