#include "c_hayami.h"

double c_hayami_get_uheps() {
    return HAYAMI_UHEPS;
}

int c_hayami_get_maxuh() {
    return HAYAMI_MAXUH;
}

/* Hayami parameters,
 * See Moussa (1996), Equation 10
 */
double c_hayami_compute_theta(double length_ref,
                              double length,
                              double eta,
                              double zeta) {
    /* Assumes eta is counted in days
     * and corresponds to length_ref */
    return eta * 86400. * length / length_ref;
}

double c_hayami_compute_z(double length_ref,
                          double length,
                          double eta,
                          double zeta) {
    /* Assumes zeta corresponds to length_ref */
    return zeta * length / length_ref;
}

double c_hayami_compute_C(double length_ref,
                          double length,
                          double eta,
                          double zeta) {
    double theta = c_hayami_compute_theta(length_ref, length, eta, zeta);
    return length / theta;
}

double c_hayami_compute_D(double length_ref,
                          double length,
                          double eta,
                          double zeta) {
    double C = c_hayami_compute_C(length_ref, length, eta, zeta);
    double z = c_hayami_compute_z(length_ref, length, eta, zeta);
    return C * length / 4 / z;
}

int hayami_minmaxparams(int nparams, double * params)
{
        double pmin, pmax;

        if(nparams < HAYAMI_NPARAMS)
            return HAYAMI_ERROR + __LINE__;

        /* eta (days) */
        pmin = 1e-4;
        pmax = 1e2;
        params[0] = c_minmax(pmin, pmax, params[0]);

        /* zeta (dimless) */
        pmin = 1e-4;
        pmax = 1e3;
        params[1] = c_minmax(pmin, pmax, params[1]);

	return 0;
}


double hayami_kernel(double theta, double z, double t) {
    /*
     * See Moussa, R. (1996).
     * https://doi.org/10.1002/(SICI)1099-1085(199609)10:9%253C1209::AID-HYP380%253E3.0.CO;2-2
     */
    double u = t / theta;
    double A = sqrt(z / UTILS_PI) / theta;
    return A * exp(z * (2 - 1. / u - u)) / sqrt(u * u * u);
}


double hayami_kernel_diff(double theta, double z, double t) {
    double u = t / theta;
    double k = hayami_kernel(theta, z, t);
    return k  / theta * (z * (- 1. + 1. / u / u) - 1.5 / u);
}


double hayami_kernel_diff2(double theta, double z, double t) {
    double u = t / theta;
    double k = hayami_kernel(theta, z, t);
    double z2 = z * z;
    double u2 = u * u;
    double u3 = u2 * u;
    double u4 = u2 * u2;
    return k  / theta / theta * (z2 + z2 / u4 - 5 * z / u3 + (15 - 8 * z2) / 4 / u2 + 3 * z / u);
}


double hayami_kernel_tmax(double theta, double z) {
    return theta * (sqrt(16 * z * z + 9) - 3) / 4 / z;
}


double hayami_kernel_halley(double theta, double z,
                            double fobj, double rtol,
                            double t) {
    int iter, nitermax = 20;
    double t0 = hayami_kernel_tmax(theta, z);

    /* Bounds depend if initial t is below t0 or above */
    double tmin = t < t0 ? t0 * 0.01 : t0 * 1.01;
    double tmax = t < t0 ? t0 * 0.99 : 1e100;

    double err, relerr, relerr_prev = 1e100;
    double f, df, d2f;

    /* Are we below t0 or above ? */
    int below = t < t0;

    t = t < tmin || isnan(t) ? tmin : t;

    for(iter = 0; iter < nitermax; iter++) {
        f = hayami_kernel(theta, z, t);

        /* Early exit */
        err = f - fobj;
        relerr = fabs(err) / fobj;
        if(relerr < rtol || relerr >= relerr_prev || isnan(f))
            break;

        /* Halley's method step */
        df = hayami_kernel_diff(theta, z, t);
        d2f = hayami_kernel_diff2(theta, z, t);
        t -= err * df / (df * df - 0.5 * err * d2f);

        /* Checks */
        t = t < tmin ? tmin : t > tmax ? tmax : t;
        relerr_prev = relerr;
    }

    return t;
}


int hayami_kernel_tbounds(double theta, double z, double eps, double tbounds[2]) {
    double tlow, thigh;
    double t0, f0, df0, d2f0, fobj;
    double a, b, c, sqD;
    double rtol = 1e-3;

    /* This is the value of the kernel we seek: eps x max(kernel) */
    t0 = hayami_kernel_tmax(theta, z);
    f0 = hayami_kernel(theta, z, t0);
    fobj = f0 * eps;

    /* Approximate bounds with taylor series at t=t0 */
    df0 = hayami_kernel_diff(theta, z, t0);
    d2f0 = hayami_kernel_diff2(theta, z, t0);
    a = d2f0 / 2;
    b = df0 - t0 * d2f0;
    c = f0 - df0 * t0 + d2f0 * t0 * t0 / 2;
    sqD = sqrt(b * b - 4 * a * c);
    tlow = (-b + sqD) / 2 / a;
    thigh = (-b - sqD) / 2 / a;

    /* Halley's iteration */
    tbounds[0] = hayami_kernel_halley(theta, z, fobj, rtol, tlow);
    tbounds[1] = hayami_kernel_halley(theta, z, fobj, rtol, thigh);

    return 0;
}


double integrate_hayami_kernel(double a, double b, double theta, double z)
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
                      int niter,
                      double timestep,
                      double theta,
                      double z,
                      int * nuh,
                      double * uh)
{
    int i, j;
    double u, suh;
    double t1, t2;
    double dt, a, b;

    /* Number of subdivisions for integration */
    niter = niter < 1 ? 1 : niter;

    double t0 = hayami_kernel_tmax(theta, z);

    double eps = 1e-3;
    double tbounds[2];
    hayami_kernel_tbounds(theta, z, eps, tbounds);
    double tlow = tbounds[0];
    double thigh = tbounds[1];

    /* UH ordinates */
    *nuh = 0;
    suh = 0;

    for(i = 0; i < nuhlengthmax - 1; i++)
    {
        /* timestep time bounds */
        t1 = timestep * (double)i;
        t2 = t1 + timestep;

        /* Check bounds if kernel maximum is in interval */
        if(i == 0) {
            t1 = t1 < tlow ? tlow : t1;
            t2 = t2 > thigh ? thigh : t2;
        }

        /* Compute */
        dt = (t2 - t1) / (double) niter;
        u = 0;
        for(j = 0; j < niter; j++) {
            a = t1 + (double)j * dt;
            b = a + dt;
            u += integrate_hayami_kernel(a, b, theta, z);
        }
        u = u > 1. ? 1. : u;

        uh[i] = u;
        suh += u;
        *nuh += 1;

        /* Break loop if we have reached accumulated
         * ordinates close to 1. Also break loop if we
         * passed the peak of the kernel and we get very
         * low uh ordinates.
         */
        if(suh > 1 - HAYAMI_UHEPS || (t1 > t0 && u < HAYAMI_UHEPS))
            break;
   }

    /* NUH is not big enough */
    if(fabs(1 - suh) > HAYAMI_UHEPS)
    {
        /* set to nan if error is too large */
        suh = suh > 1.2 || suh < 0.8 ? c_get_nan() : suh;

        for(i = 0; i < *nuh - 1; i++)
            uh[i] /= suh;
    }

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
    double qin, phi, qconvol;
    double vr0, vr;

    /* Parameters */

    /* input */
    qin = inputs[0];
    qin = qin < 0 ? 0. : qin;

    /* if lateral flow, use cumulative inflow
     * See Equation 49 in Moussa (1996)
     * states[0] = cumulative flow from prior timesteps
     * Here, we divide the cumulative flow by theta
     * to compute the Phi function from Moussa (1996).
     * */
    phi = (states[0] + qin) * dt / theta;
    qconvol = qin - is_lateral * phi;

    /* Hayami uh */
    for (k = 0; k < nuh - 1; k++)
        statesuh[k] = statesuh[1 + k] + uh[k] * qconvol;

    statesuh[nuh - 1] = uh[nuh - 1] * qconvol;

    /* flow outputs */
    /* See Equation 49 in Moussa (1996) */
    outputs[0] = statesuh[0] + is_lateral * phi;

    /* Transiting volume */
    vr0 = states[1];
    vr = (qin - outputs[0]) * dt + vr0;

    /* States -> cumulative flow and stored volume */
    states[0] += qin;
    states[1] = vr;

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
    double dt, lateral, eta, theta, zeta;
    double length, length_ref;

    /* Check dimensions */
    if(nparams < HAYAMI_NPARAMS)
        return HAYAMI_ERROR + __LINE__;

    if(nconfig < HAYAMI_NCONFIG)
        return HAYAMI_ERROR + __LINE__;

    if(nstates < HAYAMI_NSTATES)
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
    length_ref = config[1];
    length = config[2];
    lateral = config[3];

    /* Check parameters */
    ierr = hayami_minmaxparams(nparams, params);

    /* theta parameter is obtained by multiplying eta by timestep
     * duration */
    eta = params[0];
    zeta = params[1];
    theta = c_hayami_compute_theta(length_ref, length, eta, zeta);

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

