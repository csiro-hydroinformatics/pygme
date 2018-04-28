#include "c_hbv.h"
#include "c_uh.h"

/*
* The C code in this file is a translation of the fortran code
* provided in the R-cran package TUWmodel. See
* https://cran.r-project.org/web/packages/TUWmodel/index.html
*
* The code refers to the version of the HBV model published by
* Parajka, J., R. Merz, G. Bloeschl (2007) Uncertainty and multiple objective
* calibration in regional water balance modelling: case study in 320 Austrian
* catchments, Hydrological Processes, 21, 435-446, doi:10.1002/hyp.6253.
*
* Julien Lerat, 2018
*/


int hbv_minmaxparams(int nparams, double * params)
{
    if(nparams < HBV_NPARAMS)
    {
        return HBV_ERROR + __LINE__;
    }

	params[0] = c_minmax(0., 1., params[0]);   // LPRAT
	params[1] = c_minmax(0., 600., params[1]); // FC
	params[2] = c_minmax(0., 20., params[2]);  // BETA
	params[3] = c_minmax(0., 2., params[3]);   // K0
	params[4] = c_minmax(2., 30., params[4]);  // K1
	params[5] = c_minmax(30., 250., params[5]);// K2
	params[6] = c_minmax(1., 100., params[6]); // LSUZ
	params[7] = c_minmax(0., 8., params[7]);   // CPERC
	params[8] = c_minmax(0., 30., params[8]);  // BMAX
	params[9] = c_minmax(0., 50., params[9]);  // CROUTE

	return 0;
}


int hbv_soilmoisture(double rain, double etp, double moist,
        double LP, double FC, double BETA, double *prod)
{
    double moistold, xx;
    double dq, dmoist, eta, melt;

    /* No snow melt runoff, melt is set to 0 */
    melt = 0;

    /* soil mositure accounting */
    moistold = moist;
    dq = pow(moistold/FC, BETA)*(rain+melt);
    dq = dq > rain+melt ? rain+melt : dq;

    dmoist = rain+melt-dq;
    moist = moistold+dmoist;

    if(moist > FC)
    {
      dq = (moist-FC)+dq;
      moist = FC;
    }

    /* calculate evapotranspiration */
    if(moist < LP)
    {
        eta = moist*etp/LP;
        eta = eta > etp ? etp : eta;
    } else
    {
        eta = etp;
    }
    eta = eta < 0 ? 0. : eta;

    /* substract eta of soilmoisture */
    xx = moist;
    moist = moist-eta;

    if(moist < 0)
    {
       eta = xx;
       moist = 0.;
    }

    /* Store */
    prod[0] = dq;
    prod[1] = dmoist;
    prod[2] = eta;
    prod[3] = moist;

    return 0;
}


int hbv_respfunc(double dq, double K0, double LSUZ,
        double K1, double K2, double CPERC, double BMAX, double CROUTE,
        double suz, double slz,
        double *resp, int *bql, double *dquh)
{
    int j, bqlh;
    double rat, bq, suzold, slzold, slzin;
    double q0, q1, q2, qg, sum, bql2;

    /* The split rat/1-rat is not implemented because rat=1 */
    rat = 1.0;
    suzold = suz+rat*dq;
    slzold = slz+(1.-rat)*dq;

    slzin = CPERC;

    suzold = suzold < 0 ? 0 : suzold;
    slzold = slzold < 0 ? 0 : slzold;

    /* upper storage */
    if(suzold > LSUZ)
    {
        q0 = (suzold-LSUZ)/K0*exp(-1./K0);
        q0 = q0 < 0 ? 0 : q0;
        q0 = q0 > suzold-LSUZ ? suzold-LSUZ : q0;
    } else {
        q0 = 0.;
    }
    suzold = suzold-q0;

    q1 = -slzin+(slzin+suzold/K1)*exp(-1./K1);
    q1 = q1 < 0 ? 0. : q1;

    suz = suzold-q1-slzin;
    if(suz < 0)
    {
        suz = 0.;
        slzin = suzold;
    }

    /* lower storage */
    q2 = slzin-(slzin-slzold/K2)*exp(-1./K2);
    q2 = q2 < 0 ? 0. : q2;

    slz = slzold-q2+slzin;
    if(slz < 0)
    {
      slz = 0.;
      q2 = slzold+slzin;
    }
    qg = q0+q1+q2;

    /* transformation function */
    if(BMAX-CROUTE*qg > 1.)
    {
        bq = BMAX-CROUTE*qg;
        *bql = (int)bq;
        *bql = *bql > HBV_MAXUH-1 ? HBV_MAXUH-1 : *bql;
        bql2 = (double)(*bql * *bql);
        bqlh = (int)((double)*bql/2);

        sum = 0.;
        for(j=1; j<=*bql; j++)
        {
            if(j <= bqlh)
            {
                dquh[j-1] = ((j-0.5)*4.*qg)/bql2;
            }
            else if (fabs(j-(bqlh+0.5)) < 0.1)
            {
                dquh[j-1] = ((j-0.75) *4.*qg)/bql2;
            }
            else
            {
                dquh[j-1] = ((*bql-j+0.5)*4.*qg)/bql2;
            }
            sum = sum + dquh[j];
         }
    } else {
        *bql = 1;
        dquh[0] = qg;
        sum = qg;
    }

    resp[0] = q0;
    resp[1] = q1;
    resp[2] = q2;
    resp[3] = qg;
    resp[4] = sum;
    resp[5] = suz;
    resp[6] = slz;

    return 0;
}


int hbv_runtimestep(int nparams,
    int ninputs,
    int nstates,
    int noutputs,
    double * params,
    double * inputs,
    double * states,
    double * outputs,
    int * bql,
    double * dquh)
{
    int ierr=0;

    double prod[4], resp[7];
    double rain, etp;
    double LPRAT, FC, BETA, LP, K0, K1, K2, LSUZ;
    double CPERC, BMAX, CROUTE;
    double moist, suz, slz;
    double dq, dmoist, eta;
    double q0, q1, q2, qg, sum;

    /* Parameters */
    LPRAT = params[0];
    FC = params[1];
    BETA = params[2];
    LP = LPRAT*FC;

    K0 = params[3];
    K1 = params[4];
    K2 = params[5];
    LSUZ = params[6];

    CPERC = params[7];
    BMAX = params[8];
    CROUTE = params[9];

    /* inputs */
    rain = inputs[0];
    rain = rain < 0 ? 0 : rain;

    etp = inputs[1];
    etp = etp < 0 ? 0 : etp;

    /* states */
    moist = states[0];
    suz = states[1];
    slz = states[2];

    /* Production */
    ierr = hbv_soilmoisture(rain, etp, moist,
                LP, FC, BETA, prod);

    dq = prod[0];
    dmoist = prod[1];
    eta = prod[2];
    states[0] = prod[3];

    /* Response function */
    ierr = hbv_respfunc(dq, K0, LSUZ, K1, K2, CPERC, BMAX, CROUTE,
                suz, slz, resp, bql, dquh);

    q0 = resp[0];
    q1 = resp[1];
    q2 = resp[2];
    qg = resp[3];
    sum = resp[4];
    states[1] = resp[5];
    states[2] = resp[6];

    /* RESULTS
    * Skip outputs[0] and outputs[1] because the convolution
    * is done in the run routine due to variable
    * length UH
    */
    if(noutputs>2)
        outputs[2] = dq;
    else
	return ierr;

    if(noutputs>3)
	    outputs[3] = dmoist;
    else
	    return ierr;

    if(noutputs>4)
	    outputs[4] = eta;
    else
	    return ierr;

    if(noutputs>5)
        outputs[5] = q0;
    else
        return ierr;

    if(noutputs>6)
        outputs[6] = q1;
    else
        return ierr;

    if(noutputs>7)
	    outputs[7] = q2;
    else
	    return ierr;

    if(noutputs>8)
	    outputs[8] = qg;
	else
		return ierr;

    if(noutputs>9)
	    outputs[9] = sum;
	else
		return ierr;

	return ierr;
}


/* --------- Model runner ----------*/
int c_hbv_run(int nval, int nparams,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
    double * params,
    double * inputs,
    double * states,
    double * outputs)
{
    int ierr=0, i, j, *bql;
    double dquh[HBV_MAXUH];

    /* Check dimensions */
    if(noutputs > HBV_NOUTPUTS)
        return HBV_ERROR + __LINE__;

    if(nstates > HBV_NSTATES)
        return HBV_ERROR + __LINE__;

    if(start < 0)
        return HBV_ERROR + __LINE__;

    if(end >= nval)
        return HBV_ERROR + __LINE__;

    /* Check parameters */
    ierr = hbv_minmaxparams(nparams, params);

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
        /* Run timestep model and update states */
    	ierr = hbv_runtimestep(nparams,
                ninputs,
                nstates,
                noutputs,
                params,
                &(inputs[ninputs*i]),
                states,
                &(outputs[noutputs*i]),
                bql, dquh);

        outputs[noutputs*i+1] = (double)(*bql);

        /* Run variable length UH */
        for(j=0; j < *bql; j++)
            if(i+j <= end) outputs[noutputs*(i+j)] += dquh[j];

		if(ierr>0)
			return ierr;
    }

    return ierr;
}

