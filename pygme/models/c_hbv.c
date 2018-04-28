#include "c_hbv.h"
#include "c_uh.h"


int hbv_minmaxparams(int nparams, double * params)
{
    if(nparams<4)
    {
        return HBV_ERROR + __LINE__;
    }

	params[0] = c_minmax(1, 1e4, params[0]); 	// S
	params[1] = c_minmax(-50, 50, params[1]);	// IGF
	params[2] = c_minmax(1, 1e4, params[2]); 	// R
	params[3] = c_minmax(0.5, 50, params[3]); // TB

	return 0;
}


int hbv_soilmoisture(double rain, double etp, double moist,
        double lp, double fc, double beta, double *prod)
{
      subroutine soilmoisture(rain,melt,etp,LP,FC,Beta,dmoist,moist,
     *           dq,eta)

    double moistold, xx;
    double dq, dmoist;
    double etp, eta;

    /* No snow melt runoff */
    melt = 0;

    /* soil mositure accounting */
    moistold = moist;
    dq = power(moistold/FC, beta)*(rain+melt);
    dq = dq > rain+melt ? rain+melt : dq;

    dmoist = rain+melt-dq;
    moist = moistold+dmoist;

    if(moist > fc)
    {
      dq = (moist-fc)+dq;
      moist = fc;
    }

    /* calculate evapotranspiration */
    if(moist < lp)
    {
        eta = moist*etp/lp;
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


int hbv_respfunc(double dq, double k0, double lsuz,
        double kl, double k2, double cperc, double bmax, double croute,
        double suz, double slz,
        double *resp)
{
    int bql;
    double rat, bq, suz,slz,suzold,slzold,slzin
    double q0,q1,q2,qg,sum

    double dquh[HBV_MAXUH];

    /* The split rat/1-rat is not implemented */
    rat = 1.0;
    suzold = suz+rat*dq;
    slzold = slz+(1.-rat)*dq;

    slzin = cperc;

    suzold = suzold < 0 ? 0 : suzold;
    slzold = slzold < 0 ? 0 : slzold;

    /* upper storage */
    if(suzold > lsuz)
    {
        q0 = (suzold-lsuz)/k0*exp(-1./k0);
        q0 = q0 < 0 ? 0 : q0;
        q0 = q0 > suzold-lsuz ? suzold-lsuz : q0;
    } else {
        q0 = 0.;
    }
    suzold = suzold-q0;

    q1 = -slzin+(slzin+suzold/k1)*exp(-1./k1);
    q1 = q1 < 0 ? 0. : q1;

    suz = suzold-q1-slzin;
    if(suz < 0)
    {
        suz = 0.;
        slzin = suzold;
    }

    /* lower storage */
    q2 = slzin-(slzin-slzold/k2)*exp(-1./k2);
    q2 = q2 < 0 ? 0. : q2;

    slz = slzold-q2+slzin;
    if(slz < 0)
    {
      slz = 0.;
      q2 = slzold+slzin;
    }
    qg = q0+q1+q2;

    /* transformation function */
    if(bmax-croute*qg > 1.)
    {
        bq = bmax-croute*qg;
        bql = (int)bq;

        sum=0.
        for(j=1; j<=bql; j++)
        {
            if(j <= bql/2)
            {
                dquh[j-1]=((j-0.5)*4.*qg)/(bql*bql*1.);
            }
            else if (fabs(j-(bql/2.+0.5)) < 0.1)
            {
                dquh[j-1]=((j-0.75) *4.*qg)/(bql*bql*1.);
            }
            else
            {
                dquh[j-1]=((bql-j+0.5)*4.*qg)/(bql*bql*1.)
            }
            sum = sum + dquh[j];
         }
    } else {
        bql = 1;
        dquh[0] = qg;
        sum = qg;
    }

    resp[0] = q0;
    resp[1] = q1;
    resp[2] = q2;
    resp[3] = qg;
    resp[4] = sum;

    return 0;
}


int hbv_runtimestep(int nparams,
    int nuh1, int nuh2,
    int ninputs,
    int nstates,
    int noutputs,
    double * params,
    double * uh1,
    double * uh2,
    double * inputs,
    double * statesuh1,
    double * statesuh2,
    double * states,
    double * outputs)
{
    int ierr=0;

    double Q, P, E, Q1, Q9;
    double prod[7];
    double ES, PS, PR, AE;
    double PERC,ECH,TP,R2,QR,QD;
    double EN, ech1,ech2, RR, RR4;
    double uhoutput1[1], uhoutput2[1];

    double partition1 = 0.9;

    /* inputs */
    P = inputs[0];
    P = P < 0 ? 0 : P;

    E = inputs[1];
    E = E < 0 ? 0 : E;

    /* Production */
    hbv_production(P, E, params[0], states[0], prod);

    EN = prod[0];
    PS = prod[1];
    ES = prod[2];
    AE = prod[3];
    PERC = prod[4];
    PR = prod[5];
    states[0] = prod[6];

    /* UH */
    uh_runtimestep(nuh1, PR, uh1, statesuh1, uhoutput1);
    uh_runtimestep(nuh2, PR, uh2, statesuh2, uhoutput2);

    /* Potential Water exchange */
    RR = states[1]/params[2];
    ECH = params[1]*RR*RR*RR*sqrt(RR);

    /* Routing store calculation */
    Q9 = *uhoutput1 * partition1;
    TP = states[1] + Q9 + ECH;

    /* Case where Reservoir content is not sufficient */
    ech1 = ECH-TP;
    states[1] = 0;

    if(TP>=0)
    {
        states[1]=TP;
        ech1=ECH;
    }
    RR = states[1]/params[2];
    RR4 = RR*RR;
    RR4 *= RR4;
    R2 = states[1]/sqrt(sqrt(1.+RR4));
    QR = states[1]-R2;
    states[1] = R2;

    /* Direct runoff calculation */
    QD = 0;

    /* Case where the UH cannot provide enough water */
    Q1 = *uhoutput2 * (1-partition1);
    TP = Q1 + ECH;
    ech2 = ECH-TP;
    QD = 0;

    if(TP>0)
    {
        QD = TP;
        ech2 = ECH;
    }

    /* TOTAL STREAMFLOW */
    Q = QD + QR;

    /* RESULTS */
    outputs[0] = Q;

    if(noutputs>1)
        outputs[1] = ech1+ech2;
    else
	return ierr;

    if(noutputs>2)
	    outputs[2] = AE;
    else
	    return ierr;

    if(noutputs>3)
	    outputs[3] = PR;
    else
	    return ierr;

    if(noutputs>4)
        outputs[4] = QD;
    else
        return ierr;

    if(noutputs>5)
        outputs[5] = QR;
    else
        return ierr;

    if(noutputs>6)
	outputs[6] = PERC;
    else
	    return ierr;

    if(noutputs>7)
	    outputs[7] = states[0];
	else
		return ierr;

    if(noutputs>8)
	    outputs[8] = states[1];
	else
		return ierr;

	return ierr;
}


/* --------- Model runner ----------*/
int c_hbv_run(int nval, int nparams,
    int nuh1, int nuh2,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
    double * params,
    double * uh1,
    double * uh2,
    double * inputs,
    double * statesuh1,
    double * statesuh2,
    double * states,
    double * outputs)
{
    int ierr=0, i;

    /* Check dimensions */
    if(noutputs > HBV_NOUTPUTS)
        return HBV_ERROR + __LINE__;

    if(nstates > HBV_NSTATES)
        return HBV_ERROR + __LINE__;

    if(nuh1+nuh2 > NUHMAXLENGTH)
        return HBV_ERROR + __LINE__;

    if(nuh1 <= 0 || nuh2 <= 0)
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
                nuh1, nuh2,
                ninputs,
                nstates,
                noutputs,
                params,
                uh1, uh2,
                &(inputs[ninputs*i]),
                statesuh1,
                statesuh2,
                states,
                &(outputs[noutputs*i]));

		if(ierr>0)
			return ierr;
    }

    return ierr;
}

