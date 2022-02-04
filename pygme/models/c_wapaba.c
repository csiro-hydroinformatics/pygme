#include "c_wapaba.h"

/*******************************************************************************
* Run time step code for the WAPABA rainfall-runoff model
*
* --- Inputs
* ierr			Error message
* nparams			Number of paramsameters (4)
* ninputs		Number of inputs (2)
* nstates		Number of states (1 output + 2 model states + 8 variables = 11)
* nuh			Number of uh ordinates (2 uh)
*
* params			Model paramsameters. 1D Array nparams(4)x1
*					params[0] = S
*					params[1] = IGF
*
* uh			uh ordinates. 1D Array nuhx1
*
* inputs		Model inputs. 1D Array ninputs(2)x1
*
* states		Output and states variables. 1D Array nstates(11)x1
*
*******************************************************************************/
int c_wapaba_runtimestep(int nparams, int ninputs,
        int nstates, int noutputs,
	    double * params,
        double * inputs,
        double * states,
        double * outputs)
{
    int ierr=0;
    int newton_iter, newton_itermax=50;

    /* parameters */
    double ALPHA1 = params[0];  // Retention efficiency, i.e. a larger ALPHA1 value will
                                // result in more rainfall retentionand less direct runoff
    double ALPHA2 = params[1];  // Evapotranspiration efficiency
    double BETA = params[2];	// Deep drainage (recharge) coefficient
    double SMAX = params[3];	// SMAX[mm]     Soil water storage capacity
    double INVK = params[4];	// INVK[-]		Baseflow coefficient
    double K = 1/INVK;

    /* model variables */
    double P, E, Dmth;
    double S, G;
    double omega, F1, F2, ET, Y, R, Qs, Qb, Q, X, W, X0;

    /* inputs */
    P = c_max(0, inputs[0]);
    E = c_max(0, inputs[1]);
    Dmth = (int)c_minmax(1., 31., inputs[2]);

    /* states */
    S = c_minmax(0, SMAX, states[0]);
    G = c_max(0, states[1]);

    /* Catchment water consumption */
	X0 = c_max(0., E+(SMAX-S)); //    !(3)
    // !(1) Where P = Supply and X0 = demand
	F1 = 1.+P/X0-pow(1.+pow(P/X0, ALPHA1), (1./ALPHA1));
	X = X0*F1; // !(2)

    /* Catchment water yield */
    Y = c_max(0., P-X); //     !(4)

    /* !Stage 2
       !Water availability for ET(5)
    */
    W = S+X; // !(5)

    /*!Actual Evapotranspiration from eqn. (6) & (1) */
    if (W>0.)
    {
        // !(1) Where W = Supply and ET0 = demand
    	F2 = 1.+W/E-pow(1.+pow((W/E), ALPHA2), (1./ALPHA2)) ;

        // !(6) !Added min statement to ensure S never goes below zero in equation 7
        // - this is an addition to the paper
    	ET = c_min(E*F2, W);
    }
    else
    	ET = 0.;

    //!Water in the soil store from eqn. (7)
    S = W-ET;			//           !(7)

    //!Prevent S > SMAX(to resolve problem not described in original paper where st > SMAX)
    if (S>SMAX)
    {
    	Y = Y-SMAX+S;
    	S = SMAX;
    }

    //!Stage 3
    //!Recharge to groundwater
    R = BETA*Y;  //             !(8)

    // !Surface discharge
    Qs = c_max(0., Y-R); //     !(9)

    //!Stage 4
    //!Groundwater storage and baseflow estimation(10)
    omega = 1.-exp(-Dmth/K);
    Qb = G*omega+R*(1.-(K/Dmth)*omega); // !(10)
    G = G+R-Qb;	//   !(11)

    // !Total streamflow :
    //[mm] total runoff comming to channels
    Q = Qs+Qb;

    /* states */
    states[0] = S;
    states[1] = G;

    /* output */
    outputs[0] = Q;

    if(noutputs>1)
        outputs[1] = S;
    else
        return ierr;

    if(noutputs>2)
        outputs[2] = G;

    if(noutputs>3)
        outputs[3] = ET;
    else
        return ierr;

    if(noutputs>4)
        outputs[4] = F1;
    else
        return ierr;

    if(noutputs>5)
        outputs[5] = F2;
    else
        return ierr;

    if(noutputs>6)
        outputs[6] = R;
    else
        return ierr;

    if(noutputs>7)
        outputs[7] = Qb;
    else
        return ierr;

    if(noutputs>8)
        outputs[8] = Qs;
    else
        return ierr;

    if(noutputs>9)
        outputs[9] = W;
    else
        return ierr;


    return ierr;
}


// --------- Component runner --------------------------------------------------
int c_wapaba_run(int nval,
    int nparams,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
    double * params,
    double * inputs,
    double * statesini,
    double * outputs)
{
    int ierr, i;

    /* Check dimensions */
    if(nparams < WAPABA_NPARAMS)
        return WAPABA_ERROR + __LINE__;

    if(nstates < WAPABA_NSTATES)
        return WAPABA_ERROR + __LINE__;

    if(ninputs < WAPABA_NINPUTS)
        return WAPABA_ERROR + __LINE__;

    if(start < 0)
        return WAPABA_ERROR + __LINE__;

    if(end >=nval)
        return WAPABA_ERROR + __LINE__;

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
       /* Run timestep model and update states */
    	ierr = c_wapaba_runtimestep(nparams,
                ninputs,
                nstates,
                noutputs,
    		    params,
                &(inputs[ninputs*i]),
                statesini,
                &(outputs[noutputs*i]));

        if(ierr > 0 )
            return ierr;
    }

    return ierr;
}

