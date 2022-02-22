#include "c_ihacres.h"

/***
* This code is adapted from the hydromad implementation of the IHACRES-CMD
* soil moisture accounting module written by Felix Andrews and Joseph Guillaume:
* https://github.com/josephguillaume/hydromad/blob/master/R/cmd.R
*
*
****/


int ihacres_minmaxparams(int nparams, double * params)
{
    if(nparams<2)
    {
        return IHACRES_ERROR + __LINE__;
    }

	params[0] = c_minmax(0.05, 3, params[0]); 	// f
	params[1] = c_minmax(1e1, 3e3, params[1]);	// d

	return 0;
}


/*******************************************************************************
* Run time step code for the IHACRES rainfall-runoff model
*
* --- Inputs
* ierr			Error message
* nconfig		Number of configuration elements (1)
* nparams			Number of paramsameters (4)
* ninputs		Number of inputs (2)
* nstates		Number of states (1 output + 2 model states + 8 variables = 11)
* noutputs		Number of outputs (2)
*
* config		Model configuration. 1D Array
* params		Model paramsameters. 1D Array
*					params[0] = f
*					params[1] = d
*
* inputs		Model inputs. 1D Array
* states		States variables. 1D Array
* outputs		Model outputs. 1D Array
*
*******************************************************************************/

int c_ihacres_runtimestep(int nconfig, int nparams, int ninputs,
        int nstates, int noutputs,
        double * config,
	    double * params,
        double * inputs,
        double * states,
        double * outputs)
{
    int ierr=0;

    /* Shape config */
    double shape = config[0];
    double param_e = config[1];
    double a;

    /* parameters */
    double param_f = params[0];
    double param_d = params[1];
    double param_g = param_f*param_d;

    /* model variables */
    double P, E;
    double M, Mf, M_prev;
    double U, ET;

    /* inputs */
    P = inputs[0] < 0 ? 0 : inputs[0];
    E = inputs[1] < 0 ? 0 : inputs[1];

    M_prev = c_max(0, states[0]);

    /* main IHACRES procedure */
    Mf = M_prev;

    // select form of dU/dP relationship
    // rainfall reduces CMD (Mf)
    if (shape < 1-1e-10)
    {
        // linear form: dU/dP = 1 - (M/d)
        if (M_prev < param_d) {
            Mf = M_prev * exp(-P / param_d);
        } else if (M_prev < param_d + P) {
            Mf = param_d * exp((-P + M_prev - param_d) / param_d);
        } else {
            Mf = M_prev - P;
        }
    }
    else if (abs(shape-1)<1e-10)
    {
        // hyperbolic form: dU/dP = 1 - ??
        if (M_prev < param_d) {
            Mf = 1 / tan((M_prev / param_d) * (UTILS_PI / 2));
            Mf = (2 * param_d / UTILS_PI) * atan(1 / (UTILS_PI * P / (2 * param_d) + Mf));
        } else if (M_prev < param_d + P) {
            Mf = (2 * param_d / UTILS_PI) * atan(2 * param_d / (UTILS_PI * (param_d - M_prev + P)));
        } else {
            Mf = M_prev - P;
        }
    }
    else
    {
        // shape > 1
        // power form: dU/dP = 1 - (M/d)^b
        a = pow(10, (shape / 50));
        if (M_prev < param_d) {
            Mf = M_prev * pow(1 - ((1-a) * P / pow(param_d,a)) /
                              pow(M_prev, (1-a)), 1/(1-a));
        } else if (M_prev < param_d + P) {
            Mf = param_d * pow(1 - (1-a) * (P - M_prev + param_d) / param_d, 1/(1-a));
        } else {
            Mf = M_prev - P;
        }
    }

    // drainage (rainfall not accounted for in -dM)
    U = c_max(0, P - M_prev + Mf);
    // evapo-transpiration
    ET = param_e * E * c_min(1, exp(2 * (1 - Mf / param_g)));
    ET = c_max(0, ET);
    // mass balance
    M = c_max(0, M_prev - P + U + ET);

    /* states */
    states[0] = M;

    /* output */
    outputs[0] = U;

    if(noutputs>1)
        outputs[1] = M;
    else
        return ierr;

    if(noutputs>2)
        outputs[2] = Mf;

    if(noutputs>3)
        outputs[3] = ET;

    return ierr;
}


// --------- Component runner --------------------------------------------------
int c_ihacres_run(int nval,
    int nconfig,
    int nparams,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
    double * config,
    double * params,
    double * inputs,
    double * statesini,
    double * outputs)
{
    int ierr, i;

    /* Check dimensions */
    if(nconfig != IHACRES_NCONFIG)
        return IHACRES_ERROR + __LINE__;

    if(nparams != IHACRES_NPARAMS)
        return IHACRES_ERROR + __LINE__;

    if(nstates != IHACRES_NSTATES)
        return IHACRES_ERROR + __LINE__;

    if(ninputs != IHACRES_NINPUTS)
        return IHACRES_ERROR + __LINE__;

    if(noutputs > IHACRES_NOUTPUTS)
        return IHACRES_ERROR + __LINE__;

    if(start < 0)
        return IHACRES_ERROR + __LINE__;

    if(end >=nval)
        return IHACRES_ERROR + __LINE__;

    /* Check parameters */
    ierr = ihacres_minmaxparams(nparams, params);

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
       /* Run timestep model and update states */
    	ierr = c_ihacres_runtimestep(nconfig, nparams,
                ninputs,
                nstates,
                noutputs,
                config,
    		    params,
                &(inputs[ninputs*i]),
                statesini,
                &(outputs[noutputs*i]));

        if(ierr > 0 )
            return ierr;
    }

    return ierr;
}

