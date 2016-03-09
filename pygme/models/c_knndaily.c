#include "c_knndaily.h"

#define KNN_DEBUGFLAG_FLAG 0

double get_rand(void)
{
    return (double)(rand())/RAND_MAX;
}

double compute_kernel(int ordinate)
{
    return 1./(1+ordinate);
}


int c_knndaily_getnn(int nval, int nvar,
                int dayofyear_ini,
                int halfwindow,
                int nb_nn,
                double * var,
                int dayofyear,
                double * states,
                double * distances,
                int * idx_potential)
{
    int ierr, k, iyear, nleap;
    int rank, idx, istart, iend, nyears;

    double dst, delta;

    ierr = 0;
    nyears = nval/365;
    nleap = 0;

    /* compute distance */
    for(iyear=0; iyear<nyears; iyear++)
    {
        /* Start / end of window */
        istart = iyear*365 - halfwindow + dayofyear - dayofyear_ini + nleap;
        istart = istart < 0 ? 0 : istart;

        iend = istart + 2*halfwindow;
        iend = iend >= nval ? nval-1 : iend;

        if(KNN_DEBUGFLAG_FLAG >= 2)
            fprintf(stdout, "\n\tiyear %3d : %7d -> %7d\n", iyear, istart, iend);

        /* Approximate correction for leap years */
        nleap += (int)(iyear % 4 == 0);

        /* loop through KNN variables within window and compute distances
            with selected day */
        for(idx=istart; idx<=iend; idx++)
        {
            /* Computes weighted Euclidian distance with potential neighbours */
            dst = 0;
            for(k=0; k<nvar; k++)
            {
                delta = var[idx+nval*k] - states[k];
                dst += delta * delta;
            }
            dst = dst > KNNDAILY_DIST_MAX ? KNNDAILY_DIST_MAX : dst;

            if(isnan(dst))
            {
                if(KNN_DEBUGFLAG_FLAG >= 3)
                    fprintf(stdout, "\t\tidx %7d: dst=%5.10f ( ",
                            idx, dst);
                continue;
            }

            /* Perturb distance to avoid ties */
            dst += KNNDAILY_WEIGHT_MIN * get_rand()/2;

            /* Compute the rank of current neighbour distance
             * within the ones already computed */
            rank = 0;
            while(dst>distances[rank] && rank < nb_nn)
                rank ++;

            /* If current neighbour is within the closest,
             * then rearrange distances and store data */
            if(rank < nb_nn)
            {
                if(KNN_DEBUGFLAG_FLAG >= 3)
                {
                    fprintf(stdout, "\t\tidx %7d: rank=%d dst=%5.10f ( ",
                                idx, rank, dst);
                    for(k=0; k<nvar; k++)
                        fprintf(stdout, "%0.3f ", var[idx+nval*k]);
                    fprintf(stdout, ")\n");
                }

                for(k=nb_nn-1; k>=rank+1; k--)
                {
                    idx_potential[k] = idx_potential[k-1];
                    distances[k] = distances[k-1];
                }

                idx_potential[rank] = idx;
                distances[rank] = dst;
            }

        } /* loop on potential neighbours */

    } /* loop on cycles */

    return ierr;
}

/*
* Daily model :
* Calculation of nearest neighbours according the methodology detailed
* by Lall, U., Sharma, A., 1996. A nearest neighbour bootstrap for time
* series resampling. Water Resources Research 32 (3), 679–693.
*
* Inputs
*   nb_nn   Number of nearest neighbours to consider
*   WINS  Temporal window to restrain the search for nearest neighbours
*   DMAT  Matrix of feature vectors for each days (nval x nvar)
*   SMAT  Data to resample
*   NNdep Initial day
*   RND   Random number vector
*
* Outputs
*   KNNSIM Resampled data
*/

int c_knndaily_run(int nconfig, int nval, int nvar, int nrand,
    int seed,
    int start, int end,
    double * config,
    double * var,
    double * states,
    int * knndaily_idx)
{
    int ierr, i, k;
    int nb_nn, halfwindow, dayofyear;
    int dayofyear_ini, date[3];
    int idx_select, idx_potential[KNNDAILY_NKERNEL_MAX];

    double dist, sum, rnd;
    double kernel[KNNDAILY_NKERNEL_MAX];
    double distances[KNNDAILY_NKERNEL_MAX];

    ierr = 0;

    /* Check dimensions */
    if(nvar > KNNDAILY_NVAR_MAX)
        return KNNDAILY_ERROR + __LINE__;

    if(start < 0)
        return KNNDAILY_ERROR + __LINE__;

    if(end >= nval)
        return KNNDAILY_ERROR + __LINE__;

    /* Set seed */
    if(seed != -1)
        srand(seed);

    /* Half temporal window selection */
    halfwindow = rint(config[0]);
    if(halfwindow < 1 || halfwindow >= 100)
        return KNNDAILY_ERROR + __LINE__;

    /* Number of neighbours */
    nb_nn = rint(config[1]);
    if(nb_nn >= KNNDAILY_NKERNEL_MAX || nb_nn < 1)
        return KNNDAILY_ERROR + __LINE__;

    /* Starting date in input data */
    ierr = c_utils_getdate(config[2], date);
    if(ierr > 0)
        return KNNDAILY_ERROR + __LINE__;

    dayofyear_ini = c_utils_dayofyear(date[1], date[2]);
    if(dayofyear_ini < 1)
        return KNNDAILY_ERROR + __LINE__;

    /* Create resampling kernel */
    sum = 0;
    for(k=0; k<nb_nn; k++) sum += compute_kernel(k);

    kernel[0] = 1./sum;
    for(k=1; k<nb_nn; k++)
        kernel[k] = kernel[k-1] + compute_kernel(k)/sum;

    /* reset distance and potential neighbours */
    for(k=0; k<nb_nn; k++)
    {
        distances[k] = KNNDAILY_DIST_MAX;
        idx_potential[k] = -1;
    }

    /* Get start date */
    ierr = c_utils_getdate(states[nvar], date);
    if(ierr > 0)
        return KNNDAILY_ERROR + __LINE__;

    dayofyear = c_utils_dayofyear(date[1], date[2]);
    if(dayofyear < 1)
        return KNNDAILY_ERROR + __LINE__;

    /* Select the first KNN index as the closest point */
    ierr = c_knndaily_getnn(nval, nvar, dayofyear_ini,
            halfwindow,  nb_nn, var,
            dayofyear, states, distances, idx_potential);

    idx_select = idx_potential[0];
    dist = distances[0];

    /* resample */
    for(i=0; i<nrand; i++)
    {
        /* Initialise KNN variables vector */
        for(k=0; k<nvar; k++)
            states[k] = var[idx_select+nval*k];

        /* Shift by one day */
        c_utils_add1day(date);
        dayofyear = c_utils_dayofyear(date[1], date[2]);
        states[nvar] = date[0] * 1e4 + date[1] * 1e2 + date[2];

        /* reset distance and potential neighbours */
        for(k=0; k<nb_nn; k++)
        {
            distances[k] = KNNDAILY_DIST_MAX;
            idx_potential[k] = -1;
        }

        if(KNN_DEBUGFLAG_FLAG >= 1)
            fprintf(stdout, "\n[%3d] idx select = %7d, "
                    "doy = %3d (%0.0f), doyi = %3d, dist = %0.3f\n", i,
                        idx_select, dayofyear, dayofyear_ini,
                        states[nvar], dist);

        /* Find nearest neighbours */
        ierr = c_knndaily_getnn(nval, nvar, dayofyear_ini,
            halfwindow,  nb_nn, var,
            dayofyear, states, distances, idx_potential);

        if(KNN_DEBUGFLAG_FLAG >= 1)
        {
            fprintf(stdout, "\n");
            for(k=0; k<nb_nn; k++)
                fprintf(stdout, "\tkern(%d)=%0.4f idx=%7d  d=%0.10f\n",
                        k,kernel[k], idx_potential[k], distances[k]);
        }

        /* Select neighbours from candidates */
        rnd = get_rand();
        k = 0;
        while(rnd > kernel[k] && k < nb_nn) k ++;

        /* Save the following day (key of KNN algorithm!)*/
        //idx_select = idx_potential[k]+1;
        idx_select = idx_potential[k]+1;
        if(idx_select >= nval)
            idx_select = idx_potential[k];
        dist = distances[k];

        /* Selected closest index and loop back to beginning if end of the
        series is reached */
        knndaily_idx[i] = rint(idx_select);

        if(KNN_DEBUGFLAG_FLAG >= 1)
            fprintf(stdout, "\n\tRND = %0.5f -> idx_select = %7d\n\n",
                    rnd, idx_select);


    } /* loop on random numbers */

    return ierr;
}


