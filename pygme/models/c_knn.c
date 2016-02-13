#include "c_knn.h"

#define DEBUG_FLAG 1


int c_knn_getnn(int nval, int nvar,
                int ncycles, double cycle,
                int halfwindow, int nb_nn,
                int cycle_position,
                double * var,
                double * weights,
                double * states,
                double * distances,
                int * idx_potential)
{
    int ierr, k, icycle, idx, istart, iend;
    int rank;

    double w, dst, delta;

    ierr = 0;

    /* compute distance */
    for(icycle=0; icycle<ncycles; icycle++)
    {
        /* Skip the first cycle if cycle_position < 0 */
        if(cycle_position < 0)
            continue;

        istart = (int)(icycle*cycle - halfwindow
                        + cycle_position);
        istart = istart < 0 ? 0 :
                istart > nval-1 ? nval-1 : istart;

        iend = (int)(icycle*cycle + halfwindow
                        + cycle_position);
        iend = iend < 0 ? 0 :
                iend > nval-1 ? nval-1 : iend;

        if(DEBUG_FLAG == 1)
            fprintf(stdout, "\tcycle %3d : %d -> %d\n",
                    icycle, istart, iend);

        /* loop through KNN variables within window and compute distances
            with selected day */
        for(idx=istart; idx<=iend; idx++)
        {
            w = weights[idx];
            if(w < KNN_WEIGHT_MIN)
                continue;

            /* Computes weighted euclidian distance for potential neighbour */
            dst = 0;
            for(k=0; k<nvar; k++)
            {
                delta = var[idx+nval*k] - states[k];
                dst += delta * delta * w;
            }

            /* Check if the distance is lower than one
             * of the already stored distance */
            rank = 0;
            while(dst>distances[rank] && rank < nb_nn)
                rank ++;

            /* If yes, then rearrange distances */
            if(rank < nb_nn)
            {
                for(k=nb_nn-1; k>=rank+1; k--)
                {
                    idx_potential[k] = idx_potential[k-1];
                    distances[k] = distances[k-1];
                }

                idx_potential[rank] = idx;
                distances[rank] = dst;

                if(DEBUG_FLAG==1)
                    fprintf(stdout, "\t\tidx %d : dst=%0.10f rank=%d\n",
                            idx, dst, rank);
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
*   WEI   Weights to calculate the euclidian distance (nvar x 1)
*   SMAT  Data to resample
*   NNdep Initial day
*   RND   Random number vector
*
* Outputs
*   KNNSIM Resampled data
*/

int c_knn_run(int nconfig, int nval, int nvar, int nrand,
    int seed,
    double * config,
    double * weights,
    double * var,
    double * states,
    int * knn_idx)
{
    int ierr, i, k;
    int idx_select;
    int halfwindow, nb_nn, ncycles;
    int cycle_position, cycle_position_ini;
    int idx_potential[KNN_NKERNEL_MAX];

    double sum, cycle, rnd;
    double kernel[KNN_NKERNEL_MAX];
    double distances[KNN_NKERNEL_MAX];

    ierr = 0;

    /* Check dimensions */
    if(nvar > KNN_NVAR_MAX)
        return 10000+__LINE__;

    /* Set seed */
    if(seed != -1)
        srand(seed);

    /* Half temporal window selection */
    halfwindow = (int)(config[0]);

    /* Number of neighbours */
    nb_nn = (int)config[1];
    if(nb_nn >= KNN_NKERNEL_MAX)
        return 10000+__LINE__;

    /* Duration of cycle (icyclely = cycle) */
    cycle = config[2];
    if(cycle < 0)
        return 10000+__LINE__;

    /* Cycle position of first point in var matrix */
    cycle_position_ini = config[3];
    if(cycle_position_ini < 0 || cycle_position > cycle)
        return 10000+__LINE__;

    /* Number of cycles in input matrix */
    ncycles = (int)(nval/cycle)+1;

    /* Normalise weights */
    sum = 0;
    for(i=0; i<nval; i++) sum += fabs(weights[i]);

    if(sum <= 0)
        return 10000+__LINE__;

    for(i=0; i<nval; i++)
        weights[i] = fabs(weights[i])/sum;

    /* Create resampling kernel */
    sum = 0;
    for(k=0; k<nb_nn; k++) sum += 1./(double)(k+1);

    kernel[0] = 1./sum;
    for(k=1; k<nb_nn; k++)
        kernel[k] = kernel[k-1] + 1./(double)(k+1)/sum;

    /* reset distance and potential neighbours */
    for(k=0; k<nb_nn; k++)
    {
        distances[k] = KNN_DIST_INI;
        idx_potential[k] = -1;
    }

    /* Check cycle position */
    cycle_position = states[nvar];
    cycle_position -= cycle_position_ini;
    if(cycle_position < 0 || cycle_position >= cycle)
        return 10000+__LINE__;

    /* Select the first KNN index */
    ierr = c_knn_getnn(nval, nvar, ncycles, cycle,
            halfwindow,  nb_nn, cycle_position,
            var, weights,
            states, distances, idx_potential);
    idx_select = idx_potential[0];

    /* resample */
    for(i=0; i<nrand; i++)
    {
        if(DEBUG_FLAG == 1)
            fprintf(stdout, "[%3d] idx select = %d\n", i, idx_select);

        /* Initialise KNN variables vector */
        for(k=0; k<nvar; k++)
            states[k] = var[idx_select+nval*k];

        /* reset distance and potential neighbours */
        for(k=0; k<nb_nn; k++)
        {
            distances[k] = KNN_DIST_INI;
            idx_potential[k] = -1;
        }

        /* Position within cycle */
        cycle_position = fmod(idx_select, cycle);
        cycle_position -= cycle_position_ini;

        /* Find nearest neighbours */
        ierr = c_knn_getnn(nval, nvar, ncycles, cycle,
                halfwindow,  nb_nn, cycle_position,
                var, weights,
                states, distances, idx_potential);

        if(DEBUG_FLAG == 1)
            for(k=0; k<nb_nn; k++)
                fprintf(stdout, " kern(%d)= %0.4f idx=%d d=%0.10f\n",
                        k,kernel[k], idx_potential[k], distances[k]);

        /* Select neighbours from candidates */
        rnd = (double)(rand())/RAND_MAX;

        k = 0;
        while(rnd > kernel[k] && k < nb_nn) k ++;

        /* Save the following day (key of KNN algorithm!)*/
        idx_select = idx_potential[k]+1;
        idx_select = idx_select < nval ? idx_select : nval-1;

        if(idx_select < 0)
            return 10000+__LINE__;

        knn_idx[i] = idx_select;

        if(DEBUG_FLAG == 1)
            fprintf(stdout, "RND=%0.5f -> idx=%d\n\n", rnd, idx_select);

        /* Save knn variables for next iteration */
        for(k=0; k<nvar; k++)
            states[k] = var[idx_select+nval*k];

    } /* loop on random numbers */

    return ierr;
}


