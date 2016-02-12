#include "c_knn.h"

#define DEBUG_FLAG 1

int c_knn_getnn(int nval, int nvar, 
                int ncycles, double cycle_length,
                int knn_halfwindow, int knn_nb,
                int cycle_position,
                double * var,
                double * weights, 
                double * knn_var,
                double * distances,
                int * knn_idx_potential)
{
    int ierr, k, icycle, idx, istart, iend;
    int knn_rank;

    double w, dst, delta;

    ierr = 0;

    /* compute distance */
    for(icycle=0; icycle<ncycles; icycle++)
    {
        istart = (int)(icycle*cycle_length - knn_halfwindow + cycle_position);
        istart = istart < 0 ? 0 :
                istart > nval-1 ? nval-1 : istart;

        iend = (int)(icycle*cycle_length + knn_halfwindow + cycle_position);
        iend = iend < 0 ? 0 :
                iend > nval-1 ? nval-1 : iend;

        if(DEBUG_FLAG == 1)
            fprintf(stdout, "\tcycle %3d : %d -> %d\n", icycle, istart, iend);

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
                delta = var[idx+nval*k] - knn_var[k];
                dst += delta * delta * w;
            }

            /* Check if the distance is lower than one
             * of the already stored distance */
            knn_rank = 0;
            while(dst>distances[knn_rank] && knn_rank < knn_nb)
                knn_rank ++;

            /* If yes, then rearrange distances */
            if(knn_rank < knn_nb)
            {
                for(k=knn_nb-1; k>=knn_rank+1; k--)
                {
                    knn_idx_potential[k] = knn_idx_potential[k-1];
                    distances[k] = distances[k-1];
                }

                knn_idx_potential[knn_rank] = idx;
                distances[knn_rank] = dst;

                if(DEBUG_FLAG==1)
                    fprintf(stdout, "\t\tidx %d : dst=%0.10f rank=%d\n", idx, dst, knn_rank);
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
*   knn_nb   Number of nearest neighbours to consider
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

int c_knn_run(int nparams, int nval, int nvar, int nrand,
    int seed, 
    int idx_select,
    double * params,
    double * weights,
    double * var,
    int * knn_idx)
{
    int ierr, i, k;
    int knn_halfwindow, knn_nb, ncycles;
    int cycle_position;
    int knn_idx_potential[KNN_NKERNEL_MAX];

    double sum, cycle_length, rnd;
    double kernel[KNN_NKERNEL_MAX];
    double distances[KNN_NKERNEL_MAX];
    double knn_var[KNN_NVAR_MAX];

    ierr = 0;

    /* Check inputs */
    if(nvar > KNN_NVAR_MAX)
        return 10000+__LINE__;

    if(idx_select < 0 || idx_select >= nval-1)
        return 10000+__LINE__;

    /* Set seed */
    srand(seed);

    /* Half temporal window selection */
    knn_halfwindow = (int)(params[0]);

    /* Number of neighbours */
    knn_nb = (int)params[1];
    if(knn_nb >= KNN_NKERNEL_MAX)
        return 10000+__LINE__;

    /* Duration of cycle (icyclely = cycle_length) */
    cycle_length = params[2];
    if(cycle_length < 10)
        return 10000+__LINE__;

    /* Number of cycles in input matrix */
    ncycles = (int)(nval/cycle_length)+1;

    /* Normalise weights */
    sum = 0;
    for(i=0; i<nval; i++) sum += fabs(weights[i]);

    if(sum <= 0)
        return 10000+__LINE__;

    for(i=0; i<nval; i++)
        weights[i] = fabs(weights[i])/sum;

    /* Create resampling kernel */
    sum = 0;
    for(k=0; k<knn_nb; k++) sum += 1./(double)(k+1);

    kernel[0] = 1./sum;
    for(k=1; k<knn_nb; k++)
        kernel[k] = kernel[k-1] + 1./(double)(k+1)/sum;


    /* resample */
    for(i=0; i<nrand; i++)
    {
        if(DEBUG_FLAG == 1)
            fprintf(stdout, "[%3d] idx select = %d\n", i, idx_select);

        /* Initialise KNN variables vector */
        for(k=0; k<nvar; k++)
            knn_var[k] = var[idx_select+nval*k];

        /* reset distance and potential neighbours */
        for(k=0; k<knn_nb; k++)
        {   
            distances[k] = KNN_DIST_INI;
            knn_idx_potential[k] = -1;
        }

        /* Position within cycle */
        cycle_position = fmod(idx_select, cycle_length);

        /* Find nearest neighbours */
        ierr = c_knn_getnn(nval, nvar, ncycles, cycle_length,
                knn_halfwindow,  knn_nb, cycle_position,
                var, weights,
                knn_var, distances, knn_idx_potential);

        if(DEBUG_FLAG == 1)
            for(k=0; k<knn_nb; k++)
                fprintf(stdout, " kern(%d)= %0.4f idx=%d d=%0.10f\n", k,kernel[k], knn_idx_potential[k], distances[k]);

        /* Select neighbours from candidates */
        rnd = (double)(rand())/RAND_MAX;

        k=0;
        while(rnd > kernel[k] && k < knn_nb)
            k ++;

        if(k == knn_nb)
            return 10000+__LINE__;

        /* Save the following day (key of KNN algorithm!)*/
        idx_select = knn_idx_potential[k]+1;
        idx_select = idx_select < nval ? idx_select : nval-1;

        if(idx_select < 0)
            return 10000+__LINE__;
            
        knn_idx[i] = idx_select;

        if(DEBUG_FLAG == 1)
            fprintf(stdout, "RND=%0.5f -> idx=%d\n\n", rnd, idx_select);

        /* Save knn variables for next iteration */
        for(k=0; k<nvar; k++)
            knn_var[k] = var[idx_select+nval*k];

    } /* loop on random numbers */

    return ierr;
}


int c_knn_mindist(int nparams, int nval, int nvar, 
    int cycle_position, 
    int var_cycle_position,
    double * knn_var, 
    double * params,
    double * weights,
    double * var,
    int * knn_idx)
{
    int ierr;
    int k, knn_halfwindow, knn_nb, ncycles;

    int knn_idx_potential[KNN_NKERNEL_MAX];

    double cycle_length;
    double distances[KNN_NKERNEL_MAX];

    ierr = 0;

    /* Check inputs */
    if(nvar > KNN_NVAR_MAX)
        return 10000+__LINE__;

    /* Half temporal window selection */
    knn_halfwindow = (int)(params[0]);

    /* Number of neighbours */
    knn_nb = (int)params[1];
    if(knn_nb >= KNN_NKERNEL_MAX)
        return 10000+__LINE__;

    /* Duration of cycle (icyclely = cycle_length) */
    cycle_length = params[2];
    if(cycle_length < 10)
        return 10000+__LINE__;

    /* Number of cycles in input matrix */
    ncycles = (int)(nval/cycle_length)+1;

    /* reset distance and potential neighbours */
    for(k=0; k<knn_nb; k++)
    {   
        distances[k] = KNN_DIST_INI;
        knn_idx_potential[k] = -1;
    }

    /* To correct for start of var being different from cycle_position=0 */
    cycle_position -= var_cycle_position;

    /* Find closest neighbour */
    ierr = c_knn_getnn(nval, nvar, ncycles, cycle_length,
                knn_halfwindow,  knn_nb, 
                cycle_position,
                var, weights,
                knn_var, distances, knn_idx_potential);

    /* Returns nearest neighbour */
    knn_idx[0] = knn_idx_potential[0];

    return ierr;
}

