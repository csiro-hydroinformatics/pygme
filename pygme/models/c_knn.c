#include "c_knn.h"

#define KNN_DEBUG_FLAG 0

double get_rand(void)
{
    return (double)(rand())/RAND_MAX;
}

double compute_kernel(int ordinate)
{
    return 1./(1+ordinate);
}

int c_knn_getnn(int nval, int nvar,
                int ncycles,
                double cycle,
                double cycle_position_ini,
                double halfwindow,
                int nb_nn,
                double * var,
                double * weights,
                double * states,
                double * distances,
                double * idx_potential)
{
    int ierr, k, icycle;
    int rank, idx2;

    double idx, istart, iend;
    double cycle_position, w, dst, delta;
    double delta_season, sin0;

    ierr = 0;

    cycle_position = states[nvar];
    iend = -1;
    sin0 = sin(cycle_position * 2 * M_PI/cycle);

    /* compute distance */
    for(icycle=0; icycle<ncycles; icycle++)
    {
        /* Start / end of window */
        istart = rint(icycle*cycle - halfwindow
                        + cycle_position - cycle_position_ini);


        iend = rint(icycle*cycle + halfwindow
                        + cycle_position - cycle_position_ini);

        /* Skip case of overlapping periods */
        if(istart == iend)
            continue;

        if(KNN_DEBUG_FLAG >= 2)
            fprintf(stdout, "\n\tipos %0.1f - cycle %3d : %7.2f -> %7.2f\n",
                    cycle_position, icycle, istart, iend);

        /* loop through KNN variables within window and compute distances
            with selected day */
        for(idx=istart; idx<=iend; idx++)
        {
            /* round index */
            idx2 = rint(idx);
            if(idx2 < 0 || idx2 >= nval)
                continue;

            /* Get weight */
            w = weights[idx2];
            if(w < KNN_WEIGHT_MIN)
                continue;

            /* Computes weighted Euclidian distance with potential neighbours */
            dst = 0;
            for(k=0; k<nvar; k++)
            {
                delta = var[idx2+nval*k] - states[k];
                dst += delta * delta * w;
            }
            dst = dst > KNN_DIST_MAX ? KNN_DIST_MAX : dst;

            if(isnan(dst))
            {
                fprintf(stdout, "\t\tidx %7.2f (%4d): dst=%5.10f ( ",
                            idx, idx2, dst);
                continue;
            }

            /* Perturb distance to avoid ties */
            dst += KNN_WEIGHT_MIN * get_rand()/2;

            /* Compute the rank of current neighbour distance
             * within the ones already computed */
            rank = 0;
            while(dst>distances[rank] && rank < nb_nn)
                rank ++;

            /* If current neighbour is within the closest,
             * then rearrange distances and store data */
            if(rank < nb_nn)
            {
                if(KNN_DEBUG_FLAG >= 3)
                {
                    fprintf(stdout, "\t\tidx %7.2f (%4d): rank=%d dst=%5.10f ( ",
                                idx, idx2, rank, dst);
                    for(k=0; k<nvar; k++)
                        fprintf(stdout, "%0.3f ", var[idx2+nval*k]);
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
    int start, int end,
    double * config,
    double * weights,
    double * var,
    double * states,
    int * knn_idx)
{
    int ierr, i, k;
    int nb_nn, ncycles;
    int idx_select2;

    double idx_select, dist;
    double cycle_position_ini;
    double idx_potential[KNN_NKERNEL_MAX];
    double sum, cycle, rnd, halfwindow;
    double kernel[KNN_NKERNEL_MAX];
    double distances[KNN_NKERNEL_MAX];

    ierr = 0;

    /* Check dimensions */
    if(nvar > KNN_NVAR_MAX)
        return ESIZE_CONFIG;

    if(start < 0)
        return ESIZE_OUTPUTS;

    if(end >= nval)
        return ESIZE_OUTPUTS;

    /* Set seed */
    if(seed != -1)
        srand(seed);

    /* Number of neighbours */
    nb_nn = (int)config[1];
    if(nb_nn >= KNN_NKERNEL_MAX)
        return 7000 + __LINE__;

    /* Duration of cycle (icyclely = cycle) */
    cycle = config[2];
    if(cycle < 0)
        return 7000 + __LINE__;

    /* Half temporal window selection */
    halfwindow = config[0];
    if(halfwindow < 0 || halfwindow >= cycle/2)
        return 7000 + __LINE__;

    /* Cycle position of first point in var matrix */
    cycle_position_ini = config[3];
    if(cycle_position_ini < 0 || cycle_position_ini > cycle)
        return 7000 + __LINE__;


    /* Number of cycles in input matrix */
    ncycles = (int)(nval/cycle)+1;

    /* Create resampling kernel */
    sum = 0;
    for(k=0; k<nb_nn; k++) sum += compute_kernel(k);

    kernel[0] = 1./sum;
    for(k=1; k<nb_nn; k++)
        kernel[k] = kernel[k-1] + compute_kernel(k)/sum;

    /* reset distance and potential neighbours */
    for(k=0; k<nb_nn; k++)
    {
        distances[k] = KNN_DIST_MAX;
        idx_potential[k] = -1;
    }

    /* Check cycle position */
    if(states[nvar] < 0)
        return 7000 + __LINE__;

    if(states[nvar] >= cycle)
        return 7000 + __LINE__;

    /* Select the first KNN index */
    ierr = c_knn_getnn(nval, nvar, ncycles, cycle,
            cycle_position_ini,
            halfwindow,  nb_nn, var, weights,
            states, distances, idx_potential);
    idx_select = idx_potential[0];
    dist = distances[0];

    /* resample */
    for(i=0; i<nrand; i++)
    {
        idx_select2 = (int) rint(idx_select);

        /* Initialise KNN variables vector */
        for(k=0; k<nvar; k++)
            states[k] = var[idx_select2+nval*k];

        /* Position within cycle  - CRITICAL FORMULA !!*/
        states[nvar] = fmod(idx_select + cycle_position_ini, cycle);

        /* reset distance and potential neighbours */
        for(k=0; k<nb_nn; k++)
        {
            distances[k] = KNN_DIST_MAX;
            idx_potential[k] = -1;
        }

        if(KNN_DEBUG_FLAG >= 1)
            fprintf(stdout, "\n[%3d] idx select = %7.2f, "
                    "cycle pos = %3.1f, dist = %0.3f\n", i,
                        idx_select, states[nvar], dist);

        /* Find nearest neighbours */
        ierr = c_knn_getnn(nval, nvar, ncycles, cycle,
                cycle_position_ini,
                halfwindow,  nb_nn, var, weights,
                states, distances, idx_potential);

        if(KNN_DEBUG_FLAG >= 1)
        {
            fprintf(stdout, "\n");
            for(k=0; k<nb_nn; k++)
                fprintf(stdout, "\tkern(%d)=%0.4f idx=%7.2f  d=%0.10f\n",
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
        knn_idx[i] = rint(idx_select);

        if(KNN_DEBUG_FLAG >= 1)
            fprintf(stdout, "\n\tRND = %0.5f -> idx_select = %7.2f\n\n",
                    rnd, idx_select);


    } /* loop on random numbers */

    return ierr;
}


