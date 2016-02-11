#include "c_knn.h"

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
    int idx_select,
    double * params,
    double * weights,
    double * var,
    double * rands,
    int * knn_idx)
{
    int ierr;
    int i, k, idx, knn_rank;
    int knn_window, knn_nb, ncycles, icycle;
    int istart, iend;
    int knn_idx_potential[KNN_NKERNEL_MAX];

    double sum, d, w, delta, cycle_length, rnd;
    double kernel[KNN_NKERNEL_MAX];
    double distances[KNN_NKERNEL_MAX];
    double knn_var[KNN_NVAR_MAX];

    ierr = 0;

    fprintf(stdout, "\n\n %s B@%d ierr=%d\n\n", __FILE__, __LINE__, ierr);

    /* Check inputs */
    if(nvar > KNN_NVAR_MAX)
        return 444;

    if(idx_select < 0 || idx_select >= nval-1)
        return 444;

    /* Half temporal window selection */
    knn_window = (int)params[0];

    /* Number of neighbours */
    knn_nb = (int)params[1];
    if(knn_nb >= KNN_NKERNEL_MAX)
        return 444;

    /* Duration of cycle (icyclely = cycle_length) */
    cycle_length = params[2];
    if(cycle_length < 10)
        return 444;

    /* Number of cycles in input matrix */
    ncycles = (int)(nval/cycle_length)+1;

    /* Normalise weights */
    sum = 0;
    for(i=0; i<nval; i++) sum += fabs(weights[i]);

    if(sum <= 0)
        return 444;

    for(i=0; i<nval; i++)
        weights[i] = fabs(weights[i])/sum;

    /* Create resampling kernel */
    sum = 0;
    for(i=0; i<knn_nb; i++) sum += 1./(double)(i+1);

    kernel[0] = 1./sum;
    for(i=1; i<knn_nb; i++)
    {
        kernel[i] = kernel[i-1] + 1./(double)(i+1)/sum;

        /* Initialise list of potential nn candidates */
        knn_idx_potential[i] = 0;
    }

    /* Initialise variables of neighbour */
    for(i=0; i<nvar; i++)
        knn_var[i] = var[idx_select+nval*i];


    /* resample */
    for(i=0; i<nrand; i++)
    {
        /* reset distance */
        for(k=0; k<knn_nb; k++) distances[k] = 0;

        /* compute distance */
        for(icycle=-1; icycle<ncycles; icycle++)
        {
            istart = (int)(icycle*cycle_length - knn_window + fmod(idx_select, cycle_length));
            istart = istart < 0 ? 0 :
                    istart >= nval-1 ? nval-1 : istart;

            iend = (int)(icycle*cycle_length + knn_window + fmod(idx_select, cycle_length));
            iend = iend < 0 ? 0 :
                    iend >= nval-1 ? nval-1 : iend;

            /* loop through data and compute distances */
            for(idx=istart; idx<iend+1; idx++)
            {
                w = weights[idx];
                if(w < KNN_WEIGHT_MIN)
                    continue;

                /* Computes weighted euclidian distance for potential neighbour */
                d = 0;
                for(k=0; k<nvar; k++)
                {
                    delta = var[idx+nval*k] - knn_var[k];
                    d += delta * delta * w;
                }

                /* Check if the distance is lower than one
                 * of the already stored distance */
                knn_rank = 0;
                while(d>distances[knn_rank] && knn_rank <= knn_nb)
                    knn_rank ++;

                /* If yes, then rearrange distances
                 * and list of selected vectors */
                if(knn_rank < knn_nb)
                {
                    for(k=knn_nb-1; k>=knn_rank+1; k--)
                    {
                        knn_idx_potential[k] = knn_idx[k-1];
                        distances[k] = distances[k-1];
                    }

                    knn_idx_potential[knn_rank] = idx;
                    distances[knn_rank] = d;
                }

            } /* loop on potential neighbours */

        } /* loop on cycles */

        /* Select neighbours from candidates */
        rnd = rands[i];
        rnd = rnd < 0 ? 0 : rnd > 1 ? 1 : rnd;

        k=0;
        while(rnd > kernel[k] && k < knn_nb)
            k ++;

        /* Save the following day (key of KNN algorithm!)*/
        idx_select = knn_idx[k]+1;
        knn_idx[i] = idx_select;

        /* Save knn variables for next iteration */
        for(k=0; k<nvar; k++)
            knn_var[k] = var[idx_select+nval*k];

    } /* loop on random numbers */

    return ierr;
}
