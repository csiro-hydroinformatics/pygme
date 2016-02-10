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
    double * rand,
    int * knn_idx)
{
    int ierr;
    int i, k, idx;
    int knn_window, knn_nb, nyears, year;
    int winstart, winend, nn_rank;
    int knn_idx[KNN_NKERNEL_MAX];

    double sum, d, w, delta, rand;
    double kernel[KNN_NKERNEL_MAX];
    double distances[KNN_NKERNEL_MAX];
    double knn_var[KNN_NVAR_MAX];

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

    /* Number of years in input matrix */
    nyears = (int)(nval/365.25)+1;

    /* Normalise weights */
    sum = 0;
    for(i=0; i<nval; i++) sum += fabs(weights[i]);

    if(sum <= 0)
        return 444;

    for(i=0; i<nval; i++)
        weigths[i] = fabs(weights[i])/sum;

    /* Create resampling kernel */
    sum = 0;
    for(i=0; i<knn_nb; i++) sum += 1./(double)(i+1);

    kernel[0] = 1./sum;
    for(i=1; i<knn_nb; i++)
    {
        kernel[i] = kernel[i-1] + 1./(double)(i+1)/sum;

        /* Initialise list of potential nn candidates */
        knn_idx[i] = 0;
    }

    /* Initialise variables of neighbour */
    for(i=0; i<nvar; i++)
        knn_var[i] = var[idx_select+nval*i];


    /* resample */
    for(i=0; i<nrand; i++)
    {
        /* reset distance */
        for(k=0; k<knn_nb; k++) distance[k] = 0;

        /* compute distance */
        for(year=-1; year<nyears; year++)
        {
            istart = (int)(year*365.25 - knn_window + fmod(idx_selec, 365.25));
            istart = istart < 0 ? 0 :
                    istart >= nval-1 ? nval-1 : istart;

            iend = (int)(year*365.25 + knn_window + fmod(idx_selec, 365.25));
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
                for(l=0; l<nvar; l++)
                {
                    delta = var[idx+nval*l] - knn_var[l];
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
                    for(l=knn_nb-1; l>=knn_rank+1; l--)
                    {
                        knn_idx[l] = knn_idx[l-1];
                        distances[l] = distances[l-1];
                    }

                    knn_idx[knn_rank] = idx;
                    distances[knn_rank] = d;
                }

            } /* loop on potential neighbours */

        } /* loop on years */

        /* Select neighbours from candidates */
        rnd = rand[i];
        rnd = rnd < 0 ? 0 : rnd > 1 ? 1 : rnd;

        l=0;
        while(rnd > kernel && l < knn_nb)
            l ++;

        /* Save the following day (key of KNN algorithm!)*/
        idx_select = knn_idx[l]+1;
        knn_idx[i] = idx_select;

        /* Save knn variables for next iteration */
        for(k=0; k<nvar; k++)
            knn_var[k] = var[idx_select+nval*k];

    } /* loop on random numbers */

    return ierr;
}
