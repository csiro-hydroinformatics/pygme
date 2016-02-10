#include "c_knndaily.h"

/*
* Daily model : 
* Calculation of nearest neighbours according the methodology detailed  
* by Lall, U., Sharma, A., 1996. A nearest neighbour bootstrap for time 
* series resampling. Water Resources Research 32 (3), 679–693.
* 
* Inputs 
*   NBK   Number of nearest neighbours to consider
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

int c_knndaily_run(int nparams, int nval, int nvar, int nrand, 
    double * params,
    double * weights,
    double * var,
    double * rand,
    double * outputs)
{
    int ierr;
    int i, winlag, nbk, nyears;

    double sum;
    double kernel[KNN_NKERNEL_MAX];

    /* Half temporal window selection */
    winlag = (int)params[0];

    /* Number of neighbours */
    nbk = (int)params[1];

    /* Number of years in input matrix */
    nyears = (int)(nval/365.25)+1;

    /* Normalise weights */
    sum = 0;
    for(i=0; i<nval; i++) sum += fabs(weights[i]);
    if(sum <= 0)
        return 444;

    for(i=0; i<nval; i++) weigths[i] = fabs(weights[i])/sum;

    /* Create resampling kernel */
    sum = 0;
    for(i=0; i<nbk; i++) sum += 1./(double)(i+1);

    kernel[0] = 1./sum;
    for(i=1; i<nbk; i++) kernel[i] = kernel[i-1] + 1./(double)(i+1)/sum;


    return ierr;
}

int knn_sim(int nval, int nvar,
	int * _nrand, double * NBK, 
  	double * WINS, double * DMAT, double * WEI, double * DMATSELEC, 
  	double * NNdep, double * RND,double *PRINTMESSAGE, double * KNNSIM_NN)
{
    int nbk,dmatCol,nnCol,day,day2,NbAnn,ann,debDay,finDay,nnSelec;
    int NNliste[MAXCOL],nnClass,nbjDMATSELEC;
    double DIST[MAXCOL],CARAC[MAXCOL],winlag,SomW;
    double rnd, KERN[MAXCOL],sumkern,DISTtemp,print;

    /*----- Parameters check ----------*/
    // Error if the number of days is greater than SIZEMAX
    // or if the number of features is greater than NBFEATMAX
    if(nval*nvar>=SIZEMAX){return;}
    
	nrand = _nrand[0];

    nbk=(int)(NBK[0]);
    if(nbk>=MAXCOL){return;}
    
    // Half temporal window
    winlag=WINS[0]/2;if(winlag<=0){return;}

    // Number of years in DMAT and SMAT 
    NbAnn = (int)(nval/365.25)+1;

	// Print messages
	print=PRINTMESSAGE[0];

	if(print!=0.0) printf("\n\tKNN parameters: NBK=%2d WIN=%2.0f DMAT[%5d %2d],NVALSIM=%5d ",
			nbk,WINS[0],nval,nvar,nrand);

    // Normalisation of weights
    SomW=0;
    for(dmatCol=0;dmatCol<nvar;dmatCol++){SomW+=fabs(WEI[dmatCol]);}

    if(print!=0.0) printf("\n\tVariables weights:\n\t");
    if(SomW>0){
      for(dmatCol=0;dmatCol<nvar;dmatCol++){
        WEI[dmatCol]=fabs(WEI[dmatCol])/SomW;
        if(print!=0.0) printf("%3.1f ",WEI[dmatCol]);
      }
    }

    // Nb de jours sélectionnés
    nbjDMATSELEC=1;
    for(day=0;day<nval;day++){if(DMATSELEC[day]>0){nbjDMATSELEC++;}}
    if(print!=0.0) printf("\n\tPercentage of selected days: %6d over %6d (%2.0f perc)",
			nbjDMATSELEC,nval,(double)(nbjDMATSELEC)/(double)(nval)*100);

    if(DMATSELEC[(int)(NNdep[0])]<=0 && print!=0.0 ){
		printf("CAUTION ! The starting day is not a selected day >> problem with the resampling algo.");
	}
    
    /*----- Initialisation des caractéristiques ----------*/
    nnSelec=(int)(NNdep[0]);if((nnSelec<0)|(nnSelec>=nval)){nnSelec=0;}
    for(dmatCol=0;dmatCol<nvar;dmatCol++){CARAC[dmatCol]=DMAT[nnSelec+nval*dmatCol];}
    
    /*----- Resampling kernel ----------*/
    sumkern=0;for(nnCol=0;nnCol<nbk;nnCol++){sumkern+=1/(double)(nnCol+1);}
    KERN[0]=1/sumkern;

    if(print!=0.0) printf("\n\tKernel = %5.3f ",KERN[0]);

    for(nnCol=1;nnCol<nbk;nnCol++){
      KERN[nnCol]=KERN[nnCol-1]+1/(double)(nnCol+1)/sumkern;
      if(print!=0.0) printf("%5.3f ",KERN[nnCol]);
    }
    if(print!=0.0) printf("\n\tLine nb:\n\t");
        
    /*----- Resampled data generation ----------*/
    for(day=0;day<nrand;day++){    
        if(fmod(day,1000)==0 && print!=0.0){printf("%5d ",day);}
        if(fmod(day,8000)==0 && print!=0.0){printf("\n\t");}
                
        // initialize the distance vector
        for(nnCol=0;nnCol<nbk;nnCol++){DIST[nnCol]=MAXVALDIST;}

        // Calculate the distance with the value of the descriptors
        // calculate only if the current date falls within the time window
        for(ann=-1;ann<NbAnn;ann++){

            // Start and end of the temporal window for each year
            debDay=(int)(ann*365.25-winlag+fmod(nnSelec,365.25));
            finDay=(int)(ann*365.25+winlag+fmod(nnSelec,365.25));
            if(debDay<0){debDay=0;}
            if(finDay<0){finDay=0;}   
            if(debDay>=nval){debDay=nval-1;} 
            if(finDay>=nval){finDay=nval-1;}    
                    
            for(day2=debDay;day2<finDay;day2++){

                // Parcours la base de données en ne retenant que les jours sélectionnés
                if(DMATSELEC[day2]>0){
                    DISTtemp=0;

                    // Weighted euclidian distance
                    for(dmatCol=0;dmatCol<nvar;dmatCol++){
                        DISTtemp+=WEI[dmatCol]*pow(DMAT[day2+nval*dmatCol]-CARAC[dmatCol],2);
                    }

                    // Check if the distance is lower than one of the already stored distance
                    nnClass=0;
                    while((DISTtemp>DIST[nnClass])&(nnClass<=nbk)){nnClass++;}

                    // if yes, rearrange DIST and NNliste vectors 
                    if(nnClass<nbk){
                        for(nnCol=nbk-1;nnCol>=nnClass+1;nnCol--){
                            DIST[nnCol]=DIST[nnCol-1];NNliste[nnCol]=NNliste[nnCol-1];
                        }
                        NNliste[nnClass]=day2;DIST[nnClass]=DISTtemp;
                    }

                } 
            }
        }
        
        // Random number
        rnd=RND[day];if((rnd<0)|(rnd>1)){rnd=0.5;}

        // Selection of the nearest neighbour
        nnCol=0;while((rnd>KERN[nnCol])&(nnCol<nbk)){nnCol++;}
        nnSelec = (int)(NNliste[nnCol]+1);
        if(nnSelec>=nval){nnSelec=nval-1;}
        
        // Stockage des caractéristiques du jour
        for(dmatCol=0;dmatCol<nvar;dmatCol++){CARAC[dmatCol]=DMAT[nnSelec+nval*dmatCol];}

        // Attribution of the data to KNNSIM matrix
        KNNSIM_NN[day]=(double)(nnSelec);
        //printf(" >> %5.0f\n",KNNSIM_NN[day]);
        
    } // End of the simulation
    
     if(print!=0.0) printf("\n\n");    
}
