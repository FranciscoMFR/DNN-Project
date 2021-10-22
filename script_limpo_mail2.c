#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include "mnist.h"

#define NPAT 100
#define INNODES 784
#define HIDNODES 50
#define OUTNODES 10
#define EPOCHS 10
#define BATCHSIZE 20
#define randomize()((double) rand() / ((double) RAND_MAX + 1))
#define printArray(arr) printArray_((arr), sizeof(arr)/sizeof(arr[0]))

void printArray_(double *a, int len) {
    for (int i = 0; i < len; i++) printf("ARRAY_CHECK:%d\n", len);
}


void allocate_mem(double ** * arr, int n, int m) {
    * arr = (double ** ) malloc(n * sizeof(double * ));
    for (int i = 0; i < n; i++) {
        ( * arr)[i] = (double * ) calloc(m, sizeof(double));
    }
}

void allocate_mem_int(int ** * arr, int n, int m) {
    * arr = (int ** ) malloc(n * sizeof(int * ));
    for (int i = 0; i < n; i++) {
        ( * arr)[i] = (int * ) calloc(m, sizeof(int));
    }
}

int main() {

    load_mnist();
    srand(time(NULL));
    int *  patIndex;
    patIndex = malloc((NPAT) * sizeof(int));
    int randomPat[BATCHSIZE];
    int numPattern = NPAT;
    int numInput = INNODES;
    int numHidden = HIDNODES;
    int numOutput = OUTNODES;
    int epochs = EPOCHS;
    int nBatchs = NPAT/BATCHSIZE;
    double * Target;
    Target = (double * ) calloc(10, sizeof(double));
    double( * inputLayer)[784] = train_image;
    
    double * inputHidden;
    inputHidden = malloc((HIDNODES) * sizeof(double));
    double ** weightIH;
    allocate_mem( & weightIH, INNODES, HIDNODES);
    
    double* hiddenLayer;
    hiddenLayer = malloc(HIDNODES*sizeof(double));

    double * inputOutput;
    inputOutput = malloc((OUTNODES) * sizeof(double));    

    double ** weightHO;
    allocate_mem( & weightHO, HIDNODES, OUTNODES);
    
    double * outputLayer;
    outputLayer = malloc((OUTNODES) * sizeof(double));
    
    double * biasIH;
    biasIH = malloc((HIDNODES) * sizeof(double));
    
    double * sumdbiasIH;
    sumdbiasIH = malloc((HIDNODES) * sizeof(double));
    
    double * biasHO;
    biasHO = malloc((OUTNODES) * sizeof(double));

    double * sumdbiasHO;
    sumdbiasHO = malloc((OUTNODES) * sizeof(double));

    double * sumdOutput;
    sumdOutput = malloc((OUTNODES) * sizeof(double));
    
    //double * sumdwOsum;
    //sumdwOsum = malloc((HIDNODES) * sizeof(double));
    
    double * sumdHidden;
    sumdHidden = malloc((HIDNODES) * sizeof(double));
    
    double ** sumdweightIH;
    allocate_mem( & sumdweightIH, INNODES, HIDNODES);
    
    double ** sumdweightHO;
    allocate_mem( & sumdweightHO, HIDNODES, OUTNODES);

    double Error;
    double eta = 0.1;
    double alpha = 0.0;//0.9;
    double smallwt = 0.5;
    double sumdwOsum_1;
    int batchStart;
    int batchEnd;

    for (int j = 0; j < numHidden; j++) {
        /* Inicializa os pesos e d's da IH */
        for (int i = 0; i < numInput; i++) {
            sumdweightIH[i][j] = 0.0;
            weightIH[i][j] = 2.0 * (randomize() - 0.5) * smallwt;
            biasIH[j] = 2.0 * (randomize() - 0.5) * smallwt;
        }
    }
    for (int k = 0; k < numOutput; k++) {
        /* Inicializa os pesos e d's da HO  */
        for (int j = 0; j < numHidden; j++) {
            sumdweightHO[j][k] = 0.0;
            weightHO[j][k] = 2.0 * (randomize() - 0.5) * smallwt;
            biasHO[j] = 2.0 * (randomize() - 0.5) * smallwt;
        }
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {    
    /* iteração pelo nº de epochs */

        int tm = (int)time(NULL);
        int size = NPAT;
        double max_Error_h = 0.0;
        double max_outputLayer[numOutput];
        double max_Target[numOutput];
        double Error_class[numOutput];
        int nex_class[numOutput];
        int max_h;
        for (int k=0; k<numOutput; k++) {
            nex_class[k]=0;
            Error_class[k]=0.0;
        }
	for (int h = 0; h < NPAT; h++) {
             /* array que irá baralhar os exemplos  */
             patIndex[h] = h;
    	}
        
        for (int batchNum = 0; batchNum < nBatchs; batchNum++) {
        /* Aplicação a todos os exemplos de treino */

	     for (int h =0; h < BATCHSIZE; h++) {
            	  int nh;
		  if (size > 0) {
		     nh  = rand() % size +0; //size-> max || 0-> min
		  }		  
		  else {
		     nh = 0;
		  }
		  randomPat[h] = patIndex[nh];
		  printf("nh=%d->patIndex[random]=%d->randomPat[h]=%d->size=%d\n",nh,patIndex[nh],randomPat[h],size);
		  patIndex[nh] = patIndex[size];
		  patIndex[size] = -1;
		  size--;
		  
		  //continue;
             }
	    // printf("\n");
	     /*for (int aux=0; aux<BATCHSIZE ; aux++){
	     
	         printf("HERE:%d\n",randomPat[aux]);

	     }*/
	     //break;
	     //printArray(randomPat);
	     double batchError = 0.0;

	     for (int nh = 0; nh < BATCHSIZE; nh++) {           
            	  //printf("oi");
		  int h = randomPat[nh];
		  
		  Target[train_label[h]] = 1.0;
		  double Error_h = 0.0;
            
            	  for (int j = 0; j < numHidden; j++) {
                       inputHidden[j] = biasIH[j];
                       for (int i = 0; i < numInput; i++) {
                    	    inputHidden[j] += inputLayer[h][i] * weightIH[i][j]; 
                       }
                       hiddenLayer[j] = 1.0 / (1.0 + exp(-inputHidden[j])); 					/*funcão de ativação sigmoide */
                  }

                  for (int k = 0; k < numOutput; k++) {
                       inputOutput[k] = biasHO[k];
                       for (int j = 0; j < numHidden; j++) {
                            inputOutput[k] += hiddenLayer[j] * weightHO[j][k]; 
                       }
                       outputLayer[k] = 1.0 / (1.0 + exp(-inputOutput[k])); 					/* funcão de ativação sigmoide */
                       Error_h +=  (outputLayer[k]- Target[k] ) * (outputLayer[k]- Target[k]);
                       sumdOutput[k] += 2*(outputLayer[k]- Target[k] ) * outputLayer[k] * (1.0 - outputLayer[k]); //V

                  }
                  batchError += Error_h;
                  Error_class[train_label[h]] += Error_h;
                  nex_class[train_label[h]] ++;
            
            
                  if (Error_h>max_Error_h) {
                      max_Error_h = Error_h;
                      max_h = h;
                      for (int k=0; k<numOutput; k++) {
                           max_outputLayer[k] = outputLayer[k];
                           max_Target[k] = Target[k];
                      }
                  }
            
                  for (int j = 0; j < numHidden; j++) {
                  /* Backpropagation */
                       //sumdwOsum[j] = 0.0;
		       sumdwOsum_1 = 0.0;
                       for (int k = 0; k < numOutput; k++) {
                            //sumdwOsum[j] += weightHO[j][k] * sumdOutput[k]; //V
			    sumdwOsum_1 += weightHO[j][k] * sumdOutput[k];
                       }
                       //sumdHidden[j] += sumdwOsum[j] * hiddenLayer[j] * (1.0 - hiddenLayer[j]); //V
			sumdHidden[j] += sumdwOsum_1 * hiddenLayer[j] * (1.0 - hiddenLayer[j]);
                  }
                  for (int j = 0; j < numHidden; j++) {

                       sumdbiasIH[j] += (1.0-alpha)*eta * sumdHidden[j] + alpha * sumdbiasIH[j]; //V

                       for (int i = 0; i < numInput; i++) {
                            sumdweightIH[i][j] += (1.0-alpha)*eta * inputLayer[h][i] * sumdHidden[j] + alpha * sumdweightIH[i][j];

                       }
                  }
                  for (int k = 0; k < numOutput; k++) {

                       sumdbiasHO[k] = (1.0-alpha)*eta * sumdOutput[k] + alpha * sumdbiasHO[k]; //V

                       for (int j = 0; j < numHidden; j++) {
                            sumdweightHO[j][k] += (1.0-alpha)*eta * hiddenLayer[j] * sumdOutput[k] + alpha * sumdweightHO[j][k]; //V

                       }
 
                  }
        
                  Target[train_label[h]] = 0.0;
             }
	     

	     batchError = batchError/BATCHSIZE;

	     for (int j = 0; j < numHidden; j++) {
                  /* update aos pesos IH */

                  biasIH[j] -= sumdbiasIH[j] / BATCHSIZE;
                  for (int i = 0; i < numInput; i++) {

                       weightIH[i][j] -= sumdweightIH[i][j] / BATCHSIZE;
                  }
             }
	     for (int k = 0; k < numOutput; k++) {
                  /* update aos pesos HO */

                  biasHO[k] -= sumdbiasHO[k] / BATCHSIZE;
                  for (int j = 0; j < numHidden; j++) {

                       weightHO[j][k] -= sumdweightHO[j][k] / BATCHSIZE;
                  }
 
             }

	     //RESETAR VALORES DOS ARRAYS
        
      /*  printf("EPOCH:%d -> <ERROR>=%f    elapsed time:%ds\noutput with maximum error, %d:\n",epoch, batchError,(int)time(NULL)-tm,max_h);
        for( int a = 0 ; a < numOutput ; a++) {
            printf("OUTPUT:%.4f------TARGET:%f\n",max_outputLayer[a],max_Target[a]);
        }*/

       /* printf("\nMean Error per class:\n");
        for( int a = 0 ; a < numOutput ; a++) {
            printf("%d         ",nex_class[a]);
        }*/
        /*printf("\n");
        for( int a = 0 ; a < numOutput ; a++) {
            printf("%f  ",Error_class[a]/nex_class[a]);
        }
        printf("\n");
        
        
        if (Error/numPattern < 0.0000004) {
            printf("\nfinish\n");
            break; 												// condição de paragem da aprendizagem 
        }*/

	     printf("Epoch:%d\n",epoch);
	     printf("BATCH-%d/%d    ERROR:%f\n",batchNum,nBatchs,batchError);        
        }

    }
    return 1;
}
