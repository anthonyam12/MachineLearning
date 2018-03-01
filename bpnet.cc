/**********************************************************************/
/*                                                                    */
/*  Copyright (c) 1994                                                */
/*  Larry D. Pyeatt                                                   */
/*  Computer Science Department                                       */
/*  Colorado State University                                         */
/*                                                                    */
/*  Permission is hereby granted to copy all or any part of           */
/*  this program for free distribution.   The author's name           */
/*  and this copyright notice must be included in any copy.           */
/*                                                                    */
/*  Contact the author for commercial licensing.                      */


#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <my_misc.h>
#include <bpnet.h>
#include <stdio.h>
#include <iostream>
using namespace std;

static void scream_and_die(char *s)
{
  perror(s);
  exit(1);
}


/******************************************************************/
/* code for the neuron class                                      */


/****************************************************************/
/* Activation functions                                         */

/* define the activation (transfer) function and its derivative */

double bpnet::xferfunc(double x,double theta)
{
  switch(activation_function_type)
    {
    case Sigmoid:
      return 1.0 / (1.0 + exp(-(theta*x)));
      
    case TanH:
      return tanh(theta * x);
      
    default:
      cerr<<"unknown transfer function type\n";
      exit(1);
    }
}

double bpnet::xferfuncprime(double xprime,double theta)
{
  switch(activation_function_type)
    {
    case Sigmoid:
      return theta * (xprime * (1.0 - xprime));
      
    case TanH:
      return theta * (1 - xprime * xprime);
      
    default:
      cerr<<"unknown transfer function type\n";
      exit(1);
    }
}



/****************************************************************/
/* A function that returns a small random number                */
static double random_weight()
{
  return (ran1() - 0.5) / 1000.0;
}

/****************************************************************/
/* Methods for scaling the input and output data                */
double bpnet::scale(double x)
{
  return ((x - xfermin) / (xfermax - xfermin) * (outmax-outmin)) + outmin;
}

double bpnet::unscale(double x)
{
  //  return x;
  return ((x / (outmax - outmin)) * (xfermax - xfermin)); 
}


/****************************************************************/
/* Allocate storage for the bpnet                               */
void bpnet::new_all()
{
  int i,j,k;
  
  /* allocate storage for the layers */
  activation = new double*[numlayers];
  dotprod = new double*[numlayers];
  weights = new double**[numlayers-1];
  sigma = new double*[numlayers];
  delta = new double**[numlayers-1];
  
  for(i = 0 ; i < numlayers ; i++)
    {
      if(i == 0)
        /* we don't want to spend time copying data, so we just set */
        /* the pointer to the input layer every time we evaluate    */
        dotprod[i] = activation[i] = NULL;   
      
      else
        {
          activation[i] = new double[size[i]];
          dotprod[i] = new double[size[i]];
        }
      
      sigma[i] = new double[size[i]];
      
      if(i < (numlayers - 1))
        {
          weights[i] = new double*[size[i+1]];
          delta[i] = new double*[size[i+1]];
	  
          for(j = 0 ; j < size[i+1] ; j++)
            {  
	      // add one for the bias input 
              weights[i][j] = new double[size[i]+1];
              delta[i][j] = new double[size[i]+1];  
            }
        }
    }  
}

/****************************************************************/
/* Constructor for the backpropagation networks                 */
/* arguments:                                                   */
/*    number of layers,                                         */
/*    size of input layer,                                      */
/*    size of first hidden layer,                               */
/*    ...                                                       */
/*    size of output layer                                      */

bpnet::bpnet(int layers ...)
{
  va_list ap;
  int i, j, k;
  
  setdefaults();
  
  /* get the argument list */
  va_start(ap,layers);
  numlayers = layers;  
  size = new int[numlayers];
  
  for(i = 0 ; i < numlayers;i++)
    size[i] = va_arg(ap, int);
  
  va_end(ap);
  
  new_all();
  
  /* randomize the weights and set deltas to zero */
  for(i = 0 ; i < (numlayers - 1) ; i++)
    for(j = 0 ; j < size[i+1] ; j++)
      for(k = 0 ; k < size[i]+1 ; k++)  // +1 for the bias input 
	{
          weights[i][j][k] = random_weight();
          delta[i][j][k] = 0.0;
        }
}

/****************************************************************/
/* delete all storage associated with the bpnet                 */
void bpnet::delete_all()
{
  int i, j;
  
  for(i = 0 ; i < numlayers ; i++)
    {
      if((i != 0) && (activation != NULL) && (activation[i] != NULL))
        {
          delete[] activation[i];
          delete[] dotprod[i];
        }
      delete[] sigma[i];
      
      if(i < numlayers-1)
        {
          for(j = 0 ; j < size[i+1] ; j++)
            {
              delete[] weights[i][j];
              delete[] delta[i][j];
            }
          delete[] weights[i];
          delete[] delta[i];
        }
    }
  delete[] size; 
  delete[] activation;
  delete[] dotprod;
  delete[] weights;
  delete[] delta;
  delete[] sigma;
}

/****************************************************************/
/* Destructor for the backpropagation networks                  */
bpnet::~bpnet()
{
  delete_all();
};

/****************************************************************/
/* evaluate the network from inputs to outputs                  */
void bpnet::evaluate(double *input_vector,double *output_vector)
{
  int i, j, k;
  double temp;
  
  /* Don't copy input vector.  Just set the pointer  */
  activation[0] = input_vector;
  
  for(i = 0 ; i < numlayers - 1 ; i++)      // i = layer number
    //#pragma omp parallel for
    for(j = 0 ; j < size[i+1] ; j++)      // j = to node
      {
        dotprod[i+1][j] = 0.0;
	for(k = 0 ; k < size[i] ; k++)    // k = from node
          dotprod[i+1][j] += (weights[i][j][k] * activation[i][k]);
        dotprod[i+1][j] += weights[i][j][size[i]] * 1.0; // bias input 
        if((i < numlayers-2) || (!LINEAR_OUTPUT))
          activation[i+1][j] = xferfunc(dotprod[i+1][j],THETA);
        else
          activation[i+1][j] = dotprod[i+1][j] * THETA;         
      }
  
  for(i = 0 ; i < size[numlayers-1] ; i++)
    output_vector[i] = scale(activation[numlayers-1][i]);
}

/****************************************************************/
/* use the array of errors to update the weights                */
void bpnet::backprop(double *errors)
{
  int i, j, k;
  double error;
  
  /* backpropagate error and calculate sigma for every neuron */
  for(i = 0 ; i < size[numlayers-1] ; i++)
    if(LINEAR_OUTPUT)
      sigma[numlayers-1][i] = unscale(errors[i]) * activation[numlayers-1][i] / THETA;
    else
      sigma[numlayers-1][i] = 
	unscale(errors[i]) * xferfuncprime(activation[numlayers-1][i],THETA);
  
  for(i = numlayers-1 ; i > 1; i--)
    //#pragma omp parallel for
    for(j = 0 ; j < size[i-1] ; j++)  
      {
        error = 0.0;
        for(k = 0 ; k < size[i] ; k++)
          error += sigma[i][k] * weights[i-1][k][j];        
        sigma[i-1][j] = error * xferfuncprime(activation[i-1][j],THETA);
      }
  
  /* calculate delta weights, and update the weights */
  
  for(i = 0 ; i < (numlayers - 1) ; i++)
    {
      //#pragma omp parallel for
      for(j = 0 ; j < size[i+1] ; j++)
        {
          for(k = 0 ; k < size[i] ; k++)
            weights[i][j][k] += 
              (delta[i][j][k] = STEP * sigma[i+1][j] * activation[i][k]);
	  delta[i][j][size[i]] = STEP * sigma[i+1][j];
	  weights[i][j][size[i]] += delta[i][j][size[i]];
        }
    }
}



/****************************************************************/
/* Some I/O routines  */

ostream& operator<<(ostream &l,bpnet &r)
{
  int i, j;
  l << "input nodes:";
  
  for(i = 0 ; i < r.size[0] ; i++)
    l<<" "<<r.activation[0][i];
  
  l << "\n";
  
  for(i = 1 ; i < r.numlayers-1 ; i++)
    {
      l << "layer " << i << " nodes:";
      
      for(j = 0 ; j < r.size[i] ; j++)
        l << " " << r.activation[i][j];
      
      l << "\n";
    }
  
  l << "output nodes:";
  
  for(i = 0 ; i < r.size[r.numlayers-1] ; i++)
    l << " " << r.activation[r.numlayers-1][i];
  
  l << "\n";
  return l;
};

void bpnet::savenet_ascii(ostream &l)
{
  int i, j, k;
  
  l << numlayers;
  
  for(i = 0 ; i < numlayers ; i++)
    l << " " << size[i];
  
  for(i = 0 ; i < numlayers-1 ; i++)
    {
      l << "\n";
      
      for(j = 0 ; j < size[i+1] ; j++)
        {      
          l << "\n";
	  
          for(k = 0 ; k < size[i]+1 ; k++)
            {
              l.precision(10);
              l << weights[i][j][k] << " ";
            }
        }
    }
  l << "\n";
};

void bpnet::loadnet_ascii(istream &l)
{
  int i, j, k;
  
  l >> numlayers;
  
  for(i = 0 ; i < numlayers ; i++)
    l >> size[i];
  
  for(i = 0 ; i < numlayers-1 ; i++)
    for(j = 0 ; j < size[i+1] ; j++)
      for(k=0 ; k < size[i]+1 ; k++)
        l >> weights[i][j][k];
};

void bpnet::savenet(FILE *out)
{
  int i, j;
  
  fwrite(&numlayers,sizeof(int),1,out);
  fwrite(size,sizeof(int),numlayers,out);
  fwrite(&THETA,sizeof(double),1,out);
  fwrite(&STEP,sizeof(double),1,out);
  fwrite(&MOMENTUM,sizeof(double),1,out);
  fwrite(&LINEAR_OUTPUT,sizeof(int),1,out);
  fwrite(&xfermin,sizeof(double),1,out);
  fwrite(&xfermax,sizeof(double),1,out);
  fwrite(&activation_function_type,sizeof(activation_function),1,out);
  fwrite(&outmax,sizeof(double),1,out);
  fwrite(&outmin,sizeof(double),1,out);

  for(i = 0 ; i < numlayers-1 ; i++)
    for(j = 0 ; j < size[i+1] ; j++)
      fwrite(weights[i][j],sizeof(double),size[i]+1,out);
};

void bpnet::loadnet(FILE *infile)
{
  int i, j, mismatch;
  int *insize;
  
  // get rid of everything if we have already allocated
  delete_all();
  
  if(fread(&numlayers,sizeof(int),1,infile)!=1)
    scream_and_die((char*)"bpnet::loadnet");
  size = new int[numlayers];
  if(fread(size,sizeof(int),numlayers,infile)!=numlayers)
    scream_and_die((char*)"bpnet::loadnet");
  if(fread(&THETA,sizeof(double),1,infile)!=1)
    scream_and_die((char*)"bpnet::loadnet");
  if(fread(&STEP,sizeof(double),1,infile)!=1)
    scream_and_die((char*)"bpnet::loadnet");
  if(fread(&MOMENTUM,sizeof(double),1,infile)!=1)
    scream_and_die((char*)"bpnet::loadnet");
  if(fread(&LINEAR_OUTPUT,sizeof(int),1,infile)!=1)
    scream_and_die((char*)"bpnet::loadnet");
  if(fread(&xfermin,sizeof(double),1,infile)!=1)
    scream_and_die((char*)"bpnet::loadnet");
  if(fread(&xfermax,sizeof(double),1,infile)!=1)
    scream_and_die((char*)"bpnet::loadnet");
  if(fread(&activation_function_type,sizeof(activation_function),1,infile)!=1)
    scream_and_die((char*)"bpnet::loadnet");
  if(fread(&outmax,sizeof(double),1,infile)!=1)
    scream_and_die((char*)"bpnet::loadnet");
  if(fread(&outmin,sizeof(double),1,infile)!=1)
    scream_and_die((char*)"bpnet::loadnet");
  new_all();
  
  for(i = 0 ; i < numlayers-1 ; i++)
    for(j = 0 ; j < size[i+1] ; j++)
      if(fread(weights[i][j],sizeof(double),size[i]+1,infile)!=size[i]+1)
	scream_and_die((char*)"bpnet::loadnet");
}

