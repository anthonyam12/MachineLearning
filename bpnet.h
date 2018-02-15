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

#ifndef NN_H
#define NN_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

/** Base class.
  This is a simple backpropagation neural network class that
  can be used as a function approximator for RLlib.
  I have kept it simple so that it is easy to use and debug. */
class bpnet
{
  
public:
  
  enum activation_function{Sigmoid,TanH};
  
protected:
  // steepness of the transfer function  
  double THETA;        
  // step size for learning
  double STEP;          
  // momentum term for learning
  double MOMENTUM;      
  // boolean-linear transfer function on output layer
  int LINEAR_OUTPUT; 
  // minimun value produced by the transfer function
  double xfermin; 
  // maximun value produced by the transfer function
  double xfermax; 
  // transfer function type
  activation_function activation_function_type; 
  // the output activations of the neurons
  double **activation;  
  // results from the weight*input calculations
  double **dotprod;     
  // the network weights
  double ***weights;   
  // derivatives
  double **sigma;       
  // changes to applied to the weights
  double ***delta;     
  // number of layers
  int numlayers;       
  // output scaling factors
  double outmax, outmin;
  // sizes for each layer
  int *size;           
  
  // a helper function
  double scale(double x);
  // a helper function
  double unscale(double x);
  // a helper function
  double xferfunc(double x,double theta);
  // a helper function
  double xferfuncprime(double xprime,double theta);  
  // a helper function
  void setdefaults(){
    THETA = 1.0;
    STEP = 0.01;
    MOMENTUM = 0.0;
    outmax = 1.0; 
    outmin = -1.0; 
    activation_function_type = Sigmoid;
    xfermin = 0;
    xfermax = 1;
    LINEAR_OUTPUT = 0;
  }
  
  // a helper function
  void new_all();
  // a helper function
  void delete_all();
  
public:
  
  // You shouldn't use this.  Use the other constructor. 
  bpnet()
    {
      activation = NULL;
      weights = NULL;
      sigma = NULL;
      delta = NULL;
      size = NULL;
      dotprod = NULL;
      numlayers = 0;
      setdefaults();
   };
  
  // Create a neural network by specifying the number of layers and the 
  // number of nodes in each layer.  Bias nodes are added automagically.
  bpnet(int layers ...);
  
  ~bpnet();
  
  // The output range scaling is also applied to the errors that
  // you give for backpropagation.
  void set_output_range(double min, double max)
    {
      outmax = max; 
      outmin = min; 
    }
  
  // I have not gotten around to input scaling.
  // Just make sure all your inputs have roughly the same range.
  // void set\_input\_range(double min, double max){}
  
  // theta is the "steepness" of the transfer function.  
  // values between 0.25 and 2.0 seem to work.  Start with
  // theta = 1 and then try other values.
  
  void set_theta(double th)
    {
      THETA = th;
    }
  
  double get_theta()
    {
      return THETA;
    }
  
  // You can set the step size.  Small values between 0.1 and 0.001
  // seem to work in most cases.  You can start with a large step
  // like 0.5 and reduce it time goes by.
  
  void set_step(double st)
    {
      STEP = st;
    }
  
  double get_step()
    {
      return STEP;
    }
  
  // Momentum seems to help in most cases.  Try learning
  // with a small momentum around 0.1
  
  void set_momentum(double mo)
    {
      MOMENTUM = mo;
    }
  
  double get_momentum()
    {
      return MOMENTUM;
    }
  
  // Sometimes you need a linear transfer function on the output layer.
  
  void set_linear()
    {
      LINEAR_OUTPUT = 1;
    }
  
  void set_nonlinear()
    {
      LINEAR_OUTPUT = 0;
    }
  
  // You can choose between Sigmoid and Hyperbolic Tangent transfer
  // functions.  Pretty standard stuff.
  
  void set_activation_function(activation_function af)
    {
      activation_function_type = af;
      
      switch(af)
	{
	case Sigmoid:
	  xfermin = 0.0;
	  xfermax = 1.0;
	  break;
	  
	case TanH:
	  xfermin = -1.0;
	  xfermax = 1.0;
	  break;
	  
	default:
	  cerr<<"unknown transfer function type\n";
	  exit(1);
	}
    }
  
  // get the value of one of the outputs 
  
  double getoutput(int i) 
    {
      return activation[numlayers-1][i];
    };
  
  // perform backpropagation using the specified errors 
  void backprop(double *errors);
  
  // run the network in feedforward mode and calculate the outputs 
  void evaluate(double *input_vector,double *output_vector);
  
  // The I/O routines may not be what you want, feel free to change them.
  friend ostream& operator<<(ostream& l,bpnet& r);
  
  void savenet(FILE *out);
  void savenet_ascii(ostream &l);
  void loadnet(FILE *in);
  void loadnet_ascii(istream &l);
  
};


#endif




