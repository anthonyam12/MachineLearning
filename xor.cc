#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <bpnet.h>

void init_rng()
{
  FILE *seedfile;
  unsigned long int seed;
  if((seedfile=fopen("/dev/urandom","r"))==NULL)
    {
      perror("unable to open /dev/urandom");
      exit(1);
    }
  if(fread(&seed,sizeof(long int),1,seedfile)!=1)
    {
      perror("unable to read /dev/urandom");
      exit(1);
    }
  fclose(seedfile);
  // fprintf(stderr,"seed: %lu\n",seed);
  srand48(seed);
}

int main()
{
  double inputs[4][2] = {{0,0},{0,1},{1,0},{1,1},};
  double targets[4] = {0,1,1,0};
  double outputs[1];
  double errors[1];
  double sse;
  int i,j;

  init_rng();

  // Create neural network with three layers: 2 input nodes, 2 hidden, 1 out
  bpnet nn(3,2,2,1);
  nn.set_activation_function(bpnet::Sigmoid);
  nn.set_theta(1.0);
  nn.set_step(0.05);
  nn.set_momentum(0.05);

  // Run until sum squared error is 0.001 or less
  i=0;
  sse = 1.0;
  do
    {
      sse=0.0;
      for(j=0;j<4;j++)
	{
	  nn.evaluate(inputs[j],outputs);
	  errors[0] = targets[j] - outputs[0];
	  sse+=errors[0]*errors[0];
	  nn.backprop(errors);
	}
      i++;
      if(!(i % 1000))
	printf("%8d : %lf\n",i,sse);
    }while((sse > 0.001) && (i < 10000000));

  // Print table
  for(j=0;j<4;j++)
    {
      nn.evaluate(inputs[j],outputs);
      fprintf(stderr,"%lf %lf : %lf\n",inputs[j][0],inputs[j][1],outputs[0]);
    }
  
  return 0;
}
