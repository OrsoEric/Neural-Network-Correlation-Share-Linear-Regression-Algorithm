/****************************************************************
**	OrangeBot Project
*****************************************************************
**	      *
**	     *
**	    *  *
**	******* *
**	         *
*****************************************************************
**	Correlation Share Linear Regression
****************************************************************/

/****************************************************************
**	HISTORY VERSION
****************************************************************
**  V1.0
**      Worked out the math behind correlation share algorithm
**      Completed trial program to test out the math
**      Implemented factor H. Correlation contrast.
****************************************************************/

/****************************************************************
**	KNOWN BUGS
****************************************************************
**      BUG1 (V1.0) - MATH
**  Fixed math of the correlation share algorithm
****************************************************************/

/****************************************************************
**	DESCRIPTION
****************************************************************
**  I want to apply linear regression to a neuron with multiple inputs
**
**  NAIVE MATH:
**  Give all inputs an equal share of importance
**  m = index of the input. M inputs.
**  g[m], b[m] = gain and bias calculated by linear regression
**  neuron_weight[m] = g[m] / M
**  neuron_bias = SUM(j)( b[j] ) /M
**
**  PROBLEM:
**  This is not a cool way of doing it.
**  If one input is white noise, another is the perfecttly correlated signal
**  for the output, the noise will still be injected in the output.
**
**  CORRELATION SHARE ALGORITHM:
**  correlation only requires one more variable to the linear regression
**  problem is that correlation has its problems
**  rather than using the absolute value of correlation, i use a relative
**  value compared to other correlations.
**  it's actually computationally cheap and gives a way for the algorithm
**  to tell the difference between one input and another
**
**  The overall idea is to promote linear regression gain and bias from
**  inputs with high correlation share, and penalize input with low
**  correlation shares by attenuating their bias and gain contribution
**
**  This algorithm gives for free a weight decay algorithm. If an input is
**  poorly correlated with the output, the weight will decay to zero and the bias
**  will not contriute to the overall bias of the neuron
**
**  MATH:
**  x[m]        = input m of the neuron. M inputs in total.
**  y           = output of the neuron. training output is used to calculate weight and biases
**
**  x_avg[m]    = average of input signal m
**  x2_avg[m]   = average of square of input signal m
**  y_avg       = average of training output signal
**  y2_avg      = average of square of training output signal
**  xy_avg[m]   = average of dot product between input m and training output
**
**  variance of input m
**  var[m] = x2_avg[m] -x_avg[m] *x_avg[m]
**
**  standard deviation of input signal m
**  std[m] = sqrt(var[m])
**
**  covariance of input m and training output
**  covar[m] = xy_avg[m] -x_avg[m] *y_avg[m]
**
**  linear regression computes optimal gain and bias to transform
**  input m to a signal with minimum square error to training output
**  g[m] = covar[m] / var[m]
**  b[m] = y_avg -x_avg[m] *g[m]
**  what linear regression is trying to optimize for
**  y = b[m] +x[m] *g[m]
**
**  partial correlation share factor. There is a lot of math behind this
**  F[m] = covar[m] *PROD(j!=m)(std[j])
**  elevate the partial correlation factor to an high power (H = 2, 4, 8, ...)
**  H should be even to have only positive factors. sign is alreay stored in gain.
**  this enchance the contrast between correlation and is better at muting uncorrelated inputs.
**  going too far with correlation share contrast will give too much importance to correlation
**  F[m] = F[m]^H
**
**  correlation share coefficient
**  because of how the math works, the sum of all Fi will be 1.
**  it has to be this way to allow the linear regression averages still work
**  Fi[m] = F[m] / SUM(j)(F[j])
**
**  Final weight and bias calculation
**  neuron_weight[m] = g[m] *Fi[m]
**  neuron_bias = SUM(j)( b[j] *Fi[j] )
**
**  NOTE: The naive case is a special case where Fi[m] = 1/M constant for all inputs
**  this happens when all inputs are equally correlated with the trainingoutput
****************************************************************/

/****************************************************************
**	TODO
****************************************************************
**
****************************************************************/

/****************************************************************
**	INCLUDE
****************************************************************/

#include <stdio.h>
#include <stdlib.h>     //for rand
#include <stdint.h>
#include <math.h>       //for sqrt
#include <time.h>       //for random seed

#include "debug.h"      //Debug macros

//make a number positive
#define ABS( x )    \
    (((x)<0)?(-(x)):(x))

//number of samples. Will become number of patterns
#define NUM_SAMPLES     100
//Number of inputs
#define NUM_INPUTS      3
//Number of iterations for TRIAL5
#define NUM_ITER        1000

/**********************************************************************************
**	TYPEDEF
**********************************************************************************/

/**********************************************************************************
**	PROTOTYPE: STRUCTURE
**********************************************************************************/

/****************************************************************
**	GLOBAL VARIABILE
****************************************************************/

/****************************************************************
**	FUNCTIONS
****************************************************************/

//Generate a random number
extern double rand_double( double fmin, double fmax );
//fill a vector with a line
extern int fill_line( double *x, int num_samples, double x_start, double x_end, double gain, double bias );
//add noise to a vector
extern int add_noise( double *x, int num_samples, double noise );
//Calculate the average and the average of the square of the vector
extern int calc_line_stats( double *x, int num_samples, double *x_avg, double *x2_avg );
//Calculate the average of the dot product of two vectors
extern int calc_cross_line_stats( double *x, double *y, int num_samples, double *xy_avg );
//Calculate the optimal linear transformation between input and output
extern int linear_regression( double *x, double *y, int num_samples, double *gain, double *bias );
//Calculate the ratio between input-output correlations
extern int calc_correlation_share ( double **x, double *y, int num_inputs, int num_samples, double *xy_corr );
//Execute a neuron given a vector of vectors of inputs avector of gains and a bias.
extern int execute_neuron( double **x, double *y, int num_inputs, int num_samples, double *gain, double bias );
//Compute the RMS distance between two vectors
extern double calc_error( double *x, double *y, int num_samples );
//Print a number in engineering notation
extern int eng_num( char *str, float num, char *unit );
//Execute a trial to show difference between naive linear regression and correlation share powered linear regression
extern int execute_trial( double noise_a, double noise_b, double noise_c, double noise_y, double *err, double *err_corr_share );

/****************************************************************
**	MAIN
****************************************************************
**	INPUT:
**	OUTPUT:
**	RETURN:
**	DESCRIPTION:
****************************************************************/

int main()
{
	///----------------------------------------------------------------
	///	STATIC VARIABILE
	///----------------------------------------------------------------

	///----------------------------------------------------------------
	///	LOCAL VARIABILE
	///----------------------------------------------------------------

	//iteration counter
	int cnt_iter;
    //Error of the two versions of the gain and bias estimation algorithm
    double err_naive;
    double err_corr_share;
    //String for engineering notation
    char str[10];
    //Random Seed
    time_t seed;
    //Error profiling for trial 5
    double tmp;
    double err_avg;
    double err_max;

	///----------------------------------------------------------------
	///	CHECK AND INITIALIZATIONS
	///----------------------------------------------------------------

	//Start Debugging
	DSTART( 0 );
	//Enter main
	DENTER();

        ///Random Init
    //Calculate the random seed
    seed = time(NULL);
    //If something interesting happened in this run, you can recover the seed in the 'debug.log' and rerun it
    DPRINT("seed: %ld\n", seed);
    //Initialize random generator to the seed
	srand(seed);

	///----------------------------------------------------------------
	///	BODY
	///----------------------------------------------------------------

	DPRINT("Orangebot Forever!\n");

	printf("The trial generates three random input lines and one random output line\n");
	printf("A neuron tries to use the three input lines to recreate the output line \n\n");

	printf("A controlled amount of noise is inserted in the lines to see the performance of the estimation algorithms\n\n");

	printf("Two algorithms are used to compute weights and bias\n");
	printf("Algorithm 1 is a linear regression estimation in which all inputs contribute the same.\n");
	printf("Algorithm 2 uses correlation share to give more importance to input-output pairs that are more correlated\n");
	printf("A lot of math was involved in this step\n");
	printf("For details about numbers and algorithm, read 'debug.log'\n\n");

	printf("After weights and bias have been found, the neuron is executed and the errors computed.\n\n");

	printf("\n-------------------------------------------------------------------------\n");
	printf("TRIAL1\n");
    printf("NO Noise in the inputs\n");
    printf("NO Noise in the training output\n");
    printf("Result:\n");
    printf("Both algorithms are exceedingly effective. Neurons are ideally suited to transform lines into lines\n");
    //Execute a trial with no noise in neither the inputs or the outputs
    execute_trial( 0.0, 0.0, 0.0, 0.0, &err_naive, &err_corr_share );
    //Print the error in engineering notation
    eng_num( str, err_naive, NULL );
    printf("error of equal share linear regression        : %s\n", str);
    //Print the error in engineering notation
    eng_num( str, err_corr_share, NULL );
    printf("error of correlation share linear regression  : %s\n", str);

    tmp = err_corr_share / (err_corr_share+ err_naive);
    eng_num( str, tmp, NULL );
    printf("error share of enchanted algorithm            : %s\n", str);

	printf("\n-------------------------------------------------------------------------\n");
	printf("TRIAL2\n");
    printf("Large amount of noise inserted in just the first input.\n");
    printf("NO Noise in the training output\n");
    printf("Result:\n");
    printf("The correlation share algorithm give a much lower share of the gain to input 1 because of noise\n");
    printf("This results in a much more accurate gains bias estimation and much lower noise\n");
    //Execute a trial with no noise in neither the inputs or the outputs
    execute_trial( 10.0, 0.0, 0.0, 0.0, &err_naive, &err_corr_share );
    //Print the error in engineering notation
    eng_num( str, err_naive, NULL );
    printf("error of equal share linear regression        : %s\n", str);
    //Print the error in engineering notation
    eng_num( str, err_corr_share, NULL );
    printf("error of correlation share linear regression  : %s\n", str);

    //Calculate error share of enchanted algorithm
    tmp = err_corr_share / (err_corr_share+ err_naive);
    eng_num( str, tmp, NULL );
    printf("error share of enchanted algorithm            : %s\n", str);

    printf("\n-------------------------------------------------------------------------\n");
	printf("TRIAL3\n");
    printf("Small but equal amount of noise everywhere.\n");
    printf("Result:\n");
    printf("Since no input-output pair has inherent better correlation, the two algorithms performs about the same\n");
    //Execute a trial with no noise in neither the inputs or the outputs
    execute_trial( 0.1, 0.1, 0.1, 0.1, &err_naive, &err_corr_share );
    //Print the error in engineering notation
    eng_num( str, err_naive, NULL );
    printf("error of equal share linear regression        : %s\n", str);
    //Print the error in engineering notation
    eng_num( str, err_corr_share, NULL );
    printf("error of correlation share linear regression  : %s\n", str);
    //Calculate error share of enchanted algorithm
    tmp = err_corr_share / (err_corr_share+ err_naive);
    eng_num( str, tmp, NULL );
    printf("error share of enchanted algorithm            : %s\n", str);

    printf("\n-------------------------------------------------------------------------\n");
	printf("TRIAL4\n");
    printf("Varying levels of noises in input, small noise in output.\n");
    printf("Result:\n");
    printf("This is a realistic condition, with varying performance on different inputs and output that is not a perfect line\n");
    printf("the correlation share algorithm performs much better than the naive equal share algorithm\n");
    //Execute a trial with no noise in neither the inputs or the outputs
    execute_trial( 10.0, 1.0, 0.1, 0.1, &err_naive, &err_corr_share );
    //Print the error in engineering notation
    eng_num( str, err_naive, NULL );
    printf("error of equal share linear regression        : %s\n", str);
    //Print the error in engineering notation
    eng_num( str, err_corr_share, NULL );
    printf("error of correlation share linear regression  : %s\n", str);
    //Calculate error share of enchanted algorithm
    tmp = err_corr_share / (err_corr_share+ err_naive);
    eng_num( str, tmp, NULL );
    printf("error share of enchanted algorithm            : %s\n", str);

    printf("\n-------------------------------------------------------------------------\n");
	printf("TRIAL5\n");
    printf("Random noise levels\n");
    printf("Calculate the average performance and worst performance of the enchanted algorithm\n");
    printf("Run the two algorithms a number of times.\n");

    //Don't show those trial in the debug
    DSHOW( 100 );

    //init errors
    err_avg = 0.0;
    err_max = 0.0;

    //For every iteration
    for (cnt_iter = 0; cnt_iter < NUM_ITER; cnt_iter++)
    {
        //Execute a trial with no noise in neither the inputs or the outputs
        execute_trial( rand_double(+0.1, +100.0), rand_double(+0.1, +10.0), +0.1, rand_double(+0.1, +1.0), &err_naive, &err_corr_share );
        //Print the error in engineering notation
        eng_num( str, err_naive, NULL );
        //Print the error in engineering notation
        eng_num( str, err_corr_share, NULL );
        //Share of the error committed by the enchanted algorithm
        tmp = err_corr_share / (err_corr_share+ err_naive);
        //Find worst error share
        if (tmp > err_max)
        {
            err_max = tmp;
        }
        //accumulate error share
        err_avg += tmp;
        //A trial has been computed
        printf(".");
    }   //End For every iteration
    //Normalize and obtain average error share
    err_avg /= (1.0 *cnt_iter);

    eng_num( str, err_avg, NULL );
    printf("\n\naverage error share of enchanted algorithm: %s\n", str);

    eng_num( str, err_max, NULL );
    printf("worst error share of the enchanted algorithm: %s\n\n", str);

    printf("error share <500m means that the enchanted algorithm is performing better than the naive algorithm.\n");
    printf("error share 500m means the two algorightms are performing at similar levels.\n");
    printf("error share >500m means that the enchanted algorithm is performing worse than the naive algorithm.\n");

	///----------------------------------------------------------------
	///	FINALIZATIONS
	///----------------------------------------------------------------

	//Return from main
	DRETURN();
	//Stop Debugging
	DSTOP();

    return 0;
}	//end function: main

/****************************************************************************
**	rand_double | double, double
*****************************************************************************
**	PARAMETER:
**	RETURN:
**	DESCRIPTION:
**	Generate a random double
****************************************************************************/

double rand_double( double fmin, double fmax )
{
	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

	//int random
	int irnd;
	//float random
	double frnd;

	///--------------------------------------------------------------------------
	///	CHECK AND INITIALIZATIONS
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------

	//Get a random number
	irnd = rand();
	//Convert directly into float.
	frnd = fmin +(fmax -fmin)*(1.0*irnd)/(1.0*RAND_MAX);

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

	return frnd;
}	//end function: rand_double | double, double


/****************************************************************************
**	fill_line
*****************************************************************************
**	PARAMETER:
**	RETURN:
**	DESCRIPTION:
****************************************************************************/

int fill_line( double *x, int num_samples, double x_start, double x_end, double gain, double bias )
{
	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

	//fast counter
	register int t;
	//
	double tmp;


	///--------------------------------------------------------------------------
	///	CHECK
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	INITIALIZATIONS
	///--------------------------------------------------------------------------

	DENTER_ARG( "xs: %+1.3f, xe: %+1.3f, g: %+1.3f, b: %+1.3f\n", x_start, x_end, gain, bias );

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------

	DPRINT("line: ");
	//For every sample
	for (t = 0;t < num_samples;t++)
	{
        //Calculate X
        tmp = x_start +(x_end -x_start)/(1.0* (num_samples -1)) *(1.0 *t);
        //Calculate Y
		x[t] = bias +tmp *gain;
		DPRINT_NOTAB( "%+1.3f | ", x[t] );

	}

	DPRINT_NOTAB("\n");

	///--------------------------------------------------------------------------
	///	FINALIZATIONS
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

	//Return
	DRETURN();

	return 0;
}	//end function: fill_line

/****************************************************************************
**	add_noise
*****************************************************************************
**	PARAMETER:
**	RETURN:
**	DESCRIPTION:
****************************************************************************/

int add_noise( double *x, int num_samples, double noise )
{
	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

	//fast counter
	register int t;
	//
	double tmp;

	///--------------------------------------------------------------------------
	///	CHECK
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	INITIALIZATIONS
	///--------------------------------------------------------------------------

	DENTER_ARG( "noise: %+1.3f\n", noise );

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------

	DPRINT("noise: ");
	//For every sample
	for (t = 0;t < num_samples;t++)
	{
		//generate noise
		tmp = rand_double( -noise/2.0, +noise/2.0 );
		//accumulate noise
		x[t] += tmp;
		DPRINT_NOTAB( "%+1.3f | ", x[t] );
	}	//End For every sample

	DPRINT_NOTAB("\n");

	///--------------------------------------------------------------------------
	///	FINALIZATIONS
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

	//Return
	DRETURN();

	return 0;
}	//end function:

/****************************************************************************
**	calc_line_stats
*****************************************************************************
**	PARAMETER:
**	RETURN:
**	DESCRIPTION:
****************************************************************************/

int calc_line_stats( double *x, int num_samples, double *x_avg, double *x2_avg )
{

	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

	//fast counter
	register int t;
	//Temp vector stats
	double x_avg_tmp;
	double x2_avg_tmp;

	///--------------------------------------------------------------------------
	///	CHECK
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	INITIALIZATIONS
	///--------------------------------------------------------------------------

    //Trace
	DENTER();

	//Check for output pointer consistency
	if ((x == NULL) || (x_avg == NULL) || (x2_avg == NULL))
    {
        //bad input pointers
        return -1;
    }

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------

    //Initialize working vars
	x_avg_tmp = 0.0;
	x2_avg_tmp = 0.0;
	//For every sample
	for (t = 0;t < num_samples;t++)
	{
		//accumulate vector stats
		x_avg_tmp += x[t];
		x2_avg_tmp += x[t] *x[t];

	}	//End For every sample
	//DPRINT( "x_avg: %1.3f | x2_avg: %1.3f\n", x_avg_tmp, x2_avg_tmp );
    //Normalize linear average
    x_avg_tmp /= 1.0 *num_samples;
    //Normalize square average
    x2_avg_tmp /= (1.0 *num_samples);

    DPRINT( "x_avg: %+1.3f | x2_avg: %+1.3f\n", x_avg_tmp, x2_avg_tmp );

	///--------------------------------------------------------------------------
	///	FINALIZATIONS
	///--------------------------------------------------------------------------

	//Return result to output parameters
    *x_avg = x_avg_tmp;
    *x2_avg = x2_avg_tmp;

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

	//Trace Return
	DRETURN();

	return 0;
}	//end function: calc_line_stats

/****************************************************************************
**	calc_cross_line_stats
*****************************************************************************
**	PARAMETER:
**	RETURN:
**	DESCRIPTION:
**  Obtain the average of the dot product of two vectors
****************************************************************************/

int calc_cross_line_stats( double *x, double *y, int num_samples, double *xy_avg )
{

	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

	//fast counter
	register int t;
	//Temp vector stats
	double xy_avg_tmp;

	///--------------------------------------------------------------------------
	///	CHECK
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	INITIALIZATIONS
	///--------------------------------------------------------------------------

    //Trace
	DENTER();

	//Check for output pointer consistency
	if ((x == NULL) || (y == NULL) || (xy_avg == NULL))
    {
        //bad input pointers
        return -1;
    }

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------

    //Initialize working vars
	xy_avg_tmp = 0.0;
	//For every sample
	for (t = 0;t < num_samples;t++)
	{
		//accumulate vector stats
		xy_avg_tmp += x[t] *y[t];

	}	//End For every sample
	//DPRINT( "x_avg: %1.3f | x2_avg: %1.3f\n", x_avg_tmp, x2_avg_tmp );
    //Normalize linear average
    xy_avg_tmp /= 1.0 *num_samples;

    DPRINT( "xy_avg: %+1.3f\n", xy_avg_tmp );

	///--------------------------------------------------------------------------
	///	FINALIZATIONS
	///--------------------------------------------------------------------------

	//Return result to output parameters
    *xy_avg = xy_avg_tmp;

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

	//Trace Return
	DRETURN();

	return 0;
}	//end function: calc_cross_line_stats

/****************************************************************************
**  linear regression
*****************************************************************************
**	PARAMETER:
**	RETURN:
**	DESCRIPTION:
**  Calculate the optimal linear transformation between input and output
**  Does not use the individual vector element, but overall vector wide stats
**  Those stats can be conveniently accumulated during other processing part
**  MATH:
**  gain = (xy_avg -x_avg *y_avg) / (x2_avg -x_avg *x_avg);
**  bias = y_avg -gain *x_avg;
****************************************************************************/

int linear_regression( double *x, double *y, int num_samples, double *gain, double *bias )
{
	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

	//temp output
	double gain_tmp, bias_tmp;
	//Temp
	double x_avg, x2_avg, y_avg, y2_avg, xy_avg;

	///--------------------------------------------------------------------------
	///	CHECK
	///--------------------------------------------------------------------------

	//Check for output pointer consistency
	if ((x == NULL) || (y == NULL) || (gain == NULL) || (bias == NULL))
    {
        //bad input pointers
        return -1;
    }

	///--------------------------------------------------------------------------
	///	INITIALIZATIONS
	///--------------------------------------------------------------------------

	//Trace Enter
	DENTER();

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------

        ///Calculate vectors stats
    //Calculate input vector stats
    calc_line_stats( x, num_samples, &x_avg, &x2_avg );
    //Calculate output vector stats
    calc_line_stats( y, num_samples, &y_avg, &y2_avg );
    //Calculate cross stats between input and output vector
	calc_cross_line_stats( x, y, num_samples, &xy_avg );

        ///Calculate gain and bias
	//optimal gain. Optimazed as minimal square distance between linearized output and output data.
    gain_tmp = (xy_avg -x_avg *y_avg) / (x2_avg -x_avg *x_avg);
    //optimal bias
    bias_tmp = y_avg -gain_tmp *x_avg;

    DPRINT( "gain: %+1.3f | bias: %+1.3f\n", gain_tmp, bias_tmp );

	///--------------------------------------------------------------------------
	///	FINALIZATIONS
	///--------------------------------------------------------------------------

	//Write back return parameters
	*gain = gain_tmp;
	*bias = bias_tmp;

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

    //Trace Return
	DRETURN();

	return 0;
}   //end function: linear_regression

/****************************************************************************
**  calc_correlation_share
*****************************************************************************
**	PARAMETER:
**      x: pointer vector of input vecotors. **x is num_inputs elements. each vector is num_samples elements.
**      y: output vector. num_samples elements
**      num_samples: number of input vectors
**      num_samples: width of the vectors
**	RETURN:
**      *xy_corr: vector with the share of correlation of each pair input x[i] output y
**	DESCRIPTION:
**  linear regression is useful for a pair x-y
**  The problem is that not all inputs x are equal. some will be more correlated to the output than others
**  Absoulte correlation has problems.
**  What I care abut is calculating the share a correlation an input as with the output.
**  It's both cheaper and works better
****************************************************************************/

int calc_correlation_share( double **x, double *y, int num_inputs, int num_samples, double *xy_corr )
{
	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

	register int t, ti;

	double *ptr;
	//temp working vectors
    double *var;    //variance for each input
    double *covar;  //covariance of input output pair
    double *tmp;    //temp vector. used for intermediate calculations and return
    double sum;     //Sum of all partial correlation share coefficients
	//Temp statistic variables
	double *x_avg, *x2_avg, *xy_avg;        //stats for the inputs and cross inputs. (one for each input)
	double y_avg, y2_avg;                   //Stats for the output (just one)

	///--------------------------------------------------------------------------
	///	CHECK
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	INITIALIZATIONS
	///--------------------------------------------------------------------------

	//Trace Enter
	DENTER();

        ///Allocate Memory
	//Allocate vector
    ptr = (double *)malloc( sizeof(double) *num_inputs );
    DPRINT( "Allocated vector: %p\n", ptr );
    //Link vector
    x_avg = ptr;
    //Allocate vector
    ptr = (double *)malloc( sizeof(double) *num_inputs );
    DPRINT( "Allocated vector: %p\n", ptr );
    //Link vector
    x2_avg = ptr;
    //Allocate vector
    ptr = (double *)malloc( sizeof(double) *num_inputs );
    DPRINT( "Allocated vector: %p\n", ptr );
    //Link vector
    xy_avg = ptr;
       //Allocate vector
    ptr = (double *)malloc( sizeof(double) *num_inputs );
    DPRINT( "Allocated vector: %p\n", ptr );
    //Link vector
    tmp = ptr;
    //Allocate vector
    ptr = (double *)malloc( sizeof(double) *num_inputs );
    DPRINT( "Allocated vector: %p\n", ptr );
    //Link vector
    var = ptr;
    //Allocate vector
    ptr = (double *)malloc( sizeof(double) *num_inputs );
    DPRINT( "Allocated vector: %p\n", ptr );
    //Link vector
    covar = ptr;

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------

    ///Compute input and input output stats
	//For every input
	for (t = 0;t < num_inputs;t++)
    {
        //link input vector
        ptr = x[t];
        //compute input stats
        calc_line_stats( ptr, num_samples, &x_avg[t], &x2_avg[t] );
        //compute input output stats
        calc_cross_line_stats( ptr, y, num_samples, &xy_avg[t] );
    }   //End for every input
    ///Compute output stats
    //Stats for lone output vector
    calc_line_stats( y, num_samples, &y_avg, &y2_avg );
    ///Compute individual elements for correlation share
    //Compute variance of input vectors and covariance of input-output pairs
    //For every input
	for (t = 0;t < num_inputs;t++)
    {
        //variance of input vector
        var[t] = x2_avg[t] - x_avg[t] *x_avg[t];
        //Compute standard deviation
        var[t] = sqrt( var[t] );
        //Compute covariance of input-output pair
        covar[t] = xy_avg[t] -x_avg[t] *y_avg;
    }   //End for every input
    ///Compute correlation share
    //Compute partial correlation share coeffiecients for all inputs
    //For every input
	for (t = 0;t < num_inputs;t++)
    {
        //Initialize partial correlation share coefficient
        tmp[t] = 1.0;
        //Compute the prodct of all variances except the variance of input t
        //For every input
        for (ti = 0;ti < num_inputs;ti++)
        {
            //if this is not the variance of the input t
            if (ti != t)
            {
                //Accumulate product of the standard deviations
                tmp[t] *= var[ti];
            }
        }   //End for every input
        //Accumulate the covariance as well
        tmp[t] *= covar[t];
        //Square the partial correlation share to enanche contrast between different correlation levels
        tmp[t] = tmp[t] *tmp[t];
        //Square again! MOAR contrast!!! (4th power)
        tmp[t] = tmp[t] *tmp[t];
        //SQUARE AGAIN!! EVEN MOAR CONTRAST!!! (8th power)
        tmp[t] = tmp[t] *tmp[t];
        //Make partial correlation share coefficient positive. Sign of correlation is already stored in gain from linear regression.
        //tmp[t] = ABS( tmp[t] );
        DPRINT("partial correlation share %d: %1.3e\n", t, tmp[t]);
    }   //End for every input
    //Compute sum of all partial correlation share
    //Initialize sum
    sum = 0.0;
    //For every input
	for (t = 0;t < num_inputs;t++)
    {
        //Accumulate all partial correlation share coefficients
        sum += tmp[t];
    }   //End for every input
    //Final correlation share is partial correlation share coefficient t against the sum of all of them
    //For every input
	for (t = 0;t < num_inputs;t++)
    {
        //Compute correlation share coefficient for input t
        tmp[t] /= sum;
        DPRINT("correlation share %d: %+1.3f\n", t, tmp[t]);
    }   //End for every input

    ///BUG: correlation share is high for uncorrelated inputs and low for correlated inputs

	///--------------------------------------------------------------------------
	///	FINALIZATIONS
	///--------------------------------------------------------------------------

        ///Save return parameter
    //For every input
	for (t = 0;t < num_inputs;t++)
    {
        //Compute correlation share coefficient for input t
        xy_corr[t] = tmp[t];
    }   //End for every input

        ///Free Memory
    //Clear link
    ptr = x_avg;
    x_avg = NULL;
    //free memory
    if (ptr != NULL)
    {
        free( ptr );
        DPRINT( "Freed vector: %p\n", ptr );
    }
    //Clear link
    ptr = x2_avg;
    x2_avg = NULL;
    //free memory
    if (ptr != NULL)
    {
        free( ptr );
        DPRINT( "Freed vector: %p\n", ptr );
    }
    //Clear link
    ptr = xy_avg;
    xy_avg = NULL;
    //free memory
    if (ptr != NULL)
    {
        free( ptr );
        DPRINT( "Freed vector: %p\n", ptr );
    }
    //Clear link
    ptr = tmp;
    tmp = NULL;
    //free memory
    if (ptr != NULL)
    {
        free( ptr );
        DPRINT( "Freed vector: %p\n", ptr );
    }
    //Clear link
    ptr = var;
    var = NULL;
    //free memory
    if (ptr != NULL)
    {
        free( ptr );
        DPRINT( "Freed vector: %p\n", ptr );
    }
    //Clear link
    ptr = covar;
    covar = NULL;
    //free memory
    if (ptr != NULL)
    {
        free( ptr );
        DPRINT( "Freed vector: %p\n", ptr );
    }

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

    //Trace Return
	DRETURN();

	return 0;
}	//end function: calc_correlation_share

/****************************************************************************
**  execute_neuron
*****************************************************************************
**	PARAMETER:
**	RETURN:
**	DESCRIPTION:
**  Execute a neuron given a vector of vectors of inputs avector of gains and a bias
****************************************************************************/

int execute_neuron( double **x, double *y, int num_inputs, int num_samples, double *gain, double bias )
{
	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

	//Fast counters
	register int t, ti;

    double tmp;

    double y_tmp;

    //
    double *ptr;

	///--------------------------------------------------------------------------
	///	CHECK
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	INITIALIZATIONS
	///--------------------------------------------------------------------------

	//Trace Enter
	DENTER();
    //Print weights and bias
	//For every input
    for (t = 0;t < num_inputs;t++)
    {
        DPRINT("gain %d: %+1.3f\n", t, gain[t]);
    }   //End for every input
    DPRINT("bias: %+1.3f\n", bias);

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------

	DPRINT("output: ");
	//For every sample
    for (t = 0;t < num_samples;t++)
    {
        //Initialize output
        y_tmp = bias;
    	//For every input
        for (ti = 0;ti < num_inputs;ti++)
        {
            //Link vector
            ptr = x[ti];
            //Compute partial product of input against its gain
            tmp = ptr[t] *gain[ti];
            //Accumulate partial product
            y_tmp += tmp;
        }   //End for every input
        //Save output parameter
        y[t] = y_tmp;
        DPRINT_NOTAB( "%+1.3f | ", y_tmp);
    }   //End for every sample
    DPRINT("\n");

	///--------------------------------------------------------------------------
	///	FINALIZATIONS
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

    //Trace Return
	DRETURN();

	return 0;
}	//end function: execute_neuron

/****************************************************************************
**  Compute the RMS distance between two vectors
*****************************************************************************
**	PARAMETER:
**	RETURN:
**	DESCRIPTION:
**
****************************************************************************/

double calc_error( double *x, double *y, int num_samples )
{
	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

	//Fast counters
	register int t;


	double err;
	double err_tot;

	//String where print error
	char str[10];

	///--------------------------------------------------------------------------
	///	CHECK
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	INITIALIZATIONS
	///--------------------------------------------------------------------------

	//Trace Enter
	DENTER();

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------

	DPRINT( "err: | " );
	//initialize error
	err_tot = 0.0;
    //For every sample
    for (t = 0;t < num_samples;t++)
    {
        //compute square error of sample
        err = (x[t] -y[t]) *(x[t] -y[t]);
        //accumulate square error
        err_tot += err;

        DPRINT_NOTAB( "%1.3f | ", err );
    }   //For every sample
    //Normalize
    err_tot /= (1.0 *num_samples);
    //Calculate RMS error
    err_tot = sqrt( err_tot );

    DPRINT_NOTAB( "\n", err );

    //Print the error in engineering notation
    eng_num( str, err_tot, NULL );

    //DPRINT( "err: %+1.3e\n", err_tot );
    DPRINT( "err: %s\n", str );

	///--------------------------------------------------------------------------
	///	FINALIZATIONS
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

    //Trace Return
	DRETURN();

	return err_tot;
}	//end function:

/****************************************************************************
**	eng_num
*****************************************************************************
**	Show a number in engineering notation
**	four sgnificant digit, 10^n*3 suffix, optional unit of measure
**	x.xxx [xU]
**	xx.xx [xU]
**	xxx.x [xU]
**	Exponents
**	index	|	-6	-5	-4	-3	-2	-1	0	+1	+2	+3	+4	+5
**	exp		|	-18	-15	-12	-9	-6	-3	0	+3	+6	+9	+12	+15	+18
**	suffix	|	a	f	p	n	u	m		K	M	G	T	P	E
****************************************************************************/

int eng_num( char *str, float num, char *unit )
{
	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

	//exponents suffix string
	static char suffix[] =
	{
		'a', 'f', 'p', 'n', 'u', 'm', ' ', 'K', 'M', 'G', 'T', 'P', 'E'
	};

	char *unit_str;

	char continue_flag;

	int cnt;

	///--------------------------------------------------------------------------
	///	CHECK AND INITIALIZATIONS
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------
	//	ALGORITHM:
	//	multiply or divide by 1000 until the number is > 1.0 && < 1000.0
	//	create a string with the '.' in the right place and with the right suffix and unit if applicable

		///STEP1: Make num >1.0 <1000.0 and calculate exponent
	//if: input is exactly 0, i don't have to scale it
	if (num == 0.0)
	{
		continue_flag = 0;
	}
	else
	{
		continue_flag = 1;
	}
	//While Init
	cnt = 0;				//exponent counter
	//while: i'm not done
	while (continue_flag == 1)
	{
		//if: too small
		if (num < 1.0)
		{
			//inflate by 10^3
			num = num *1000.0;
			cnt--;
		}
		//if: too big
		else if (num > 1000.0)
		{
			//shrink by 10^3
			num = num /1000.0;
			cnt++;
		}
		//if: sweet spot
		else
		{
			//done
			continue_flag = 0;
		}
		//if: i run out of suffix
		if ((cnt > 6) || (cnt < -6))
		{
			//done
			continue_flag = 0;

			//fail
			return -1;
		}
	}	//end While: i'm not done

		///STEP2: calculate string

	if (unit == NULL)
	{
		unit_str = (char *)"";
	}
	else
	{
		unit_str = unit;
	}

	if (num < 10.0)
	{
		sprintf(str, "%1.3f %c%s", num, suffix[cnt +6], unit_str);
	}
	else if (num < 100.0)
	{
		sprintf(str, "%2.2f %c%s", num, suffix[cnt +6], unit_str);
	}
	else if (num < 1000.0)
	{
		sprintf(str, "%3.1f %c%s", num, suffix[cnt +6], unit_str);
	}

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

	//OK
	return 0;
}	//end function: eng_num

/****************************************************************************
**  execute_trial
*****************************************************************************
**	PARAMETER:
**	RETURN:
**	DESCRIPTION:
****************************************************************************/

//Execute a trial to show difference between naive linear regression and correlation share powered linear regression
int execute_trial( double noise_a, double noise_b, double noise_c, double noise_y, double *err, double *err_corr_share )
{
	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

    //three input vectors
	double a[NUM_SAMPLES];
	double b[NUM_SAMPLES];
	double c[NUM_SAMPLES];
	//one output vector
	double y[ NUM_SAMPLES ];
	//output of the computed neuron
	double y_exe[ NUM_SAMPLES ];

	//Characteristic of input and output vectors
	double a_gain, a_bias;
	double b_gain, b_bias;
	double c_gain, c_bias;
	double y_gain, y_bias;

    //Linear regression optimal input-output gain and bias
	double ay_gain, ay_bias;
	double by_gain, by_bias;
	double cy_gain, cy_bias;

	double *x[NUM_INPUTS];

	//Neuron parameters
	double gain[NUM_INPUTS];   //One gain for each input. weights ofthe neuron
	double bias;                //One bias. A neuron has just one bias.

	//correlation share
	double xy_corr_share[ NUM_INPUTS ];

	//errors of the two ways to compute
	double err1, err2;

	///--------------------------------------------------------------------------
	///	CHECK
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	INITIALIZATIONS
	///--------------------------------------------------------------------------

	//Trace Enter
	DENTER();

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------

    DPRINT("Generate input and output vectors\n");

    ///Generate biases and slopes
    //random gains
    a_gain = rand_double(+1.0, +2.0);
    b_gain = rand_double(-1.0, -2.0);
    c_gain = rand_double(-2.0, +2.0);
    y_gain = rand_double(+1.0, +10.0);
    //random biases
    a_bias = rand_double(-5.0, +5.0);
    b_bias = rand_double(-5.0, +5.0);
    c_bias = rand_double(-5.0, +5.0);
    y_bias = rand_double(-5.0, +5.0);

	fill_line( a, NUM_SAMPLES, -1.0, 1.0, a_gain, a_bias );
	add_noise( a, NUM_SAMPLES, noise_a );

	fill_line( b, NUM_SAMPLES, -1.0, +1.0, b_gain, b_bias );
	add_noise( b, NUM_SAMPLES, noise_b );

	fill_line( c, NUM_SAMPLES, -1.0, +1.0, c_gain, c_bias );
	add_noise( c, NUM_SAMPLES, noise_c );

	fill_line( y, NUM_SAMPLES, 0.0, 1.0, y_gain, y_bias );
	add_noise( y, NUM_SAMPLES, noise_y );


	DPRINT("Compute optimal gains and biases via linear regression\n");
	linear_regression( a, y, NUM_SAMPLES, &ay_gain, &ay_bias );
	linear_regression( b, y, NUM_SAMPLES, &by_gain, &by_bias );
	linear_regression( c, y, NUM_SAMPLES, &cy_gain, &cy_bias );

	DPRINT("Compute correlation share\n");
	//link input vectors
	x[0] = a;
	x[1] = b;
	x[2] = c;
    //Compute correlation share
    calc_correlation_share( x, y, NUM_INPUTS, NUM_SAMPLES, &xy_corr_share[0] );

    DPRINT("Compute naive linear regression neuron parameters and its error\n");
    //The simplest way to make linear regression on a sum of inputs is to just take a share from each of them.
    //The math works out, but if input 0 is noisier, the output is noisier, increasing the overall error.
    //An equal share of error from each input is carried to the output
    gain[0] = ay_gain / NUM_INPUTS;
    gain[1] = by_gain / NUM_INPUTS;
    gain[2] = cy_gain / NUM_INPUTS;
    bias = (ay_bias +by_bias +cy_bias) / NUM_INPUTS;
    //Execute the neuron on inputs and neuron parameters
    execute_neuron( x, y_exe, NUM_INPUTS, NUM_SAMPLES, gain, bias );
    //Calculate error between desied output vector 'y' and computed output vector 'y_exe'
    err1 = calc_error( y, y_exe, NUM_SAMPLES );

    DPRINT("Compute correlation share powered linear regression neuron parameters and its error\n");
    //Instead of giving an equal share of gain to all inputs, i give more share to those that have an higher share of correlation
    gain[0] = ay_gain *(xy_corr_share[0]);
    gain[1] = by_gain *(xy_corr_share[1]);
    gain[2] = cy_gain *(xy_corr_share[2]);
    bias = ay_bias *(xy_corr_share[0]) +by_bias *(xy_corr_share[1]) +cy_bias *(xy_corr_share[2]);
    //Execute the neuron on inputs and neuron parameters
    execute_neuron( x, y_exe, NUM_INPUTS, NUM_SAMPLES, gain, bias );
    //Calculate error between desied output vector 'y' and computed output vector 'y_exe'
    err2 = calc_error( y, y_exe, NUM_SAMPLES );

	///--------------------------------------------------------------------------
	///	FINALIZATIONS
	///--------------------------------------------------------------------------

	//save return parameters
	*err = err1;
	*err_corr_share = err2;

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

    //Trace Return
	DRETURN();

	return 0;
}	//end function: execute_trial


/****************************************************************************
**
*****************************************************************************
**	PARAMETER:
**	RETURN:
**	DESCRIPTION:
****************************************************************************/

int f( void )
{
	///--------------------------------------------------------------------------
	///	STATIC VARIABILE
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	LOCAL VARIABILE
	///--------------------------------------------------------------------------

    register int t;
    int num_samples;

	///--------------------------------------------------------------------------
	///	CHECK
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	INITIALIZATIONS
	///--------------------------------------------------------------------------

	//Trace Enter
	DENTER();

	///--------------------------------------------------------------------------
	///	BODY
	///--------------------------------------------------------------------------

    //For every sample
    for (t = 0;t < num_samples;t++)
    {

    }   //For every sample

	///--------------------------------------------------------------------------
	///	FINALIZATIONS
	///--------------------------------------------------------------------------

	///--------------------------------------------------------------------------
	///	RETURN
	///--------------------------------------------------------------------------

    //Trace Return
	DRETURN();

	return 0;
}	//end function:


