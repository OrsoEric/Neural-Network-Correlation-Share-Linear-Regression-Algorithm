# Neural-Network-Correlation-Share-Linear-Regression-Algorithm
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
**  F[m] = std[m] *PROD(j!=m)(std[j])
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
