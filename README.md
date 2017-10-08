# FCANN

Simple module for fully (F) connected (C) artificial (A) neural (N) networks (N).  Minimum functionality for changing error functions.  Current output task is fixed at regression (floating point value output) but in the future the option to define a network for classification will be added.

## Installation

Within Julia, execute

```julia
Pkg.clone("https://github.com/Blackbody-Research/FCANN.jl")
```

Ensure packaging has been installed properly by running ```Pkg.test("FCANN")``` 

----

## Usage

To start using the package type ```using FCANN```.  Unless otherwise stated, all of the code below will function without invoking the package name as these functions are exported.

### Network Setup

The following examples cover preliminary and auxilliary functions to working with data and network parameters.

#### Data Formatting

Input and output data must be formatted into ``` Array{Float32, 2}``` such that each row contains an example datapoint.  The number of rows between the input and output sets passed for training must match.  Below is an example of generating sample training and testing data.

```julia
#number of inputs, outputs, and total examples
M = 10
O = 2
numTrain = 100000
numTest = 10000

#random training and test data
Xtrain = randn(Float32, numTrain, M)
ytrain = randn(Float32, numTrain, O)
Xtest = randn(Float32, numTest, M)
ytest = randn(Float32, numTest, O)
```

#### Initializing Network Parameters

Using the above test data we can initialize an ANN model with a number of hidden layers.  The model structure is specified with a vector of type ```Array{Int64, 1}```. For example, ```H=[10, 10]``` would create a network with two hidden layers of 10 neurons each.  Below is an example of generating initial random parameters for such a network.

``` julia
#Define hidden layer sizes
H = [10, 10]
#Initialize parameters
T0, B0 = initializeParams(M, H, O)
```

T0 and B0 are each arrays of arrays that contain the "theta" and "bias" parameters for the model.

#### Reading and Writing Network Parameters

Since networks can get very large, there are functions to save a network to disk as a binary file with the first several bytes indicating how to read the file properly.  

```julia
#Save parameter to disk
writeParams([(T0, B0)], "testParams.bin")
```

Note that the parameters must be passed in a tuple inside of a vector with the thetas followed by the biases.  This write function can accomodate ensemble networks which contain multiple full networks of the same architecture.  In this case the array passed to writeParams would contain more than one tuple.

```julia
#Read parameters from disk
params = readBinParams("testParams.bin")
(T1, B1) = params[1]
```

The returned value from readBinParams will contain the same type of vector passed to the save function.  To extract a single set of parameters one must access the tuple from a particular index in the vector.

#### Generating Output from Network Parameters

The following is an example of using a set of network parameters with some input data. 

```julia
testOut = predict(T0, B0, Xtest)
sqErr = mean((testOut.-ytest).^2)
```

The testOut format will have the same dimensionality as the true output data so an error calculation can be made comparing the two with broadcast operations.  

----

### Training a Network

A number of helper functions exit to perform commonly desired training tasks that automate preparation of the data and saving outputs.  Examples of those routines will appear below but first is code showing how to use the raw training function which is not exported by the package.  This training procedure uses batched stochastic gradient descent and an ADAMAX algorithm to scale learning rate per parameter.  There is also a base learning rate alpha and an exponential decay rate R that applies every 10 epochs. 

```julia
#define training parameters
batchSize = 1024
N = 1000 # number of training epochs (iterations through entire data set)
lambda = 0.0f0 # L2 regularization constant
c = 2.0f0 # max norm regularization constant (Inf means it doesn't apply)

#run training saving output variables containing trained parameters (T, B), the lowest cost achieved on the training set, a record of costs per epoch, and a record of timestamps each epoch
T, B, bestCost, record, timeRecord = FCANN.ADAMAXTrainNN(Xtrain, ytrain, batchSize, T0, B0, N, M, H, lambda, c; alpha=0.002f0, R = 0.1f0, printProgress = false, dropout = 0.0f0, costFunc = "absErr")
```

They keyword arguments can be omitted but here show the default values.  The costFunc keyword can be used to change the cost function in the final output layer that calculates the error between the model output and the training data output.  Whenever a model is trained, it is done with a particular cost function in mind.  This is specified in the training process, not in the model construction.

#### Training a Single Network

A helper function wraps some of the above code in a manner that allows easier loading and saving of data.  

``` julia
#write data to disk with the name "ex"
writecsv("Xtrain_ex.csv", X)
writecsv("Xtest_ex.csv", Xtest)
writecsv("ytrain_ex.csv", ytrain)
writecsv("ytest_ex.csv", ytest)

#define training parameters
alpha = 0.002f0
R = 0.1f0
ID = "trial1"

#train a network based on the "ex" data
record, T, B = fullTrain("ex", N, batchSize, H, lambda, c, alpha, R, ID)
```

Here the fullTrain function automatically reads the training and test data from disk with the naming format specified above and initializes a network of size H to conduct training with.  The following files will also be saved to disk:

- ```trial1_costRecord_ex_10_input_[10, 10]_hidden_2_output_0.0_lambda_2.0_maxNorm_0.002_alpha_ADAMAX.csv```  Record of cost function on training set each epoch
- ```trial1_timeRecord_ex_10_input_[10, 10]_hidden_2_output_0.0_lambda_2.0_maxNorm_0.002_alpha_ADAMAX.csv``` Record of times for each training epoch
- ```trial1_performance_ex_10_input_[10, 10]_hidden_2_output_0.0_lambda_2.0_maxNorm_0.002_alpha_ADAMAX.csv``` Final performance in terms of cost function on training and test set
- ```trial1_params_ex_10_input_[10, 10]_hidden_2_output_0.0_lambda_2.0_maxNorm_0.002_alpha_ADAMAX.bin```  Parameters in binary format

An alternative version of fullTrain can take two additional inputs after the name that directly pass an X and Y set of training data already in the proper format.  In this case, the outputs will be the same but the error calculations will not include a test set.

#### Training a Group of Networks in Parallel

Usually we want to evaluate a number of networks on a single dataset to get an idea of what complexity is needed to reduce error to desired levels.  In this case the archEval helper function is one method of training a number of networks at once saving training and test set errors for each one.

```julia
#Set up package for parallel execution
addprocs(6) #add 6 parallel workers
@everywhere using FCANN #initialize package on all workers

#Define which networks should be trained
hiddenList = [[], [1], [2, 2], [10, 5, 2]]

#Run training in parallel
archEval("ex", N, batchSize, hiddenList)
```

A single file will be generated named ```archEval_ex_10_input_2_output_ADAMAX.csv``` that contains a table of errors for each architecture.  Note that one of the networks was simply an empty array.  In this case the model will contain no hidden layers and will simply be a linear model.

----

## Notes

One may see the available error function strings by calling ```requestCostFunctions()``` after initializing the package.  Currently there are only two options: "absErr" and "sqErr" with "absErr" as the default.

The error function is defined during the training process but the saved files will specify which error function was used.  The network parameters themselves do not contain that information outside of the file name.  

The final output layer does not apply any transformation function in line with a regression task so the output data in the examples should be actual numbers rather than a classification vector.

----

## Future Plans

- Add additional error functions
- Add GPU backend switching
- Add switching between regression and classification

## Credits

Written by Jason Eckstein for use by Blackbody Research