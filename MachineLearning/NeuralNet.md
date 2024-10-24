(sec:NeuralNet)=
# Neural networks

Artificial neural networks are computational systems that can learn to
perform tasks by considering examples, generally without being
programmed with any task-specific rules. It is supposed to mimic a
biological system, wherein neurons interact by sending signals in the
form of mathematical functions between layers. All layers can contain
an arbitrary number of neurons, and each connection is represented by
a weight variable.

```{image} https://ml4a.github.io/images/temp_fig_mnist.png
:width: 500px
:align: center
:alt: from Machine Learning for Artists [](https://ml4a.github.io/)
```

## Terminology

Each time we describe a neural network algorithm we will typically specify three things. 

```{admonition} Architecture
  The architecture specifies what variables are involved in the network and their topological relationships – for example, the variables involved in a neural net might be the weights of the connections between the neurons, along with the activities of the neurons.
  ```

```{admonition} Activation rule
  Most neural network models have short time-scale dynamics: local rules define how the activities of the neurons change in response to each other. Typically the activation rule depends on the weights (the parameters) in the network.
  ```
  
```{admonition} Learning algorithm
  The learning algorithm specifies the way in which the neural network’s weights change with time. This learning is usually viewed as taking place on a longer time scale than the time scale of the dynamics under the activity rule. Usually the learning rule will depend on the activities of the neurons. It may also depend on the values of target values supplied by a teacher and on the current value of the weights.
  ```

## Artificial neurons

The field of artificial neural networks has a long history of
development, and is closely connected with the advancement of computer
science and computers in general. A model of artificial neurons was
first developed by McCulloch and Pitts in 1943 to study signal
processing in the brain and has later been refined by others. The
general idea is to mimic neural networks in the human brain, which is
composed of billions of neurons that communicate with each other by
sending electrical signals.  Each neuron accumulates its incoming
signals, which must exceed an activation threshold to yield an
output. If the threshold is not overcome, the neuron remains inactive,
i.e. has zero output.

This behaviour has inspired a simple mathematical model for an artificial neuron.

\begin{equation}
 y = f\left(\sum_{i=1}^n w_jx_j + b \right) = f(z),
 \label{artificialNeuron}
\end{equation}

where the bias $b$ is sometimes denoted $w_0$.
Here, the output $y$ of the neuron is the value of its activation function, which have as input
a weighted sum of signals $x_1, \dots ,x_n$ received from $n$ other neurons.

Conceptually, it is helpful to divide neural networks into four
categories:
1. general purpose neural networks, including deep neural networks (DNN) with several hidden layers, for supervised learning,
2. neural networks designed specifically for image processing, the most prominent example of this class being Convolutional Neural Networks (CNNs),
3. neural networks for sequential data such as Recurrent Neural Networks (RNNs), and
4. neural networks for unsupervised learning such as Deep Boltzmann Machines.

In natural science, DNNs and CNNs have already found numerous
applications. In statistical physics, they have been applied to detect
phase transitions in 2D Ising and Potts models, lattice gauge
theories, and different phases of polymers, or solving the
Navier-Stokes equation in weather forecasting.  Deep learning has also
found interesting applications in quantum physics. Various quantum
phase transitions can be detected and studied using DNNs and CNNs:
topological phases, and even non-equilibrium many-body
localization. 

In quantum information theory, it has been shown that one can perform
gate decompositions with the help of neural networks. 

The scientific applications are certainly not limited to the natural sciences. In fact, there is a plethora of applications in essentially all disciplines, from the humanities to life science and medicine. However, the real expansion has been into the tech industry and other private sectors.

## Neural network types

An artificial neural network (ANN), is a computational model that
consists of layers of connected neurons, or nodes or units.  We will
refer to these interchangeably as units or nodes, and sometimes as
neurons.

It is supposed to mimic a biological nervous system by letting each
neuron interact with other neurons by sending signals in the form of
mathematical functions between layers.  A wide variety of different
ANNs have been developed, but most of them consist of an input layer,
an output layer and eventual layers in-between, called *hidden
layers*. All layers can contain an arbitrary number of nodes, and each
connection between two nodes is associated with a weight variable.

Neural networks (also called neural nets) are neural-inspired
nonlinear models for supervised learning.  As we will see, neural nets
can be viewed as natural, more powerful extensions of supervised
learning methods such as linear and logistic regression and soft-max
methods we discussed earlier.


### Feed-forward neural networks

The feed-forward neural network (FFNN) was the first and simplest type
of ANNs that were devised. In this network, the information moves in
only one direction: forward through the layers.

Nodes are represented by circles, while the arrows display the
connections between the nodes, including the direction of information
flow. Additionally, each arrow corresponds to a weight variable
(figure to come).  We observe that each node in a layer is connected
to *all* nodes in the subsequent layer, making this a so-called
*fully-connected* FFNN.


### Convolutional Neural Network

A different variant of FFNNs are *convolutional neural networks*
(CNNs), which have a connectivity pattern inspired by the animal
visual cortex. Individual neurons in the visual cortex only respond to
stimuli from small sub-regions of the visual field, called a receptive
field. This makes the neurons well-suited to exploit the strong
spatially local correlation present in natural images. The response of
each neuron can be approximated mathematically as a convolution
operation.  (figure to come)

Convolutional neural networks emulate the behaviour of neurons in the
visual cortex by enforcing a *local* connectivity pattern between
nodes of adjacent layers: Each node in a convolutional layer is
connected only to a subset of the nodes in the previous layer, in
contrast to the fully-connected FFNN.  Often, CNNs consist of several
convolutional layers that learn local features of the input, with a
fully-connected layer at the end, which gathers all the local data and
produces the outputs. They have wide applications in image and video
recognition.

<!-- !split -->
### Recurrent neural networks

So far we have only mentioned ANNs where information flows in one
direction: forward. *Recurrent neural networks* on the other hand,
have connections between nodes that form directed *cycles*. This
creates a form of internal memory which are able to capture
information on what has been calculated before; the output is
dependent on the previous computations. Recurrent NNs make use of
sequential information by performing the same task for every element
in a sequence, where each element depends on previous elements. An
example of such information is sentences, making recurrent NNs
especially well-suited for handwriting and speech recognition.

<!-- !split -->
### Other types of networks

There are many other kinds of ANNs that have been developed. One type
that is specifically designed for interpolation in multidimensional
space is the radial basis function (RBF) network. RBFs are typically
made up of three layers: an input layer, a hidden layer with
non-linear radial symmetric activation functions and a linear output
layer (''linear'' here means that each node in the output layer has a
linear activation function). The layers are normally fully-connected
and there are no cycles, thus RBFs can be viewed as a type of
fully-connected FFNN. They are however usually treated as a separate
type of NN due the unusual activation functions.

<!-- !split -->
## Neural network architecture

Let us restrict ourselves in this lecture to feed-forward ANNs. The term *multilayer perceptron* (MLP) is used ambiguously in the literature, sometimes loosely to mean any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons (with threshold activation). A general MLP consists of

1. a neural network with one or more layers of nodes between the input and the output nodes.
2. the multilayer network structure, or architecture, or topology, consists of an input layer, one or more hidden layers, and one output layer.
3. the input nodes pass values to the first hidden layer, its nodes pass the information on to the second and so on till we reach the output layer.

As a convention it is normal to call a network with one layer of input units, one layer of hidden units and one layer of output units as  a two-layer network. A network with two layers of hidden units is called a three-layer network etc.

The number of input nodes does not need to equal the number of output
nodes. This applies also to the hidden layers. Each layer may have its
own number of nodes and activation functions.

The hidden layers have their name from the fact that they are not
linked to observables and as we will see below when we define the
so-called activation $\boldsymbol{z}$, we can think of this as a basis
expansion of the original inputs $\boldsymbol{x}$. 

<!-- !split -->
### Why multilayer perceptrons?

According to the universal approximation
theorem {cite}`Cybenko1989`, a feed-forward
neural network with just a single hidden layer containing a finite
number of neurons can approximate a continuous multidimensional
function to arbitrary accuracy, assuming the activation function for
the hidden layer is a **non-constant, bounded and
monotonically-increasing continuous function**. The theorem thus
states that simple neural networks can represent a wide variety of
interesting functions when given appropriate parameters. It is the
multilayer feedforward architecture itself which gives neural networks
the potential of being universal approximators.

Note that the requirements on the activation function only applies to
the hidden layer, the output nodes are always assumed to be linear, so
as to not restrict the range of output values.


<!-- !split -->
### Mathematical model

The output $y$ from a single neuron is produced via the activation function $f$

\begin{equation}
y = f\left( \sum_{i=1}^n w_i x_i + b \right) = f(z),
\end{equation}

This function receives $x_i$ as inputs.
Here the activation $z=(\sum_{i=1}^n w_ix_i+b)$. 
In an FFNN of such neurons, the *inputs* $x_i$ are the *outputs* of
the neurons in the preceding layer. Furthermore, an MLP is
fully-connected, which means that each neuron receives a weighted sum
of the outputs of *all* neurons in the previous layer.

<!-- !split -->
First, for each node $j$ in the first hidden layer, we calculate a weighted sum $z_j^1$ of the input coordinates $x_i$,


\begin{equation} z_j^1 = \sum_{i=1}^{n} w_{ji}^1 x_i + b_j^1
\end{equation}


Here $b_j^1$ is the so-called bias which is normally needed in
case of zero activation weights or inputs. How to fix the biases and
the weights will be discussed below.  The value of $z_j^1$ is the
argument to the activation function $f$ of each node $j$, The
variable $n$ stands for all possible inputs to a given node $j$ in the
first layer.  The output $y_j^1$ from neuron $j$ in layer 1 is


\begin{equation}
 y_j^1 = f(z_j^1) = f\left(\sum_{i=1}^n w_{ji}^1 x_i  + b_j^1\right),
 \label{outputLayer1}
\end{equation}

where we assume that all nodes in the same layer have identical
activation functions, hence the notation $f$. In general, we could assume in the more general case that different layers have different activation functions.
In this case we would identify these functions with a superscript $l$ for the $l$-th layer,


\begin{equation}
 y_i^l = f^l(z_i^l) = f^l\left(\sum_{j=1}^{N_{l-1}} w_{ij}^l y_j^{l-1} + b_i^l\right),
 \label{generalLayer}
\end{equation}

where $N_{l-1}$ is the number of nodes in layer $l-1$. When the output of
all the nodes in the first hidden layer are computed, the values of
the subsequent layer can be calculated and so forth until the output
is obtained.


<!-- !split -->
The output of neuron $i$ in layer 2 is thus,

\begin{align}
 y_i^2 &= f^2\left(\sum_{j=1}^N w_{ij}^2 y_j^1 + b_i^2\right) \\
 &= f^2\left[\sum_{j=1}^N w_{ij}^2f^1\left(\sum_{k=1}^M w_{jk}^1 x_k + b_j^1\right) + b_i^2\right]
 \label{outputLayer2}
\end{align}

where we have substituted $y_k^1$ with the inputs $x_k$. Finally, the ANN output reads

\begin{align}
 y_i^3 &= f^3\left(\sum_{j=1}^N w_{ij}^3 y_j^2 + b_i^3\right) \\
 &= f^3\left[\sum_{j} w_{ij}^3 f^2\left(\sum_{k} w_{jk}^2 f^1\left(\sum_{m} w_{km}^1 x_m + b_k^1\right) + b_j^2\right)
  + b_1^3\right]
\end{align}


<!-- !split -->
We can generalize this expression to an MLP with $L$ hidden
layers. The complete functional form is,


\begin{align}
&y^{L+1}_i = f^{L+1}\left[\!\sum_{j=1}^{N_L} w_{ij}^L f^L \left(\sum_{k=1}^{N_{L-1}}w_{jk}^{L-1}\left(\dots f^1\left(\sum_{n=1}^{N_0} w_{mn}^1 x_n+ b_m^1\right)\dots\right)+b_k^{L-1}\right)+b_1^L\right] &&
 \label{completeNN}
\end{align}

which illustrates a basic property of MLPs: The only independent
variables are the input values $x_n$.

<!-- !split -->
This confirms that an MLP, despite its quite convoluted mathematical
form, is nothing more than an analytic function, specifically a
mapping of real-valued vectors $\boldsymbol{x} \in \mathbb{R}^n \rightarrow
\boldsymbol{y} \in \mathbb{R}^m$.

Furthermore, the flexibility and universality of an MLP can be
illustrated by realizing that the expression is essentially a nested
sum of scaled activation functions of the form


\begin{equation}
 f(x) = c_1 f(c_2 x + c_3) + c_4,
\end{equation}

where the parameters $c_i$ are weights and biases. By adjusting these
parameters, the activation functions can be shifted up and down or
left and right, change slope or be rescaled which is the key to the
flexibility of a neural network.

<!-- !split -->
### Matrix-vector notation

We can introduce a more convenient matrix-vector notation for all quantities in an ANN. 

In particular, we represent all signals as layer-wise row vectors $\boldsymbol{y}^l$ so that the $i$-th element of each vector is the output $y_i^l$ of node $i$ in layer $l$. 

We have that $\boldsymbol{W}^l$ is an $N_{l-1} \times N_l$ matrix, while $\boldsymbol{b}^l$ and $\boldsymbol{y}^l$ are $1 \times N_l$ row vectors. 
With this notation, the sum becomes a matrix-vector multiplication, and we can write
the equation for the activations of hidden layer 2 (assuming three nodes for simplicity) as

\begin{equation}
 \boldsymbol{y}^2 = f^2(\boldsymbol{y}^{1} \boldsymbol{W}^2 + \boldsymbol{b}^{2}) = 
 f_2\left(
     \left[
           y^1_1,
           y^1_2,
           y^1_3
          \right]
 \left[\begin{array}{ccc}
    w^2_{11} &w^2_{12} &w^2_{13} \\
    w^2_{21} &w^2_{22} &w^2_{23} \\
    w^2_{31} &w^2_{32} &w^2_{33} \\
    \end{array} \right] 
 + 
    \left[
           b^2_1,
           b^2_2,
           b^2_3
          \right]\right).
\end{equation}

This is not just a convenient and compact notation, but also a useful
and intuitive way to think about MLPs: The output is calculated by a
series of matrix-vector multiplications and vector additions that are
used as input to the activation functions. For each operation
$\mathrm{W}_l \boldsymbol{y}_{l-1}$ we move forward one layer.


<!-- !split -->
## Activation rules

A property that characterizes a neural network, other than its
connectivity, is the choice of activation function(s).  The following restrictions are imposed on an activation function for a FFNN to fulfill the universal approximation theorem

  * Non-constant
  * Bounded
  * Monotonically-increasing
  * Continuous

<!-- !split -->
*Logistic and Hyperbolic activation functions*
The second requirement excludes all linear functions. Furthermore, in
a MLP with only linear activation functions, each layer simply
performs a linear transformation of its inputs.

Regardless of the number of layers, the output of the NN will be
nothing but a linear function of the inputs. Thus we need to introduce
some kind of non-linearity to the NN to be able to fit non-linear
functions Typical examples are the logistic *Sigmoid*

\begin{equation}
 f_\mathrm{sigmoid}(z) = \frac{1}{1 + e^{-z}},
\end{equation}

and the *hyperbolic tangent* function

\begin{equation}
 f_\mathrm{tanh}(z) = \tanh(z)
\end{equation}

```{admonition} Noisy networks
Both the sigmoid and tanh activation functions imply that signals will be non-zero everywhere. This leads to inefficiencies in both feed-forward and back-propagation. 
```

<!-- !split -->
The ambition to make some neurons quiet (or to simplify the gradients) leads to the family of *rectifier activation functions*
The Rectifier Linear Unit (ReLU) uses the following activation function

\begin{equation}
f_\mathrm{ReLU}(z) = \max(0,z).
\end{equation}

To solve a problem of dying ReLU neurons, practitioners often use a  variant of the ReLU
function, such as the leaky ReLU or the so-called
exponential linear unit (ELU) function

\begin{equation}
f_\mathrm{ELU}(z) = \left\{\begin{array}{cc} \alpha\left( \exp{(z)}-1\right) & z < 0,\\  z & z \ge 0.\end{array}\right. 
\end{equation}

Finally, note that the final layer of a MLP often uses a different activation rule as it must produce a relevant output signal. This could be some linear activation function to give a continuous output for regression, or a softmax function for classification probabilities.

## Learning algorithm

The determination of weights (learning) involves multiple choices

1. Choosing a cost function, i.e., how to compare outputs with targets.
   * Possible choices include: Mean-squared error (MSE), Mean-absolute error (MAE), Cross-Entropy.
   * Regularization can be employed to avoid overfitting.
   * Physics (model) knowledge can be incorporated in the construction of a relevant cost function.
2. Optimization algorithm.
   * Back-propagation (see below) can be used to extract gradients of the cost function with respect to the weights. It corresponds to using the chain rule on the activation functions while traversing the different layers backwards.
   * Popular gradient descent optimizers include:
     - Standard stochastic gradient descent (SGD), possibly with batches.
     - Momentum SGD (to incorporate a moving average)
     - AdaGrad (with per-parameter learning rates)
     - RMSprop (adapting the learning rates based on RMS gradients)
     - Adam (combination of AdaGrad and RMSprop; also uses the second moment of weight gradients).
3. Splits of data
   * Training data; used for training.
   * Validation data; used to monitor learning and to adjust hyperparameters.
   * Test data; for final test of perfomance.
4. Training
   * It is critical that neither validation nor test data is used for adjusting the parameters.
   * Data is often split into bacthes such that gradients are computed for a batch of data.
   * A full pass of all data is known as an epoch. The validation score is evaluated at the end of each epoch.
   * Hyperparameters that can be tuned include:
     - The number of epochs (overfitting eventually)
     - The batch size (interplay between stochasticity and efficiency)
     - Learning rates.
     
In conclusion, there are scary many options when designing an ANN. A physicist should always have the ambition to learn about the model itself.


<!-- !split  -->
## A top-down perspective on Neural networks

The first thing we would like to do is divide the data into two or three
parts. A training set, a validation or dev (development) set, and a
test set. 
* The training set is used for learning and adjusting the weights.
* The dev/validation set is a subset of the training data. It is used to

check how well we are doing out-of-sample, after training the model on
the training dataset. We use the validation error as a proxy for the
test error in order to make tweaks to our model, e.g. changing hyperparameters such as the learning rate.
* The test set will be used to test the performance of or predictions with the final neural net. 

It is crucial that we do not use any of the test data to train the algorithm. This is a cardinal sin in ML. T If the validation and test sets are drawn from the same distributions, then a good performance on the validation set should lead to similarly good performance on the test set. 

However, sometimes
the training data and test data differ in subtle ways because, for
example, they are collected using slightly different methods, or
because it is cheaper to collect data in one way versus another. In
this case, there can be a mismatch between the training and test
data. This can lead to the neural network overfitting these small
differences between the test and training sets, and a poor performance
on the test set despite having a good performance on the validation
set. To rectify this, Andrew Ng suggests making two validation or dev
sets, one constructed from the training data and one constructed from
the test data. The difference between the performance of the algorithm
on these two validation sets quantifies the train-test mismatch. This
can serve as another important diagnostic when using DNNs for
supervised learning.

<!-- !split -->
## Limitations of supervised learning with deep networks

Like all statistical methods, supervised learning using neural
networks has important limitations. This is especially important when
one seeks to apply these methods, especially to physics problems. Like
all tools, DNNs are not a universal solution. Often, the same or
better performance on a task can be achieved by using a few
hand-engineered features (or even a collection of random
features). 

Here we list some of the important limitations of supervised neural network based models. 



* **Need labeled data**. All supervised learning methods, DNNs for supervised learning require labeled data. Often, labeled data is harder to acquire than unlabeled data (e.g. one must pay for human experts to label images).
* **Supervised neural networks are extremely data intensive.** DNNs are data hungry. They perform best when data is plentiful. This is doubly so for supervised methods where the data must also be labeled. The utility of DNNs is extremely limited if data is hard to acquire or the datasets are small (hundreds to a few thousand samples). In this case, the performance of other methods that utilize hand-engineered features can exceed that of DNNs.
* **Homogeneous data.** Almost all DNNs deal with homogeneous data of one type. It is very hard to design architectures that mix and match data types (i.e.\ some continuous variables, some discrete variables, some time series). In applications beyond images, video, and language, this is often what is required. In contrast, ensemble models like random forests or gradient-boosted trees have no difficulty handling mixed data types.
* **Many problems are not about prediction.** In natural science we are often interested in learning something about the underlying distribution that generates the data. In this case, it is often difficult to cast these ideas in a supervised learning setting. While the problems are related, it is possible to make good predictions with a *wrong* model. The model might or might not be useful for understanding the underlying science.

Some of these remarks are particular to DNNs, others are shared by all supervised learning methods. This motivates the use of unsupervised methods which in part circumvent these problems.


## Exercises

```{exercise} Weights and signal propagation of a simple neural netwrok
:label: exercise:NeuralNet:simple-network

'{figure} ./figs/fig-NeuralNet-simple.pdf`

Consider a neural network with two nodes in the input layer, one hidden layer with three nodes, and an output layer with one node (see figure). All active nodes (i.e., those that map input signals to an output via an activation function) employ the same activation function $f(z) = 1 / (1 + e^{-z})$ and include a bias weight when computing the activation $z$.

a) How many free (trainable) parameters does this neural network have?
b) Write an explicit expression for the output signal $y$ from the output node as a function of its weights $(w^2_{11}, w^2_{12},w^2_{13})$, bias weight $b^2_1$, and the signals $(y^1_1, y^1_2,y^1_3)$ coming from the nodes in the hidden layer.
c) Write a vector expression for the activations $(z^1_1, z^1_2, z^1_3)$ of the three nodes in the hidden layer given a vector of inputs $(x_1, x_2)$ and a matrix that contains the weights of the hidden layer and a vector that contains the bias weights. Be explicit how the weights are ordered in the matrix such that the activation vector can be obtained with matrix-vector operations.
d) How can the above expression be generalized to handle a scenario when there are four instances of input data?

$$
\boldsymbol{X} = \begin{pmatrix}
x^{(1)}_1 & x^{(1)}_2  \\
x^{(2)}_1 & x^{(2)}_2  \\
x^{(3)}_1 & x^{(3)}_2  \\
x^{(4)}_1 & x^{(4)}_2  
\end{pmatrix}
$$
```

## Solutions

```{solution} exercise:NeuralNet:simple-network
:label: solution:NeuralNet:simple-network
:class: dropdown

a) Trainable parameters are in the active nodes which, in this case, are the nodes in the hidden layer and the output layer. There are three weights per node in the hidden layer (two linear ones multiplying the two input signals and one bias) and four weights in the output node (three linear weights and one bias). Therefore, in total, there are $3 \times 3 + 4 = 13$.
b) $y = 1 / (1+e^{-z})$, where $z = w^2_{11} y^1_1 + w^2_{12} y^1_2 + w^2_{13} y^1_3 + b^2_1$.
c) $\boldsymbol{z}^{1} = \boldsymbol{x} \cdot \boldsymbol{W}^{1} + \boldsymbol{b}^{1}$, where $\boldsymbol{z}^{1} = (z^1_1, z^1_2, z^1_3)$, $\boldsymbol{x} = (x_1, x_2)$, $\boldsymbol{b}^{1} = (b^1_1, b^1_2,b^1_3)$ are all row vectors and $\boldsymbol{W}^{1} = \begin{pmatrix}
w^{1}_{11} & w^{1}_{12} & w^{1}_{13} \\
w^{1}_{21} & w^{1}_{22} & w^{1}_{23}
\end{pmatrix}$ is a $2 \times 3$ matrix.
d) It would look the same $\boldsymbol{Z}^{1} = \boldsymbol{X} \cdot \boldsymbol{W}^{1} + \boldsymbol{b}^{1}$ but $\boldsymbol{Z}^{1}$ would now be a $4 \times 3$ matrix with each row containing the hidden layer node activations for the corresponding instance of input data (row in $\boldsymbol{X}$). Applying the activation function $f(z)$ to each element of this matrix would give the corresponding output signals $\boldsymbol{Y}^{1}$ as another $4 \times 3$ matrix.
```

