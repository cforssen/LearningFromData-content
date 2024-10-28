---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec:BayesianLinearRegression)=
# Bayesian Linear Regression

```{epigraph}
> “La théorie des probabilités n'est que le bon sens réduit au calcul” 

(trans.) Probability theory is nothing but common sense reduced to calculation.

-- Pierre Simon de Laplace
```

In this chapter we use Bayes' theorem to infer a (posterior) probability density function for parameters of the linear model, conditional on $\data$. This approach expands on the ordinary linear regression method outlined in {numref}`sec:LinearModels`. If you have not read that chapter yet, please do so now since we will build on this and also use some of the notation introduced there. 

The advantages of doing *Bayesian* instead of ordinary (frequentist) linear regression are many. The Bayesian approach yields a probability distribution for the unknown parameters and for future model predictions. It also enables us to make all assumptions explicit whereas the frequentist approach puts nearly all emphasis on the collected data. 

In the Bayesian approach to parameter estimation we start from Bayes' theorem {eq}`eq_bayes` which implies that we must make (prior) assumptions for the model parameters. In most realistic data analyses we will then have to resort to numerical evaluation (or sampling) of the posterior. However, certain combinations of likelihoods and priors facilitate analytical derivation of the posterior. In this chapter we will explore one such situation and also demonstrate how we can recover the results from an ordinary least squares approach with certain assumptions. A slightly more general approach involves so called **conjugate priors**. This class of probability distributions have clever functional relationships with corresponding likelihood distributions that facilitate analytical derivation. 


## Bayes' theorem for the normal linear model

Recall from {numref}`sec:LinearModels` that we are relating data $\data$ to the output of a linear model expressed in terms of its design matrix $\dmat$ and its model parameters $\pars$

\begin{equation}
\data = \dmat \pars + \residuals.
\end{equation}

For the special case of one dependent, response variable ($\output$) and a single independent variable ($\inputt$) the data set ($\data$) and the residual vector ($\residuals$) are both $N_d \times 1$ column vector with $N_d$ the length of the data set. The design matrix ($\dmat$) has dimension $N_d \times N_p$ and the parameter vector ($\pars$) is $N_p \times 1$.

In linear regression using the ordinary least-squares method we made a leap of faith and decided that we were seeking a "best" model with an optimal set of parameters $\pars^*$ that minimizes the  Euclidean norm of the residual vector $\residuals$. We then found that these were given by the normal equation {eq}`eq:NormalEquation` with solution {eq}`eq:LinearModels:OLS_optimum`

$$
\pars^* =\left(\dmat^T\dmat\right)^{-1}\dmat^T\data.
$$

Let us instead consider a statistical model for these residuals which describe the mismatch between our model and observations as in Eq. {eq}`eq:DataModelsPredictions:mismatch`. Knowledge (and/or assumptions) concerning measurement uncertainties, or modeling errors, then allows to describe the residuals as a vector of random variables that are distributed according to a PDF

\begin{equation}
\residuals \sim \pdf{\residuals}{I},
\end{equation}

where we introduce the relation $\sim$ to indicate how a (random) variable is *distributed*. 

A very common assumption is that errors are normally distributed with zero mean. As before we let $N_d$ denote the number of data points in the (column) vector $\data$. Introducing the $N_d \times N_d$ covariance matrix $\covres$ for the errors we then have

$$
\pdf{\residuals}{\covres, I} = \mathcal{N}(\zeros,\covres).
$$ (eq:BayesianLinearRegression:ResidualErrors)

Having such a statistical model for the errors makes it possible to derive an expression for the data likelihood $\pdf{\data}{\pars,\covres,I}$ (see below). Using Bayes' theorem {eq}`eq:BayesTheorem:bayes-theorem-for-data` we can then "invert" this conditional probability distribution and write the parameter posterior

$$
\pdf{\pars}{\data, \covres, I} = \frac{\pdf{\data}{\pars,\covres,I}\pdf{\pars}{I}}{\pdf{\data}{I}}.
$$ (eq:BayesianLinearRegression:BayesTheorem)

To evaluate this posterior we must have expressions for both factors in the numerator on the right-hand side: the likelihood $\pdf{\data}{\pars,\covres,I}$ and the prior $\pdf{\pars}{I}$. Note that the prior does not depend on the data and the error model. The denominator $\pdf{\data}{I}$, sometimes known as the evidence, becomes irrelevant for the task of parameter estimation since it does not depend on $\pars$. It is typically quite challenging, if not impossible, to evaluate the evidence for a multivariate inference problem unless in some very special cases. In this chapter we will only be dealing with analytically tractable problems and will therefore (in principle) be able to evaluate also the evidence.

```{admonition} Discuss
- Why is it possible to perform parameter estimation without computing the evidence? 
- Can you think of why it is so challenging to compute the evidence? 
```

## The likelihood

Assuming normally distributed residuals it turns out to be straightforward to express the data likelihood. In the following we will make the further assumption that errors are *independent*. This implies that the covariance matrix is diagonal and given by a vector $\sigmas$,

$$
\covres &= \mathrm{diag}(\sigmas^2), \, \text{where} \\ 
\sigmas^2 &= \left( \sigma_0^2, \sigma_1^2, \ldots, \sigma_{N_d-1}^2\right),
$$ (eq:BayesianLinearRegression:independent_errors)

and $\sigmai^2$ is the variance for residual $\epsilon_i$. 

Let's first consider a single data $\data_i$ and the corresponding model prediction $M_i = \left( \dmat \pars \right)_i$. We are interested in the likelihood for this single data point

\begin{equation}
\pdf{\data_i}{\pars,\sigmai^2,I}.
\end{equation}

We can follow the recipe in {numref}`Chapter {number}: {name} <sec:BayesianAdvantages:ChangingVariables>`, since the relation between data and residual is a simple linear transformation $\data_i = \modeloutput_i + \varepsilon_i$, and find 

\begin{align}
\pdf{\data_i}{\pars,\sigmai^2,I} &= \pdf{\varepsilon_i = \data_i - \modeloutput_i}{\pars,\sigmai^2,I} \left| \frac{d \varepsilon_i}{d \data_i} \right|\\
&= \frac{1}{\sqrt{2\pi}\sigmai} \exp \left[ -\frac{(\data_i - \modeloutput_i)^2}{2\sigmai^2} \right]
\end{align}

where we used that $\epsilon_i \sim \mathcal{N}(0,\sigmai^2)$. Note that the parameter dependence sits in $\modeloutput_i \equiv \modeloutput(\pars, \inputs_i)$.

Furthermore, since we assume that the residuals are independent we find that the total likelihood becomes a product of the individual ones

$$
\pdf{\data}{\pars,\sigmas^2,I} &= \prod_{i=0}^{N_d-1} \pdf{\data_i}{\pars,\sigmai^2,I} \\
&= \left(\frac{1}{2\pi}\right)^{N_d/2} \frac{1}{\left| \covres \right|^{1/2}} \exp\left[ -\frac{1}{2} (\data - \dmat \pars)^T \covres^{-1} (\data - \dmat \pars) \right],
$$ (eq_normal_likelihood)

where we note that the diagonal form of $\covres$ implies that $\left| \covres \right|^{1/2} = \prod_{i=0}^{N_d-1} \sigmai$ and that the exponent becomes a sum of squared and weighted residual terms

$$
-\frac{1}{2} (\data - \dmat \pars)^T \covres^{-1} (\data - \dmat \pars) = -\frac{1}{2} \sum_{i=0}^{N_d - 1} \frac{(\data_i - (\dmat \pars)_i)^2}{\sigma_i^2}.
$$

In the special case that all residuals are both *independent and identically distributed* (i.i.d.) we have that all variances are the same, $\sigmai^2 = \sigmares^2$, and the full covariance matrix is completely specified by a single parameter $\sigmares^2$. For this special case, the likelihood becomes

$$
\pdf{\data}{\pars,\sigmares^2,I} = \left(\frac{1}{2\pi\sigmares^2}\right)^{N_d/2} \exp\left[ -\frac{1}{2\sigmares^2} \sum_{i=0}^{N_d - 1} (\data_i - (\dmat \pars)_i)^2 \right].
$$ (eq_normal_iid_likelihood)

```{caution} 
For computational performance it is always better (if possible) to write sums, such as the one in the exponent of {eq}`eq_normal_iid_likelihood`, in the form of vector-matrix operations rather than as for-loops. This particular sum should therefore be implemented as $(\data - \dmat \pars)^T (\data - \dmat \pars)$ to employ powerful optimization for vectorized operations in existing numerical libraries (such as [`numpy`](https://numpy.org/) in `python` and [`gsl`](https://www.gnu.org/software/gsl/), [`mkl`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) for C and other compiled programming languages).
```

```{admonition} Two views on the likelihood
Since observed data is generated stochastically, through an underlying "data-generating process", it is appropriately described by a probabibility distribution. This is the "data likelihood" which describes the probability distribution for observed data given a specific data-generating process (as indicated by the information on the right-hand side of the conditional). 

- View 1: Assuming fixed values of $\pars$; what are long-term frequencies of future data observations as described by the likelihood? 
- View 2: Focusing on the data $\data_\mathrm{obs}$ that we have; how does the likelihood for this data set depend on the values of the model parameters?

This second view is the one that we will be adopting when allowing model parameters to be associated with probability distributions. The likelihood still describes the probability for observing a set of data, but we emphasize its parameter dependence by writing

\begin{equation}
\pdf{\data}{\pars,\sigma^2,I} = \mathcal{L}(\pars).
\end{equation}

This function is **not** a probability distribution for model parameters. The parameter posterior, left-hand side of Eq. {eq}`eq:BayesianLinearRegression:BayesTheorem`, regains status as a probability density for $\pars$ since the likelihood is multiplied with the prior $\pdf{\pars}{I}$ and normalized by the evidence $\pdf{\data}{I}$.
```

## The prior

Next we assign a prior probability $\pdf{\pars}{I}$ for the model parameters. In order to facilitate analytical expressions we will explore two options: (i) a very broad, uniform prior, and (ii) a Gaussian prior. For simplicity, we consider both these priors to have zero mean and with all model parameters being i.i.d. 

The uniform prior for the $N_p$ parameters is then

$$
\pdf{\pars}{I} = \frac{1}{(\Delta\para)^{N_p}} \left\{ 
\begin{array}{ll}
1 & \text{if all } \para_i \in [-\Delta\para/2, +\Delta\para/2] \\
0 & \text{else},
\end{array}
\right.
$$ (eq:BayesianLinearRegression:uniform_iid_prior)

with $\Delta\para$ the width of the prior range in all parameter directions. 

The Gaussian prior that we will also be exploring is

$$
\pdf{\pars}{I} = \left(\frac{1}{2\pi\sigma_\para^2}\right)^{N_p/2} \exp\left[ -\frac{1}{2}\frac{\pars^T\pars}{\sigma_\para^2} \right],
$$ (eq:BayesianLinearRegression:gaussian_iid_prior)

with $\sigma_\para$ the standard deviation of the prior for all parameters.

## The posterior

Given the likelihood with i.i.d. errors {eq}`eq_normal_iid_likelihood` and the two alternative priors, {eq}`eq:BayesianLinearRegression:uniform_iid_prior` and {eq}`eq:BayesianLinearRegression:gaussian_iid_prior`, we will derive the corresponding two different expressions for the posterior (up to multiplicative normalization constants). 

### Rewriting the likelihood

First, let us rewrite the likelihood in a way that is made possible by the fact that we are considering a linear model. Given the quadratic dependence on model parameters in the exponent one can show (by performing a Taylor expansion of the log likelihood around the mode) that the likelihood becomes proportional to the functional form of a multivariate normal distribution for the model parameters

$$
\pdf{\data}{\pars,\sigmares^2,I} = \pdf{\data}{\optpars,\sigmares^2,I} \exp\left[ -\frac{1}{2} (\pars-\optpars)^T \covrespars^{-1} (\pars-\optpars) \right].
$$ (eq:BayesianLinearRegression:likelihood_pars)

Note that this expression still describes a probability distribution for the data. The data dependence sits in the amplitude of the mode, $\pdf{\data}{\optpars,\sigmares^2,I}$, and its position, $\optpars = \optpars(\data) = \left(\dmat^T\dmat\right)^{-1}\dmat^T\data$. The latter is the solution {eq}`eq:LinearModels:OLS_optimum` of the normal equation. Furthermore, the statistical model for the errors (here with $\covres = \mathrm{diag}(\sigmares^2)$) enter in the covariance matrix,

$$
\covrespars^{-1} = \frac{\dmat^T\dmat}{\sigmares^2},
$$ (eq:BayesianLinearRegression:likelihood_hessian)

which can be understood as the curvature (Hessian) of the negative log-likelihood.

```{exercise} Prove the Gaussian likelihood
:label: exercise:BayesianLinearRegression:likelihood_pars

Prove Eq. {eq}`eq:BayesianLinearRegression:likelihood_pars`. 
```

### Posterior with a uniform prior

Let us first consider a uniform prior as expressed in Eq. {eq}`eq:BayesianLinearRegression:uniform_iid_prior`. The prior can be considered very broad if its boundaries $\pm \Delta\para/2$ are very far from the mode of the likelihood {eq}`eq:BayesianLinearRegression:likelihood_pars`. "Distance" in this context is measured in terms of standard deviations. A "far distance", therefore, implies that $\pdf{\data}{\pars,\sigmares^2,I}$ is very close to zero. This implies that the posterior

\begin{equation}
\pdf{\pars}{\data,\sigmares^2,I} \propto \pdf{\data}{\pars,\sigmares^2,I} \pdf{\pars}{I},
\end{equation}

simply becomes proportional to the data likelihood (with the prior just truncating the distribution at very large distances). Thus we find from Eq. {eq}`eq:BayesianLinearRegression:likelihood_pars`

$$
\pdf{\pars}{\data,\sigmares^2,I} \propto \exp\left[ -\frac{1}{2} (\pars-\optpars)^T \covrespars^{-1} (\pars-\optpars) \right],
$$ (eq:BayesianLinearRegression:posterior_with_iid_uniform_prior)

if all $\para_i \in [-\Delta\para/2, +\Delta\para/2]$ while it is zero elsewhere. The mode of this distribution is obviously the mean vector $\optpars = \optpars(\data)$. We can therefore say that we have recovered the ordinary least-squares result. At this stage, however, the interpretation is that this parameter optimum corresponds to the maximum of the posterior PDF {eq}`eq:BayesianLinearRegression:posterior_with_iid_uniform_prior`. Such an optimum is sometimes known as the maximum a posteriori, or MAP.

```{admonition} Discuss
In light of this result, what assumption(s) are implicit in linear regression while they are made explicit in Bayesian linear regression?
```


### Posterior with a Gaussian prior

Assigning instead a Gaussian prior for the model parameters, as expressed in Eq. {eq}`eq:BayesianLinearRegression:gaussian_iid_prior`, we find that the posterior is proportional to the product of two exponential functions

$$
\pdf{\pars}{\data,\sigmares^2,I} &\propto \exp\left[ -\frac{1}{2} (\pars-\optpars)^T \covrespars^{-1} (\pars-\optpars) \right] \exp\left[ -\frac{1}{2}\frac{\pars^T\pars}{\sigma_{\para}^2} \right] \\
&\propto \exp\left[ -\frac{1}{2} (\pars-\tilde{\pars})^T \tildecovpars^{-1} (\pars-\tilde{\pars}) \right].
$$ (eq:BayesianLinearRegression:posterior_with_iid_gaussian_prior)

The second proportionality is a consequence of both exponents being quadratic in the model parameters, and therefore that the full expression looks like the product of two Gaussians. This product is proportional to another Gaussian distribution which has mean vector and (inverse) covariance matrix given by

$$
\tilde{\pars} &= \tildecovpars \covrespars^{-1} \optpars \\
\tildecovpars^{-1} &= \covrespars^{-1} + \sigma_{\para}^{-2} \boldsymbol{1} 
$$ (eq:BayesianLinearRegression:posterior_pars_with_iid_gaussian_prior)

where $\boldsymbol{1}$ is the $N_p \times N_p$ unit matrix. In effect, what has happend is that the prior normal distribution becomes updated to a posterior normal distribution via an inference process that involves a data likelihood. In this particular case, learning from data implies that the mode changes from $\boldsymbol{0}$ to $\tilde{\pars}$ and the covariance from a diagonal structure with $\sigma_{\para}^2$ in all directions to the covariance matrix $\tildecovpars$.

```{admonition} Discuss
What happens if the data is of high quality (i.e., the likelihood $\mathcal{L}(\pars)$ is sharply peaked around $\optpars$), and what happens if it is of poor quality (providing a very broad likelihood distribution)?
```

### Marginal posterior distributions

Given a multivariate probability distribution we are often interested in lower dimension, marginal distributions. Consider for example $\pars^T = [\pars_1^T, \pars_2^T$], that is partitioned into respective dimensions $D_1$ and $D_2$. The marginal distribution corresponds to the integral

$$
\p{\pars_2} = \int d\pars_1 \p{\pars}.
$$


```{admonition} Transformation property of multivariate normal distributions
Let $\mathbf{Y}$ be a multivariate normal-distributed random variable of length $N_p$ with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$. We use the notation $\psub{\mathbf{Y}}{\mathbf{y}} = \mathcal{N} (\mathbf{y} | \mathbf{\mu}, \mathbf{\Sigma})$ to emphasize which variable that is normally distributed.

Consider now a general $N_p \times N_p$ matrix $\boldsymbol{A}$ and $N_p \times 1$ vector $\boldsymbol{b}$. Then, the random variable $\mathbf{Z} = \boldsymbol{A} \mathbf{Y} + \boldsymbol{b}$ is also multivariate normal-distributed with the PDF

$$
\psub{\mathbf{Z}}{\mathbf{z}} = \mathcal{N} (\mathbf{z} \vert \mathbf{A}\boldsymbol{\mu} + \boldsymbol{b},\boldsymbol{A}\boldsymbol{\Sigma}\boldsymbol{A}^T).
$$ (eq:BayesianLinearRegression:transformed-normal)
```

For multivariate normal distributions we can employ a useful transformation property, shown in Eq. {eq}`eq:BayesianLinearRegression:transformed-normal`. Considering the posterior {eq}`eq:BayesianLinearRegression:posterior_with_iid_gaussian_prior` we partition the parameters $\pars^T = [\pars_1^T, \pars_2^T$] and the mean vector and covariance matrix into $\boldsymbol{\mu}^T = [\boldsymbol{\mu}_1^T,\boldsymbol{\mu}_2^T]$ and

$$
\boldsymbol{\Sigma} = \left[
    \begin{array}{cc}
        \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\	
        \boldsymbol{\Sigma}_{12}^T & \boldsymbol{\Sigma}_{22}
    \end{array}
\right].
$$

We can obtain the marginal distribution for $\pars_2$ by setting

$$
\mathbf{A} = \left[
    \begin{array}{cc}
        0 & 0 \\
        0 & \mathbf{1}_{D_2\times D_2}
    \end{array}
\right], \,\, \mathbf{b} = 0,
$$

which yields 

$$
\pdf{\pars_2}{\data, I} = 
\mathcal{N}(\pars_2|\boldsymbol{\mu}_2,\boldsymbol{\Sigma}_{22}).
$$ (eq_marginal_N)

(sec:ppd)=
## The posterior predictive

One can also derive the posterior predictive distribution (PPD), i.e., the probability distribution for predictions $\widetilde{\boldsymbol{\mathcal{F}}}$ given the model $M$ and a set of new inputs for the independent variable $\boldsymbol{x}$. The new inputs give rise to a new design matrix $\widetilde{\dmat}$.

We obtain the posterior predictive distribution by marginalizing over the uncertain model parameters that we just inferred from the given data $\data$.

$$
\pdf{\widetilde{\boldsymbol{\mathcal{F}}}}{\data}
\propto \int \pdf{\widetilde{\boldsymbol{\mathcal{F}}}}{\pars,\sigmares^2,I}  \pdf{\pars}{\data,\sigmares^2,I}\, d\pars,
$$ (eq:BayesianLinearRegression:ppd_pdf)

where both distributions in the integrand can be expressed as Gaussians. Alternatively, one can express the PPD as the set of model predictions with the model parameters distributed according to the posterior parameter PDF

$$
\left\{ \widetilde{\dmat} \pars \, : \, \pars \sim \pdf{\pars}{\data,\sigmares^2,I} \right\}.
$$ (eq:BayesianLinearRegression:ppd_pdf_set)

This set of predictions can be obtained if we have access to a set of samples from the parameter posterior.

(sec:warmup)=
## Bayesian linear regression: warmup

To warm up, we consider the same situation as in [](sec:ols_warmup).

For the time being we assume to know enough about the data to consider a normal likelihood with i.i.d. errors. Let us first set the known residual variance to $\sigmares^2 = 0.5^2$. 

This time we also have prior knowledge that we would like to build into the inference. Here we use a normal prior for the parameters with $\sigma_\para = 5.0$, which is to say that before looking at the data we believe $\pars$ to be centered on zero with a variance of $5^2$.

Let us plot this prior. The prior is the same for $\theta_0$ and $\theta_1$, so it is enough to plot one of them. 

```{code-cell} python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def normal_distribution(mu,sigma2):
    return norm(loc=mu,scale=np.sqrt(sigma2))

thetai = np.linspace(-10,10,100)
prior = normal_distribution(0,5.0**2)

fig, ax = plt.subplots(1,1)
ax.plot(thetai,prior.pdf(thetai))
ax.set_ylabel(r'$p(\theta_i \vert I )$')
ax.set_xlabel(r'$\theta_i$');
```

It is straightforward to evaluate Eq. {eq}`eq:BayesianLinearRegression:posterior_pars_with_iid_gaussian_prior`, which gives us

$$
\tildecovpars^{-1} &=  4 \begin{pmatrix} 2.01 & -1.0 \\ -1.0 & 5.01 \end{pmatrix} \\
\tilde{\pars} &= ( 0.992, 1.994)
$$ (eq_warmup_results)

This should be compared with the parameter vector $(1,2)$ we recovered using ordinary linear regression. With Bayesian linear regression we start from an informative prior with both parameters centered on zero with a rather large variance.

```{exercise} Warm-up Bayesian linear regression
:label: exercise:BayesianLinearRegression:warmup

Reproduce the posterior mean and covariance matrix from Eq. {eq}`eq_warmup_results`. You can use `numpy` methods to perform the linear algebra operations.
```

We can plot the posterior probability distribution for $\pars$, i.e., by plotting the bi-variate $\mathcal{N}-$distribution with the parameter in Eq. {eq}`eq_warmup_results`. 

````{code-cell} python3

from scipy.stats import multivariate_normal

mu = np.array([0.992,1.992])
Sigma = np.linalg.inv(4 * np.array([[2.01,-1.0],[-1.0,5.01]]))

posterior = multivariate_normal(mean=mu, cov=Sigma)

theta0, theta1 = np.mgrid[-0.5:2.5:.01, 0.5:3.5:.01]
theta_grid = np.dstack((theta0, theta1))

fig,ax = plt.subplots(1,1)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
im = ax.contourf(theta0, theta1, posterior.pdf(theta_grid),cmap=plt.cm.Reds);
fig.colorbar(im);
````

Using Eq. {eq}`eq_marginal_N` we can obtain, e.g., the $\theta_1$ marginal density and compare with the prior

````{code-cell} python3

theta1 = np.linspace(-0.5,4.5,50)
mu1 = mu[1]
Sigma11_sq = Sigma[1,1]

posterior1 = normal_distribution(mu1,Sigma11_sq)

fig, ax = plt.subplots(1,1)
ax.plot(theta1,posterior1.pdf(theta1),'r-',\
label=r'$p(\theta_1 \vert \mathcal{D}, \sigma_\epsilon^2, I )$')
ax.plot(theta1,prior.pdf(theta1), 'b--',label=r'$p(\theta_1 \vert I )$')
ax.set_ylabel(r'$p(\theta_1 \vert \ldots )$')
ax.set_xlabel(r'$\theta_1$')
ax.legend(loc='best');
````

The key take-away with this numerical exercise is that Bayesian inference yields a probability distribution for the model parameters whose values we are uncertain about. With ordinary linear regression techniques you only obtain the parameter values that optimize some cost function, and not a probability distribution. 

```{exercise} Warm-up Bayesian linear regression (data errors)
:label: exercise:BayesianLinearRegression:warmup_errors

Explore the sensitivity to changes in the residual errors $\sigmares$. Try to increase and reduce the error.
```

```{exercise} Warm-up Bayesian linear regression (prior sensitivity)
:label: exercise:BayesianLinearRegression:warmup_priors

Explore the sensitivity to changes in the Gaussian prior width $\sigma_\para$. Try to increase and reduce the width.
```

```{exercise} "In practice" Bayesian linear regression 
:label: exercise:BayesianLinearRegression:in_practice

Perform Bayesian Linear Regression on the data that was generated in [](sec:ols_in_practice). Explore:
- Dependence on the quality of the data (generate data with different $\sigma_\epsilon$) or the number of data.
- Dependence on the polynomial function that was used to generate the data.
- Dependence on the number of polynomial terms in the model.
- Dependence on the parameter prior.

In all cases you should compare the Bayesian inference with the results from Ordinary Least Squares and with the true parameters that were used to generate the data.
```


## Solutions

```{solution} exercise:BayesianLinearRegression:likelihood_pars
:label: solution:BayesianLinearRegression:likelihood_pars
:class: dropdown

Hints:

1. Identify $\optpars$ as the position of the mode of the likelihood by inspecting the negative log-likelihood $L(\pars)$ and comparing with the derivation of the normal equation.
2. Taylor expand $L(\pars)$ around $\optpars$. For this you need to argue (or show) that the gradient vector $\nabla_{\pars} L(\pars) = 0$ at $\pars=\optpars$, and show that the Hessian $\boldsymbol{H}$ (with elements $H_{ij} = \frac{\partial^2 L}{\partial\para_i\partial\para_j}$) is a constant matrix $\boldsymbol{H} = \frac{\dmat^T\dmat}{\sigmares^2}$.
3. Compare with the Taylor expansion of a normal distribution $\mathcal{N}\left( \pars \vert \optpars, \covpars \right)$.

Full solution:

- The likelihood can be written $\pdf{\data}{\pars,I} = \exp\left[ -L(\pars) \right]$, where we include information on the error distribution ($\sigmares$) in the conditional $I$. The negative log-likelihood, including the normalization factor, is

$$
L(\pars) = \frac{N_d}{2}\log(2\pi\sigmares^2) + \frac{1}{2\sigmares^2} \sum_{i=0}^{N_d - 1} (\data_i - (\dmat \pars)_i)^2.
$$

- Comparing with Eq. {eq}`eq:LinearRegression:cost-function` and the corresponding gradient vector {eq}`eq:LinearRegression:gradient` we find that

  $$
  \nabla_{\pars} L(\pars) = -\frac{\dmat^T\left( \data-\dmat\pars\right)}{\sigmares^2},
  $$

  which is zero at $\pars = \optpars = \left(\dmat^T\dmat\right)^{-1}\dmat^T\data$ corresponding to the solution of the normal equation.

- We can Taylor expand $L(\pars)$ around $\pars=\optpars$ realizing that the linear (gradient) term is zero. Furthermore, the quadrating term depends on the second derivative (hessian) which is a constant matrix since $L$ only depends quadratically on the parameters

$$
H = \Delta L = \nabla_{\pars} \cdot (\nabla_{\pars} L(\pars)) = \frac{\dmat^T\dmat}{\sigmares^2}
$$

- Since higher derivatives therefore must be zero, the Taylor expansion actually terminates at second order

$$
L(\pars) = L(\optpars) + \frac{1}{2} (\pars-\optpars)^T \frac{\dmat^T\dmat}{\sigmares^2} (\pars-\optpars)
$$

- We introduce $\covrespars^{-1} \equiv {\dmat^T\dmat} / {\sigmares^2}$ and use that $\exp\left[ - L(\optpars) \right] = \pdf{\data}{\optpars,I}$. Therefore, evaluating $\exp\left[ -L(\pars) \right]$ gives

  $$
  \pdf{\data}{\pars,I} = \pdf{\data}{\optpars,I} \exp\left[ -\frac{1}{2} (\pars-\optpars)^T \covrespars^{-1} (\pars-\optpars) \right],
  $$

  as we wanted to show.
```



