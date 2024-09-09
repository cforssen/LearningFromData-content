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

```{math}
\newcommand{\residuals}{\boldsymbol{\epsilon}}
\newcommand{\zeros}{\boldsymbol{0}}
\newcommand{\covres}{\boldsymbol{\Sigma_\epsilon}}
\newcommand{\covpars}{\boldsymbol{\Sigma_\pars}}
\newcommand{\tildecovpars}{\boldsymbol{\widetilde{\Sigma}_\pars}}
\newcommand{\sigmas}{\boldsymbol{\sigma}}
\newcommand{\sigmai}{\sigma_i}
\newcommand{\sigmares}{\sigma_\epsilon}
```

(sec:BayesianLinearRegression)=
# Bayesian Linear Regression

```{epigraph}
> “La théorie des probabilités n'est que le bon sens réduit au calcul” 

(trans.) Probability theory is nothing but common sense reduced to calculation.

-- Pierre Simon de Laplace
```

In this chapter we use Bayes' theorem to infer a (posterior) probability density function for parameters of the linear model, conditional on $\data$. This approach expands on the ordinary linear regression method outlined in the chapter on [](sec:LinearModels). If you have not read that chapter yet, please do so now since we will build on this and also use some of the notation introduced there. A large fraction of the text in this chapter was written by Andreas Ekström.

The advantages of doing *Bayesian* instead of ordinary (frequentist) linear regression are many. The Bayesian approach yields a probability distribution for the unknown parameters and for future model predictions. It also enables us to make all assumptions explicit whereas the frequentist approach puts nearly all emphasis on the collected data. 

In the Bayesian approach we start from Bayes' theorem {eq}`eq_bayes` which implies that we must make (prior) assumptions for the model parameters. In most cases we will then have to resort to numerical evaluation (or sampling) of the posterior. However, certain combinations of likelihoods and priors facilitate analytical derivation of the posterior. In this chapter we will explore one such situation and also show how we can recover the results from the ordinary least squares approach. A slightly more general situation is explored in the next chapter where we introduce so called **conjugate priors** that have a clever functional relationship with the relevant likelihood that again facilitate analytical derivation. 


## Bayes' theorem for the normal linear model

Recall from [](sec:LinearModels) that we are relating a (column) vector of data $\data$ to a linear model expressed in terms of its design matrix $\dmat$ and (column) vector of model parameters $\pars$

\begin{equation}
\data = \dmat \pars + \boldsymbol{\epsilon}.
\end{equation}

In linear regression we made a leap of faith and decided that we were seeking an optimal set of parameters $\pars^*$ that minimize the 2-norm of the residual vector $\boldsymbol{\epsilon}$. Let us instead consider these residuals to statistically describe the mismatch between our model and observations as in Eq. {eq}`eq:DataModelsPredictions:mismatch`. Knowledge (and/or assumptions) concerning the error terms then allows to assign a statistical model in which the residuals are (random) variables that are distributed according to a PDF

\begin{equation}
\residuals \sim \pdf{\residuals}{I},
\end{equation}

where we introduce the relation $\sim$ to indicate how a (random) variable is *distributed*. 

A very common assumption is that errors are normally distributed with zero mean. As before we let $N_d$ denote the number of data points in the (column) vector $\data$. Introducing the $N_d \times N_d$ error covariance matrix $\covres$ we then have

$$
\pdf{\residuals}{\covres, I} = \mathcal{N}(\zeros,\covres).
$$ (eq:BayesianLinearRegression:ResidualErrors)

Having an error model will make it possible to write the data likelihood and using Bayes' theorem we can write

$$
\pdf{\pars}{\data, \covres, I} = \frac{\pdf{\data}{\pars,\covres,I}\pdf{\pars}{I}}{\pdf{\data}{I}},
$$ (eq:BayesianLinearRegression:BayesTheorem)

where we have assumed that the error model given by $\covres$ is known.
To evaluate the posterior, i.e., the left-hand side, we must develop expressions for the factors in the numerator on the right-hand side: the likelihood $\pdf{\data}{\pars,\covres,I}$ and the prior $\pdf{\pars}{I}$. Note that the prior does not depend on the error model. The denominator $\pdf{\data}{I}$, sometimes known as the evidence, is an overall normalization constant that becomes irrelevant for the task of parameter estimation. It is typically quite challenging, if not impossible, to evaluate the evidence for a multivariate inference problem unless in some very special cases. In this chapter we will only be dealing with analytically tractable problems and will therefore (in principle) be able to evaluate also the evidence.

```{admonition} Discuss
Why is it irrelevant to compute the evidence when doing parameter estimation? 
```

```{admonition} Discuss
Can you think of why it is so challenging to compute the evidence? 
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

We can follow the recipe in [](sec:BayesianAdvantages:ChangingVariables), since the relation between data and residual is a simple transformation, and find 

\begin{align}
\pdf{\data_i}{\pars,\sigmai^2,I} &= \pdf{\varepsilon_i = \data_i - M_i}{\pars,\sigmai^2,I} \\
&= \frac{1}{\sqrt{2\pi}\sigmai} \exp \left[ -\frac{(\data_i - M_i)^2}{2\sigmai^2} \right]
\end{align}

where we used that $\epsilon_i \sim \mathcal{N}(0,\sigmai^2)$. Note that the parameter dependence sits in $M_i$.

Furthermore, since we assume that the residuals are independent we find that the total likelihood becomes a simple product of the individual ones

$$
\pdf{\data}{\pars,\sigmas^2,I} &= \prod_{i=0}^{N_d-1} \pdf{\data_i}{\pars,\sigmai^2,I} \\
&= \left(\frac{1}{2\pi}\right)^{N_d/2} \frac{1}{\left| \covres \right|^{1/2}} \exp\left[ -\frac{1}{2} (\data - \dmat \pars)^T \covres^{-1} (\data - \dmat \pars) \right],
$$ (eq_normal_likelihood)

where we note that the diagonal form of $\covres$ implies that the exponent becomes a sum of residual terms and that $\left| \covres \right|^{1/2} = \prod_{i=0}^{N_d-1} \sigmai$.

The likelihood is often viewed as a function of the parameters $\pars$ although one should note that this multivariate normal distribution is *not* normalized with respect to them since they are on the right hand side of the conditional. To emphasize the parameter dependence of the likelihood one sometimes denotes it as 

\begin{equation}
\pdf{\data}{\pars,\sigma^2,I} = \mathcal{L}(\pars).
\end{equation}

Upon multiplication with the prior $\pdf{\pars}{I}$ and normalization by the evidence $\pdf{\data}{I}$ the right-hand side of Eq. {eq}`eq:BayesianLinearRegression:BayesTheorem` regains the status of a probability density, as it should.

In the special case that all residuals are both *independent and identically distributed* (i.i.d.) we have that all variances are the same, $\sigmai^2 = \sigmares^2$, and the full covariance matrix is completely specified by a single parameter $\sigmares^2$. For this special case, the likelihood becomes

$$
\pdf{\data}{\pars,\sigmares^2,I} = \left(\frac{1}{2\pi\sigmares^2}\right)^{N_d/2} \exp\left[ -\frac{1}{2}\frac{(\data - \dmat \pars)^T(\data - \dmat \pars)}{\sigmares^2} \right].
$$ (eq_normal_iid_likelihood)

## The prior

Next we assign a prior probability $\pdf{\pars}{I}$ for the model parameters. In order to facilitate analytical expressions we will explore two options: (i) a very broad, uniform prior, and (ii) a Gaussian prior. For simplicity, we consider both these priors to have zero mean and with all parameters being i.i.d. 

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

Given the likelihood with i.i.d. errors {eq}`eq_normal_iid_likelihood` and the two alternative priors, {eq}`eq:BayesianLinearRegression:uniform_iid_prior` and {eq}`eq:BayesianLinearRegression:gaussian_iid_prior`, we will derive an expression for the posterior up to a multiplicative normalization constant. 

First, let us rewrite the likelihood in a way that is made possible by the fact that we are considering a linear model. Despite the fact that the likelihood is a PDF for the data one can show, via Taylor expansion around the mode, that it is proportional to a multivariate normal distribution for the model parameters

$$
\pdf{\data}{\pars,\sigmares^2,I} \propto \exp\left[ -\frac{1}{2} (\pars-\pars^*)^T \covpars^{-1} (\pars-\pars^*) \right],
$$ (eq:BayesianLinearRegression:likelihood_pars)

where $\pars^* = \left(\dmat^T\dmat\right)^{-1}\dmat^T\data$ is the solution {eq}`eq:LinearModels:OLS_optimum` of ordinary linear regression and 

$$
\covpars^{-1} = \frac{\dmat^T\dmat}{\sigmares^2},
$$ (eq:BayesianLinearRegression:likelihood_hessian)

is the Hessian with respect to model parameters of the negative log-likelihood.

```{exercise} Prove the Gaussian likelihood
:label: exercise:BayesianLinearRegression:likelihood_pars

Prove Eq. {eq}`eq:BayesianLinearRegression:likelihood_pars`. 
```

### Posterior with a uniform prior

Let us first consider a uniform prior as expressed in Eq. {eq}`eq:BayesianLinearRegression:uniform_iid_prior`. The prior can be considered very broad if its boundaries $\pm \Delta\para/2$ are very far from the mode of the likelihood {eq}`eq:BayesianLinearRegression:likelihood_pars`, where distance is measured in terms of standard deviations. This implies that the posterior

\begin{equation}
\pdf{\pars}{\data,\sigmares^2,I} \propto \pdf{\data}{\pars,\sigmares^2,I} \pdf{\pars}{I},
\end{equation}

becomes proportional to the data likelihood with the prior just truncating the distribution at very large distances (that contained a negligible probability mass). Thus we find

$$
\pdf{\pars}{\data,\sigmares^2,I} \propto \exp\left[ -\frac{1}{2} (\pars-\pars^*)^T \covpars^{-1} (\pars-\pars^*) \right],
$$ (eq:BayesianLinearRegression:posterior_with_iid_uniform_prior)

if all $\para_i \in [-\Delta\para/2, +\Delta\para/2]$ while it is zero elsewhere. The mode of this distribution is obviously the mean vector $\pars^*$. We can therefore say that we have recovered the ordinary least-squares result with the interpretation that this solution is the maximum of the posterior PDF (sometimes known as the maximum a posteriori, or MAP).

```{admonition} Discuss
In light of this result, what assumption(s) are implicit in linear regression while they are made explicit in Bayesian linear regression?
```


### Posterior with a Gaussian prior

Assigning instead a Gaussian prior as expressed in Eq. {eq}`eq:BayesianLinearRegression:gaussian_iid_prior` corresponds to a situation in which we have prior information on the expected magnitude of model parameters. The posterior is then proportional to the product of two multivariate normal distributions

$$
\pdf{\pars}{\data,\sigmares^2,I} &\propto \exp\left[ -\frac{1}{2} (\pars-\pars^*)^T \covpars^{-1} (\pars-\pars^*) \right] \exp\left[ -\frac{1}{2}\frac{\pars^T\pars}{\sigma_\para^2} \right] \\
&= \exp\left[ -\frac{1}{2} (\pars-\tilde\pars)^T \tildecovpars^{-1} (\pars-\tilde\pars) \right],
$$ (eq:BayesianLinearRegression:posterior_with_iid_gaussian_prior)

i.e., another Gaussian with covariance matrix and mean vector

$$
\tildecovpars^{-1} &= \covpars^{-1} + \sigma_\para^{-2} \boldsymbol{1} \\
\tilde\pars &= \tildecovpars \covpars^{-1} \pars^*
$$ (eq:BayesianLinearRegression:posterior_pars_with_iid_gaussian_prior)

where $\boldsymbol{1}$ is the $N_p \times N_p$ unit matrix. In effect, the prior distribution becomes the posterior one via an inference process that involves learning from data. In this particular case, the inference returns a Gaussian PDF with updated parameters where the mode changes from $\boldsymbol{0}$ to $\tilde\pars$ and the variance from $\sigma_\para^2$ in all directions to the covariance matrix $\tildecovpars$.

```{admonition} Discuss
What happens if the data is of high quality (sharply peaked around $\pars^*$), and what happens if it is of poor quality (providing a very broad likelihood distribution)?
```


### Marginal posterior distributions

Next, we will explore a useful transformation property of $\mathcal{N}-$distributions. Let $\mathbf{Y}$ be a multivariate $\mathcal{N}$-distributed random variable. Consider a matrix $\boldsymbol{A}$ and (column) vector $\boldsymbol{b}$. Then, the random variable $\mathbf{Z} = \boldsymbol{A} \mathbf{Y} + \boldsymbol{b}$ is also multivariate $\mathcal{N}$-distributed with the PDF

$$
\mathcal{N} (\mathbf{Z}|\mathbf{A}\boldsymbol{\mu} + \boldsymbol{b},\boldsymbol{A}\boldsymbol{\Sigma}\boldsymbol{A}^T),
$$

where we use the $\mathcal{N} (\mathbf{X} | \mathbf{\mu}, \mathbf{\Sigma})$ notation to emphasize which variable that is normally distributed.

From this we can obtain marginal $\mathcal{N}$-distributions, i.e., the distributions for which we have integrated over some $\pars$-directions. Assume we have a $(D_1+D_2)$-dimensional $\mathcal{N}$-distributed random variable $\mathbf{X} = [\mathbf{X}_1, \mathbf{X}_2]^T$, here partitioned into respective dimensions $D_1$ and $D_2$, with $\boldsymbol{\mu} = [\boldsymbol{\mu}_1,\boldsymbol{\mu}_2]^T$ and

$$
\boldsymbol{\Sigma} = \left[
    \begin{array}{cc}
        \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\	
        \boldsymbol{\Sigma}_{12}^T & \boldsymbol{\Sigma}_{22}
    \end{array}
\right].
$$

We can transform to either marginal distribution by setting, e.g.,

$$
\mathbf{A} = \left[
    \begin{array}{cc}
        0 & 0 \\
        0 & \mathbf{1}_{D_2\times D_2}
    \end{array}
\right], \,\, \mathbf{b} = 0
$$

This yields the marginal density 

$$
\mathcal{N}(\mathbf{X}_2|\boldsymbol{\mu}_2,\boldsymbol{\Sigma}_{22}).
$$ (eq_marginal_N)

(sec:ppd)=
## The posterior predictive

One can also derive the posterior predictive distribution (PPD), i.e., the probability distribution for predictions $\widetilde{\boldsymbol{\mathcal{F}}}$ given the model $M$ and a set of new inputs for the independent variable $\boldsymbol{x}$. The new inputs also give rise to a new design matrix $\widetilde{\dmat}$.

We obtain the posterior predictive distribution by marginalizing over the uncertain model parameters that we just inferred from the old data $\data$.

$$
\pdf{\widetilde{\boldsymbol{\mathcal{F}}}}{\data}
\propto \int \pdf{\widetilde{\boldsymbol{\mathcal{F}}}}{\pars,\sigmares^2,I}  \pdf{\pars}{\data,\sigmares^2,I}\, d\pars,
$$ (eq:BayesianLinearRegression:ppd_pdf)

where both distributions in the integrand can be expressed as Gaussians. Alternatively, one can express the PPD as the set of model predictions with the model parameters distributed according to the posterior parameter PDF

$$
\left\{ \widetilde{\dmat} \pars \, : \, \pars \sim \pdf{\pars}{\data,\sigmares^2,I} \right\}.
$$ (eq:BayesianLinearRegression:ppd_pdf_set)

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
\tilde\pars &= ( 0.992, 1.994)
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

1. Identify the solution $\pars^*$ as the maximum of the likelihood by introducing $L(\pars)$ as the negative log-likelihood.
2. Taylor expand $L(\pars)$ around $\pars^*$. For this you need the Hessian $\boldsymbol{H}$ with elements $H_{ij} = \left. \frac{\paratial^2 L}{\paratial\para_i\paratial\para_j} \right|_{\pars = \pars^*}$.
3. Compare with the Taylor expansion of a normal distribution $\mathcal{N}\left( \pars^*, \covpars \right)$.
```



