# Learning from data: Inference

```{epigraph}
> "The goal is to turn data into information, and information into insight."

-- Carly Fiorina, (former CEO of Hewlett-Packard)
```

The general problem that will be adressed in this series of lectures is illustrated in {numref}`fig-inference`. The learning process depicted there is known as **inference** and involves steps in reasoning to move from premises to logical consequences. 

<!-- <img src="./figs/inference.png" width=600><p><em>Learning from data is an inference process. <div id="fig-inference"></div></em></p> -->

```{figure} ./figs/inference.png
:name: fig-inference
:width: 600px
:align: center

Learning from data is an inference process.
```

(intro:inference)=
## Inference

```{admonition} Inference
  *"the act of passing from one proposition, statement or judgment considered as true to another whose truth is believed to follow from that of the former"* (Webster)<br/>
  Does the truth of premises $A, B, C$ imply the truth of hypothesis, $H$? 
  ```
```{admonition} Deductive inference
  The premises allow definite determination of the truth/falsity of $H$ .<br/>
  $\cprob{H}{A,B,C} = 0$ ($H$ must be false if $A,B,C$ are true), or <br/>
  $\cprob{H}{A,B,C} = 1$ ($H$ must be true if $A,B,C$ are true) <br/>
  (see {numref}`sec:Statistics`: {ref}`sec:Statistics` for a reminder on conditional probabilities)
  ```
```{admonition} Inductive inference
  The premises bear on the truth/falsity of $H$, but don’t allow its definite determination.<br/>
  $A, B, C, D$ share properties $x, y, z$; $E$ has properties $x, y$<br/>
  $\Rightarrow E$ probably has property $z$. 
  ```
In the natural sciences, the premise on which inference is built is often a finite set of measurements while the process of learning is usually achieved by confronting that data with scientific hypotheses and models. The conclusion might ultimately be falsification of a hypothesis or the invalidation of a phenomenological model. However, the end result will not be the ultimate determination of the truth of the hypothesis. More commonly, the conclusion might be an improved model that can be used for predictions of new phenomena. Thus, we are typically dealing with inductive inference.


```{figure} ./figs/scientific_wheel_data.png
:name: fig-scientific-wheel
:width: 400px
:align: center

This process of learning from data is fundamental to the scientific wheel of progress.
```


## Statistical inference

* Quantifies the strength of inductive inference from propositions. In science, these propositions are usually in the form of data ($D$) and other premises such as models and the target hypothesis is about the phenomena producing the data.
* The quantification is done via probabilities, or averages calculated using probabilities. Frequentists ($\mathcal{F}$) and Bayesians ($\mathcal{B}$) use probabilities very differently for this:
  * To the pioneers such as Bernoulli, Bayes and Laplace, a probability represented a *degree-of-belief* or plausability: how much they believed something as true based on the evidence at hand. This is the Bayesian approach.
  * To the 19th century scholars this seemed too vague and subjective. They redefined probability as the *long run relative frequency* with which an event occurred, given (infinitely) many repeated (experimental) trials.



## Machine learning

The basic process illustrated in {numref}`fig-inference` is employed also in the field of machine learning. Here, the learning part might take place when confronting a large set of data with a machine learning algorithm, and the specific aim might be tasks such as classification or clusterization. 
<!--<img src="./figs/MLinference.png" width=600> -->

```{figure} ./figs/MLinference.png
:name: fig-ML-inference
:width: 600px
:align: center

Machine learning can also be seen as an inference process.
```

On the one hand, we will study statistical inference methods for learning from data as needed for scientific applications. Simultaneously this statistical foundation will provide a deeper understanding (and probabilistic interpretation) of machine learning algorithms. This will allow to achieve **statistical learning**.

*Edwin Jaynes*, in his influential [How does the brain do plausible reasoning?](https://link.springer.com/chapter/10.1007%2F978-94-009-3049-0_1) {cite}`Jaynes1988`, wrote:
> One of the most familiar facts of our experience is this: that there is such a thing as common sense, which enables us to do plausible reasoning in a fairly consistent way. People who have the same background of experience and the same amount of information about a proposition come to pretty much the same conclusions as to its plausibility. No jury has ever reached a verdict on the basis of pure deductive reasoning. Therefore the human brain must contain some fairly definite mechanism for plausible reasoning, undoubtedly much more complex than that required for deductive reasoning. But in order for this to be possible, there must exist consistent rules for carrying out plausible reasoning, in terms of operations so definite that they can be programmed on the computing machine which is the human brain.

Jaynes went on to show that these "consistent rules" are just the axiomatic rules of Bayesian probability theory supplemented by Laplace's principle of indifference and, its generalization, Shannon's principle of maximum entropy. This key observation implies that we can define an *extended logic* such that a computer can be programmed to "reason", or rather to update probabilities based on data. Given some very minimal desiderata, the rules of Bayesian probability are the only ones which conform to what, intuitively, we recognize as rationality. Such probability update rules can be used recursively to impute causal relationships between observations. That is, a machine can be programmed to "learn".

```{admonition} Summary
:class: tip
Statistical inference and statistical learning are the creative applications of Bayesian probability to problems of rational inference and causal knowledge discovery based on data.
```

## Learning from data: A physicist's perspective

In this course we will focus in particular on the statistical foundations that allow learning from data. More specifically we will take the Bayesian viewpoint of extended logic. Although we aim for theoretical depth, we will take an empirical learning approach with many opportunities to apply the theories in practice using simple computer programs. 

The ambition to teach the theoretical foundation implies that there will be less time to cover the plethora of machine learning methods, or to consider examples from other contexts than physics. We believe that striving for theoretical depth and computational experience will give the best preparation for being able to apply the knowledge in new situations and the broader range of problems that might be encountered in future studies and work. 

This course has been designed specifically for physics students. It is different from an applied mathematics / computer science course. We expect that you have:
* A strong background and experience with mathematical tools (linear algebra, multivariate calculus, mathematical statistics) that will allow to immediately engage in rigorous discussions of multivariate statistics.
* Experience with the use of (physics) models to describe phenomena and an understanding for various uncertainties associated with experimental observations.
* Considerable training in general problem solving skills.

```{admonition} What is special about machine learning in physics?
Physics research takes place within a special context:
  * Physics data and models are connected with physical processes and are often fundamentally different from those encountered in typical computer science contexts. 
  * Physicists ask different types of questions about their data, sometimes requiring new methods.
  * Physicists have different priorities for judging the quality of a model: interpretability, error estimates, predictive power, etc.

Providing slightly more detail:
  * Physicists are data **producers**, not (only) data consumers:
    * Experiments can (sometimes) be designed according to needs.
    * Statistical errors on experimental data can be quantified.
    * Much effort is spent to understand systematic errors.

  * Physics data represents measurements of physical processes:
    * Dimensions and units are important.
    * Measurements often reduce to counting photons, etc, with known a-priori random errors.
    * In some experiments and scientific domains, the data sets are *huge* ("Big Data")

  * Physics models are usually traceable to an underlying physical theory:
    * Models might be constrained by theory and previous observations.
    * There might exist prior knowledge about underlying physics that should be taken into account.
    * Parameter values are often intrinsically interesting.
    * The error estimate of a prediction is just as important as its value.
  ```
