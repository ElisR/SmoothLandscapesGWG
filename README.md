# Improving Protein Optimisation with Smoothed Fitness Landscapes


## 💬 Talk Notes

> [!NOTE]
>
> Unfortunately, GitHub's LaTeX parser is slightly limited, and will aggressively interpret subscript indicators as attempts to italicise text, so I will be using superscript more than I would like.

## 🏖️  Destination: Optimising over Fitness Landscapes

The number of possible proteins that have been explored by nature is a small fraction of the total space.
This total space is also so large that brute-force search is infeasible.

In some cases, we have a natural protein that we would like to optimise, such that it performs better at some function.
This performance is generally dubbed "fitness".
"Optimising" sequences corresponds to finding maxima in a protein fitness landscape that we only have partial access to, namely some noisy fitness measurements over a small domain.
The optimal protein may be multiple mutations away from this domain.

The first point that this paper makes is that proteins fitness landscapes are far from smooth.
Data is also scarce because fitness datasets are expensive to produce.
This makes protein optimisation using machine learning methods susceptible to getting stuck in local optima.

## 🟰 Problem Formulation

On more formal footing, one has a starting set of $N$ proteins $\mathcal{D} = (X, Y)$ where $X = \{ x^1, \ldots, x^N \} \subset \mathcal{V}^M$ are sequences and $Y = \{ y^1, \ldots, y^N \}$ are their scalar fitness labels.
$\mathcal{V}$ is the vocabulary of amino acids, which is 20.
The aim is to generate sequences with higher fitness than the starting set.

Evaluating a protein optimiser is difficult, because if an optimiser proposes a sequence, the only way to access the true fitness is to perform a biological experiment.
Hence, prior work in this field perform in-silico evalution with an approximation $g^\phi$ to the true fitness black-box function $g: \mathcal{V}^M \to \mathbb{R}$ satisfying $g(x^*) = y^*$.
That is, they consider the starting dataset $\mathcal{D}$ to be a strict subset $\mathcal{D} \subset \mathcal{D}^*$ of a larger known dataset $\mathcal{D}^* = (X^*, Y^*)$.

## 💆 Graph-Based Smoothing

The philosophy of this paper is to optimise over a _smoothed_ fitness landscape, even if the true fitness landscape is indeed non-smooth.
They do this with a graph-based method borrowed from graph signal processing.



## 📉 Training Procedure


## 🧬 Sampling Procedure



## ↗️ Gibbs with Gradient (GWG)

Since GWG makes direct use of the gradient, there is reason to expect that smoothing the landscape (and hence gradients) will improve a GWG-based optimiser. 

> [!WARNING]
>
> The following goes into the mathematics of GWG.

> [!NOTE]
> 
> Elis: Should make a comment about GWG trying to sample from a distribution, whereas the optimiser is trying to find diverse optima. 

## 🧪 Experiments

They test graph-based smoothing with other protein optimisers, in addition to their GWG-based optimiser.


## 📊 Results

## 🥡 Takeaways


- Optimising over a smoothed fitness landscape gives better results. 
