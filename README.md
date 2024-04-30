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

On more formal footing, one has a starting set of $N$ proteins $\mathcal{D} = (X, Y)$ where $X = \{ x\_1, \ldots, x\_N \} \subset \mathcal{V}^M$ are sequences and $Y = \{ y\_1, \ldots, y\_N \}$ are their scalar fitness labels.
$\mathcal{V}$ is the vocabulary of amino acids, which is 20.
The aim is to generate sequences with higher fitness than the starting set.

Evaluating a protein optimiser is difficult, because if an optimiser proposes a sequence, the only way to access its true fitness is to perform a biological experiment.
Hence, prior works use an in-silico approximation for evaluation.

Suppose that all known sequence and fitness measurements are contained in $\mathcal{D}^{\*} = (X^{\*}, Y^{\*})$.
One assumes the existence of black-box fitness function $g: \mathcal{V}^M \to \mathbb{R}$ satisfying $g(x^{\*}) = y^{{\*}}$.

Hence, one trains an evaluator model $g\_{\phi}$ to minimise the error on $\mathcal{D}^{\*}$, which one can use to score any sequence that an optimiser gives us (even outside the domain of $\mathcal{D}^{\*}$).
The starting dataset $\mathcal{D}$ we mentioned above is a strict subset $\mathcal{D} \subset \mathcal{D}^{\*}$, so that the evaluator model has access to more data than whatever model or optimiser is under investigation.
This situation mimics the setting of real life protein optimisation, just with the real oracle function $g$ replaced by the best in-silico approximation we can make: a machine learning model $g^{\*}$ trained on more data.

## 💆📉 Graph-Based Smoothing and Training

The philosophy of this paper is to optimise over a _smoothed_ fitness landscape, even if the true fitness landscape is indeed non-smooth.
They do this with a graph-based method borrowed from graph signal processing.

Firstly, they train a noisy fitness model $f\_{\tilde{\theta}}$ on the initial dataset $\mathcal{D}$.
They then augment the dataset by using $f\_{\tilde{\theta}}$ to infer the fitness of neighbouring sequences, which are sequences from $X$ with random point mutations.

```
AA: 0.8                          AA: 0.8
AB: 0.3  ----- trained NN ---->  AB: 0.3
BA: 0.5                          BA: 0.5
                                 BB: 0.1
```

The augmented sequences then become nodes $V$ on a graph, where edges $E$ are constructed as a $k$-nearest neighbour graph based on Levenshtein distance.
They then define the smoothness of the fitness graph as the sum of squares of local variability:

$$\mathbf{TV}\_{2}(Y) = \frac{1}{2} \sum\_{i \in V} \sum\_{(i, j) \in E} (y\_i - y\_j)^2$$

This is used as a regulariser for defining smoothed fitness labels $\hat{Y}$:

$$\underset{\hat{Y} \in \mathbb{R}^{|V|}}{\mathrm{arg min}} \| Y - \hat{Y} \|^2\_2 + \gamma \mathbf{TV}\_{2}(\hat{Y}).$$

(This is a quadratic convex problem which has a closed form solution $\hat{Y} = (1 + \gamma L)^{-1} Y$ in terms of the graph Laplacian.[^1])

$\gamma$ is a hyperparameter that sets the amount of smoothing: setting it too high can lead to underfitting, whereas setting it too low negates the purpose of smoothing in the first place.
A disadvantage of this approach is that there is no principled way of choosing $\gamma$, and the authors proceeded with a trial-and-error approach for this hyperparameter. 

```
AA: 0.8                          AA: 0.8                          AA: 0.6
AB: 0.3  ----- trained NN ---->  AB: 0.3  ------- smooth ------>  AB: 0.4  ----- retrain NN ---->  
BA: 0.5                          BA: 0.5                          BA: 0.4
                                 BB: 0.1                          BB: 0.2
```

[^1]: A graph Laplacian for an undirected graph is a symmetric matrix $L = D - A$, where $D$ is the degree matrix counting each node's edges, and $A$ is the adjacency matrix. This has uses in _spectral clustering_ of graphs.

## 🧬 Clustered Sampling




## ↗️ Gibbs with Gradient (GWG)

Since GWG makes direct use of the gradient, there is reason to expect that smoothing the landscape (and hence gradients) will improve a GWG-based optimiser. 

### 🧠 Aside: Gibbs Sampling

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
