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
AA: 0.8                      AA: 0.8
AB: 0.3  --- trained NN -->  AB: 0.3
BA: 0.5                      BA: 0.5
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
AA: 0.8                      AA: 0.8                      AA: 0.6
AB: 0.3  --- trained NN -->  AB: 0.3  ----- smooth ---->  AB: 0.4  --- retrain NN -->
BA: 0.5                      BA: 0.5                      BA: 0.4
                             BB: 0.1                      BB: 0.2
```

They then train a fitness model $f\_{\theta}$ on these smoothed fitness labels $\hat{Y}$.

[^1]: A graph Laplacian for an undirected graph is a symmetric matrix $L = D - A$, where $D$ is the degree matrix counting each node's edges, and $A$ is the adjacency matrix. This has uses in _spectral clustering_ of graphs.

## 🧬 Clustered Sampling

Now that we have a model for estimating fitness, one can in principle pass this to any discrete sampler.
In the paper, they focus on a particular discrete sampler called "Gibbs with Gradient", whose details we will leave until the next section.

Nevertheless, most discrete samplers can be abstracted into a process that, in every round, gets given a set of sequences and produces another set of mutated sequences.
These mutated sequences then get added so the next round.
This procedure can lead to an intractable number of sequences to consider since each sequence can be mutated in multiple ways.

To limit computational cost, the authors therefore perform hierarchical clustering on all the sequences from a round, and pick the sequence from each cluster that has the highest predicted fitness according to $f\_{\theta}$.
In their notation, they denote this procedure by $\mathrm{Reduce}$, where $\mathcal{C}$ is the number of clusters, so that $\mathrm{Cluster}(X; \mathcal{C}) = \{ X^c \}\_{c=1}^{\mathcal{C}}$ and

$$\mathrm{Reduce}(X; \theta) = \bigcup\_{c=1}^{\mathcal{C}} \{ \underset{x \in X^c}{\mathrm{arg max}} f\_{\theta}(x) \}$$

Each round $r$ culls sequences from the previous round and performs some sampling:

$$\tilde{X}\_r = \mathrm{Reduce}(X\_r; \theta), \qquad X\_{r + 1} = \mathrm{Sample}(\tilde{X}\_r; \theta).$$

A reasonable initial round can be the sequences $X$ used to train the model.
After $R$ rounds, the top $K$ sequences are kept.

## 🧪 Benchmark

They reuse two well-studied proteins for their protein optimisation because of the relative abundance of data: Green Fluorescent Protein (GFP) and Adeno-Associated Virus (AAV).
Both datasets have around ~50k variants each, with variants up to 15 mutations away from the wiltype.
GFP fitness is its fluorescence, whereas AAV's is the ability to package a DNA payload, apparently.

One measure of the difficulty of a protein optimisation benchmark is the "mutational gap", which is the number of mutations away from the starting set required to achieve the highest known fitness.
(In practice, the gap is taken to a set of sequences belonging to 99th fitness percentile, since the true optimum is unknown.)
A second difficulty measure is the fitness range in the starting set of sequences, since a small range of fitness reuiqres the method to learn from barely functional proteins.
They take care to define these notions of "difficulty", because they want to highlight that the deficiencies of previous methods only become apparent at the harder difficulties, and that previous literature used an easier benchmark with more data leakage.
(Their "medium" and "hard" difficulties have gaps of 6 and 7, respectively, with percentile ranges of 20-40 and <30, respectively.)

## 📊 Results

They test the effects of graph-based smoothing against against a variety of baseline protein optimisers, which I very briefly summarise below:

- GFlowNets ([GFN-AL](https://arxiv.org/abs/2203.04115)): an approach somewhat reminiscent of reinforcement learning, where proteins are sampled with probability proportional to the reward function.
- Model-based adaptive sampling (CbAs):
- Greedy search (AdaLead):
- Bayesian optimisation (BO-qei): 
- Conservative model-based optimisation (CoMs): 
- Proximal exploration (PEX): 

The following approach to protein optimisation was not benchmarked because the framework was too tied to antibody optimisation (rather than generic proteins):

- Guided discrete diffusion (NOS):

Their model architecture for $f\_{\theta}$ was a 1D CNN.
Each method generates a set $\hat{X}$ of 128 sequences.

The main success metric is _fitness_, which is the median fitness amongst the final sequences, as judged by the (imperfect!) evaluator $g\_{\theta}$.
(They normalise this based on the lowest and highest known fitness in $Y^\*$.)
In addition, they also quote two other metrics that are not equivalent to better performance, namely _diversity_ and _novelty_.
Diversity is defined as the median pairwise Levenshtein distance in $\hat{X}$, wherease novelty is the median minimal distance to the starting set $X$. 

## ↗️ Gibbs with Gradient (GWG) [Grathwohl et al.](https://arxiv.org/abs/2102.04509)

Since GWG makes direct use of the gradient, there is reason to expect that smoothing the landscape (and hence gradients) will improve a GWG-based optimiser. 
Let us now summarise the key features of this sampler.

The fitness model $f\_{\theta}$ can be viewed as an energy based model defining a Boltzmann distribution $\log{p(x)} = f\_{\theta} - \log{Z}$ with normalisation constant $Z$.
Fit sequences are more likely under this distribution, but there is still diversity amongst functional sequences.

> [!NOTE]
>
> Let's pause to highlight the different objectives of the original GWG paper and how it is used here.
> GWG aims to fairly sample from the distribution $p(x)$ as few costly function evaluations as possible.
> On the other hand, this paper's "directed evolution" procedure of culling unfit sequences after every round targets only the fittest sequences while trying to maintain diversity by applying an arbitrary clustering heuristic.

### 🧠 Aside: Gibbs Sampling

> [!WARNING]
>
> The following goes into the mathematics of GWG.

## 🥡 Takeaways

- Optimising over a smoothed fitness landscape can give better results. 
- There is no principled way to decide on how much smoothing to apply, and it probably varies from dataset to dataset.
- Previous GFP and AAV protein optimsiation benchmarks have been easier than the harder benchmark in this paper.
- Their Gibbs with Gradient sampler benefitted the most from a smoother fitness landscape.
