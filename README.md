# Improving Protein Optimisation with Smoothed Fitness Landscapes 

## 💬 Talk Notes

> These are brief notes for a talk about [Kirjner et al. 2023](https://arxiv.org/abs/2307.00494)'s paper.

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
In the paper, they focus on a particular discrete sampler called "Gibbs-With-Gradients", whose details we will leave until the next section.

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

The authors take care to define the "difficulty" of a protein optimisation task, because they want to highlight that the deficiencies of previous methods only become apparent at the harder difficulties, and that previous literature used an easier benchmark with more data leakage.
One measure of the difficulty of a protein optimisation benchmark is the "mutational gap", which is the number of mutations away from the starting set required to achieve the highest known fitness.
(In practice, the gap is taken to a set of sequences belonging to 99th fitness percentile, since the true optimum is unknown.)
A second difficulty measure is the fitness range in the starting set of sequences, since a small range of fitness requires the method to learn from barely functional proteins.
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
All their results are averaged across five random seeds (with standard deviations also quoted).
For their GWG sampler, they ran their optimiser for $R=15$ rounds.

The main success metric is _fitness_, which is the median fitness amongst the final sequences, as judged by the (imperfect!) evaluator $g\_{\theta}$.
(They normalise this based on the lowest and highest known fitness in $Y^\*$.)
In addition, they also quote two other metrics that are not equivalent to better performance, namely _diversity_ and _novelty_.
Diversity is defined as the median pairwise Levenshtein distance in $\hat{X}$, wherease novelty is the median minimal distance to the starting set $X$.
Together, these shed some light on how the optimisers trade exploration for exploitation.

<!-- Show the results table. -->

The results tables are quite dense with information, so I will only point out two conclusions relating to this paper's optimiser.
Firstly, without the graph-based smoothing step, the GWG-based optimiser is among the worst performers.
Secondly, with graph-based smoothing, the GWG-based optimiser is the best performer.

Their appendices test the effects of varying some hyperparameters, namely:

- the smoothing factor $\gamma$, which shouldn't get too large;
- the number of nodes in the smoothed graph they use to train the fitness model $f\_{\theta}$, where larger graphs seem to help at the expense of more compute;
- the number of rounds $R$, where more rounds just aid in reaching convergence.

## ↗️ Gibbs-With-Gradients (GWG) [Grathwohl et al. 2021](https://arxiv.org/abs/2102.04509)

Since GWG makes direct use of the gradient, there is reason to expect that smoothing the landscape (and hence gradients) will improve a GWG-based optimiser. 
Let us now summarise the key features of this sampler.

> [!NOTE]
>
> Let's pause to highlight the different objectives of the original GWG sampler and this _Kirjner et al._ paper.
> GWG aims to fairly sample from the distribution $p(x)$ with as few costly function evaluations as possible.
> On the other hand, this paper's "directed evolution" procedure of culling unfit sequences after every round targets only the fittest sequences.
> Mathematically, _Kirjner et al._'s clustering heuristic is arbitrary and is only there to maintain some diversity in practice.


### 🧠 Aside: Gibbs Sampling

The fitness model $f\_{\theta}$ can be viewed as an energy based model defining a Boltzmann distribution $\log{p(x)} = f\_{\theta}(x) - \log{Z}$ with normalisation constant $Z$.
Fit sequences are more likely under this distribution.

> [!WARNING]
>
> This section is optional for understanding the features of this paper.

The problem with trying to sample from $p(x)$ is that we don't have the normalisation constant $Z$, only the relative fitness of sequences: $f\_{\theta}(x') - f\_{\theta}(x)$.
Metropolis-Hastings sampling (an example of Markov-chain Monte Carlo) is one way of overcoming this hurdle.
With this kind of sampling, one starts with one sequence, then iteratively forms a chain of sequences through gradual mutation. 
After enough time passes, the distribution of sequences in the chain follow the distribution $p(x)$, provided one uses a valid rule for when to mutate a sequence in the chain and when to leave it alone.

Recall that the data we are interested in is often high-dimensional, and "Gibbs" part of the sampling means that we're only updating a few dimensions of $x$ at a time, often simply just one dimension e.g. making one point mutation at a time.
The $i$-th position of the next sequence in the chain is chosen according to $p(x\_i | x\_{-i})$, where $-i$ refers to the set of all other dimensions.
If there are $K$ possible categories, we can calculate this normalised probability through $K$ evaluations of $f\_{\theta}$.
One can then iterate through the dimensions in some fixed ordering to ensure all dimensions eventually get changed.
Notice that a particular dimension does not have to change after an iteration.
Indeed, in certain problems certain dimensions will very rarely change: for example, consider the outer pixels in MNIST, or conserved regions in the SARS-CoV-2 spike protein.
**Proposing a dimension that does not subsequently change is wasted computation.**
This motivates being judicious in choosing how often to propose changing certain dimensions.

The classic Metropolis-Hastings sampler follows this principle of first proposing a sequence $x'$ according to $q(x'|x)$, then using the (potentially expensive) energy $f\_{\theta}$ function to accept the proposal with probability[^2]

[^2]: Why this strange acceptance probability? We're trying to make sure that the sampler satisfies _detailed balance_, meaning every transition is reversible and hence that there exists an equilibrium distribution if run for enough time.
For this, we need $p(x' | x) p(x) = p(x | x') p(x')$, where above we have split up the transition probability into propoposal and acceptance: $p(x' | x) = q(x' | x) A(x', x)$.
This requirement sets $A(x', x)$ as written.

$$A(x', x) = \mathrm{min}(e^{f\_{\theta}(x') - f\_{\theta}(x)} \frac{q(x|x')}{q(x'|x)}, 1).$$

When writing the proposal distribution as $q(x'|x) = \sum\_i q(x' | x, i) q(i)$ where $q(i)$ is a distribution over indices $i \in \{ 1, \ldots, D \}$, the Metropolis-Hastings approach can lead to performance improvements when $q(i)$ is biased towards dimensions that are more likely to change.
Going one step further, if we swapped the unconditional proposal $q(i)$ for an input dependent proposal $q(i | x)$, we could theoretically do even better.
For example, in MNIST, the pixels most likely to change are at the edge of a digit, while in a protein, some residues are likely to co-mutate when they are close in 3D space. 

### 🔙 Back to GWG

Now that we know have a flavour for why preferentially proposing dimensions most likely to change (i.e. positions most likely to mutate) makes sampling more efficient, let's see how GWG uses gradients to inform its proposals.

An nice visualisation of the process is shown in Figure 1 of _Grathwohl et al._.

<!-- Show Figure 1 -->

The GWG paper contains a theorem stating that how close this sampler is to being optimally efficient is linked to how smooth the energy function is.
This helps explain why spiky (and possibly inaccurate) gradients in the protein landscape without graph-based smoothing led to bad results.

Literature prior to GWG showed that the following proposal is an optimal locally-informed proposal:

$$q(x' | x) \propto e^{(f\_{\theta}(x') - f\_{\theta}(x))/2} \mathbf{1}(x' \in H(x)),$$

where $H(x)$ is the Hamming ball around $x$.
Unfortunately, even with a Hamming window of size $1$, this would still require $\mathcal{O}(D K)$ evaluations of $f\_{\theta}$ for $D$ dimensions and $K$ categories per iteration in order to perform the sotfmax.
GWG manages to cut this down to $\mathcal{O}(1)$ evaluations, while incurring minimal decrease in the sampling efficiency.

So, how can we perform fewer evaluation of $d\_{\theta}(x) \equiv f\_{\theta}(x') - f\_{\theta}(x))$?
The key insight is that $f\_{\theta}$ is often a continuous, differentiable function, even if it is only meant to be evaluated on certain discrete (e.g. one-hot encoded) inputs.
That means we can estimate many $f\_{\theta}(x') - f\_{\theta}(x)$ by evaluating a single gradient $\nabla\_x f\_{\theta}(x)$ and picking out the relevant components.

Letting $\tilde{d}\_{\theta}(x)\_{ij} \approx d\_{\theta}(x)\_{ij}$ approximate the log-likelihood ratio of changing the $i$-th dimension of $x$ from its current value to the value $j$, we have

$$\tilde{d}(x)\_{ij} = [\nabla\_x f\_{\theta}(x)]\_{ij} - \sum\_k x\_{ik} [\nabla\_x f\_{\theta}(x)]\_{ik}.$$

In practice, we can get this gradient with automatic differentiation once, and do $\mathcal{O}(D K)$ cheap operations to get the whole matrix $\tilde{d}(x)$ to construct a valid categorical distribution over proposals.

> [!WARNING]
>
> I'm pretty sure that Eq. 2 of _Kirjner et al._'s paper is technically wrong for the dimensions that do change, whereas GWG's Eq. 4 is correct.
> I choose a different notation altogether to try and make things clearer.

## 🥡 Takeaways

- Optimising over a **smoothed** fitness landscape can give better results. 
- There is no principled way to decide on how much smoothing to apply, and it probably varies from dataset to dataset.
- **Previous** GFP and AAV protein optimsiation **benchmarks** have been **easier** than the harder benchmark in this paper.
- Their **Gibbs-With-Gradients** sampler **benefitted** the most from a **smoother** fitness landscape.
