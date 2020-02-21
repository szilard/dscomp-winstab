
## Winner stability in data science competitions

Many data science competitions (e.g. Kaggles) are won with small, 4th digit (0.0001s) margins
in the respective metric over the 2nd place. If the dataset used for evaluation ("private 
leaderboard") is not large enough, the top ranks might not reflect true underlying value
(the ranks will not be statistically significant, i.e. a new/another evaluation set with the 
same characteristics could produce another "winner"). 

In this repo we'll study the rank overlapping problem of the top models in a data science competition 
*simulation* as a function of the evaluation/test set size and the number of competitors. It is
to be expected that for a large number of competitors (lots of models within a small range of the
accuracy metric) and not large enough evaluation datasets, the "winner" will be somewhat random (among
the top models). 

The simulation captures an idealized setting. A training set (along with a separate "validation" set
used for early stopping) will be randomly drawn from a larger dataset  
A large "population" evaluation set will be drawn from the same data, and then we'll draw repeatedly
smaller subsamples from the latter to be used as "evaluation" set/"private leaderboard". 
We'll train several models on the training set and we'll compare their "true" performance and rank
(as measured on the larger "population" evaluation set) with the "competition" performance and rank
as measured on the smaller "private leaderboard" in each resample. For a competition to be meaningful, the rank
of the top models in each "private leaderboard" resample should coincide with their rank on the
larger "population" evaluation set. 

Kaggle rankings in a competition are determined based on a single finite "private leaderboard" test set.
In the real world a larger "population" set is not available, however competition organizers could still evaluate
the stability/statistical significance of the top ranks using bootstrapping (from the "private 
leaderboard" test set). In fact, it was later found by this author that this bootstrapping procedure
has been already used long time ago in the KDD Cup 2004, and in one of its sub-competitions a 3-way tie
has been declared due to the statistical overlapping of the top models. In fact, the bootstrapping 
procedure could even be used as a "fair" way to distribute the prize money between the top competitors.

We have to note that this simulation only captures the effect of finite evaluation sample in presence of 
many competitors. In real-world projects, distributions (slowly) change in time and for example the
training and test sets have slightly different distribution. Also once models are deployed in production,
the data distributions change even further and it is not necessarily the best model on the evaluation test set
that is going to perform the best on the new online data. (For example it is a conjecture of this author that less
complex models will be more robust to non-stationarity and will perform better in practice than highly
tuned models that "win" a "competition" on a fixed test set).


### Simulation setup

From a dataset of 10 million records, we get a training sample of 100K and a validation set of 20K records. 
We train `K` (e.g. `K=1000`) GBM models (binary classification) with lightgbm by using random search over a grid of hyperparameter values. 
We measure the AUC of the models on a larger "population" evaluation set and `B` samples of size `M` (e.g. `M=100K`) 
simulating repeated "competitions" on finite "private leaderboard" test sets of size `M`. 
We rank the models (based on AUC) on the large "population" evaluation set ("true rank") and
also on each of the `B` "competitions" (on the "private leaderboards" test sets).



