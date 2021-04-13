# BOLASSO Example

This worked example illustrates a machine-learning approach to predictive modeling that combines bootstrapping and LASSO ("BOLASSO") to create a linear ensemble model with strong predictive performance. The steps are:

1. Perform a BOLASSO (Bach 2008): sample *N* bootstrap replicates from the data and fit a cross-validated LASSO model to each bootstrap replicate.
2. Select the variables that have non-zero coefficients in *at least 90% of the BOLASSO models*.
3. Fit a *simple linear regression* to the selected variables, which is called the post-BOLASSO model and describes the strongest consistent predictors. 
4. Create an ensemble model by *averaging the coefficients across the BOLASSO models*. This is analagous to a random forest in its use of bootstrap aggregation (e.g. "bagging"), except that the individual models that make up the forest are regularized linear models instead of decision trees.

## How to Run

The analysis is automated using the [SCons](https://scons.org/) software construction tool and implemented in Python and R using the package lists below. Once installed, the analysis can be run with the `scons` command in the root of the repo. Analysis output is written to the `scratch` subdirectory.

### R Dependencies

- tidyverse
- data.table
- gamlr
- AUC
- assertthat

### Python Dependencies

- pandas
- scons
- codebooks

## Data

This example uses the [Adult](https://archive.ics.uci.edu/ml/datasets/Adult) (or "Census Income") data set from the UC Irvine Machine Learning Repository. Briefly, these are weighted data from the 1994 Census with 14 explanatory variables and a binary outcome variable for whether an individual earns more or less than $50,000 annually. The data have already been split into 32,561 training records (in `data/uci-adult-train.csv`) and 16,281 test records (in `data/uci-adult-test.csv`). The pipeline generates a codebook for the training data in `scratch/train_codebook.html`.

## Feature Engineering

We identify each variable in the data as one of the following types (see `feature-engineering.py`):

- **Categorical**. A variable that takes on a discrete number of values (as observed in the training data) that are expanded into dummy variables with missing values treated as a separate category. The dummy variable for the most frequently occurring category is dropped to prevent collinearity.
- **Continuous**. A variable that takes on a continuous distribution of values, which we standardize by subtracting the mean and dividing by the standard deviation from the training data. The same transformation is applied to the test data. Missing values are imputed with zero (which is the mean value in the standardized form) and a dummy variable is added as a missing indicator. Higher-order (squared, cubed) can also be added to allow for flexible functional forms.
- **Hurdle**. A variable that comes from a combined discrete/continuous distribution (usually zero-inflated) and either takes on a discrete value (zero) or a continuous value (greater than zero). We expand the variable into its extensive margin (a dummy variable indicating non-zero) and intensive margin (the continuous value conditional on the value being non-zero). Because the intensive margin excludes zero, it can be log-transformed if the continuous distribution is exponential.

The pipeline generates a codebook for all of the features in `scratch/features_codebook.html`.

The features are then transformed into a model matrix (see `model-matrix.R`) by:

- Dropping any features that are constant (take on only a single value) in the training data.
- Calculating all pairwise correlations among features and dropping highly correlated features (>0.99) to avoid numerical instability in model fitting.
- Packaging the training/testing matrices separately as sparse matrix objects in an RData file.

## Models

We fit LASSO models to the bootstrap replicates in R using the `gamlr` package (Taddy 2017) with its default cross-validation settings (see `bolass-replicate.R`). We turn standardization off, as the model matrix has already been standardized by our feature engineering pipeline.

We fit the post-LASSO model to the selected variables in R with the `glm` function (see `post-lasso.R`).

## Results

The BOLASSO selects 43 of the features as consistent predictors of whether an individual earns more or less than $50,000 annually. The selected features include interactions and higher-order terms, illustrating the additional predictive power that flexible functional forms can provide.

BOLASSO helps to identify consistent predictors, avoiding arbitrary choices among highly correlated pairs. While the post-LASSO coefficients on the selected variables do not necessarily have a causal interpretation, they can be interpreted as the factors that are the strongest predictors among observables, and can point to potential underlying mechanisms for further study. The post-LASSO coefficients can be found in `scratch/post-lasso-coefficients.csv`.

The ensemble model has 177 non-zero coefficients, found in `scratch/bolasso-ensemble-coefficients.csv`. The ensemble model is simply another linear model with the averaged coefficients (including zeros) from the BOLASSO models.

Both the post-LASSO and ensemble models allow us to fit a flexible functional form to the data in a similar way to a random forest or neural network, but with models that are easier to interpet in the sense that they are linear and provide coefficients that summarize the contribution of each variable, interaction and higher-order term.

A common metric for assessing the performance of a machine-learning model is the area under the receiver-operating characteristic curve (AUC). In this example, the difference in predictive performance is negligible between the post-LASSO model that only uses the BOLASSO selected variables (AUC 0.909) and the full ensemble model (AUC 0.907). In other cases, however, the additional predictors from the ensemble can outperform the post-LASSO (see the opioid prediction model in Hastings, Howison, and Inman 2020).

## References

Bach FR. 2008. BOLASSO: Model consistent LASSO estimation through the bootstrap.
In *Proceedings of the 25th International Conference on Machine Learning*,
Association for Computing Machinery, New York, NY, 2008, pp. 33–40.
doi:[10.1145/1390156.1390161](https://doi.org/10.1145/1390156.1390161)

Taddy M. 2017. One-Step Estimator Paths for Concave Regularization.
*Journal of Computational and Graphical Statistics* **26**(3): 525-536.
doi:[10.1080/10618600.2016.1211532](https://doi.org/10.1080/10618600.2016.1211532)

Hastings JS, Howison M, Inman SE. 2020. Predicting high-risk opioid prescriptions before they are given.
*Proceedings of the National Academy of Sciences* **117**(4): 1917-1923.
doi:[10.1073/pnas.1905355117](10.1073/pnas.1905355117)

## License

Copyright 2021, Innovative Policy Lab (d/b/a Research Improving People's Lives), Providence, RI. All Rights Reserved.

See [LICENSE](LICENSE) for more details.
