# BOLASSO Example

This example analysis illustrates a machine-learning approach to predictive
modeling that uses regularized linear models as the 

1. Perform a BOLASSO: sample *N* bootstrap replicates from the data and fit a cross-validated LASSO model to each bootstrap replicate.
2. Select the variables that have non-zero coefficients in at least 90% of the BOLASSO models.
3. Fit a *simple linear regression* to the selected variables, which is called the post-BOLASSO model and can be interpreted as 
4. Create an ensemble model by averaging the coefficients across the BOLASSO models. This is analagous to a random forest in its use of bootstrap aggregation (e.g. "bagging"), except that the individual models that make up the forest are regularized linear models instead of decision trees.

## Data

This example uses the [Adult](https://archive.ics.uci.edu/ml/datasets/Adult) (or "Census Income")
data set from the UC Irvine Machine Learning Repository. Briefly, these are weighted data from the 1994 Census
with 14 predictors and a binary outcome variable for whether an individual earns more or less than $50,000
annually. The data are split into 32,561 training records (in `data/uci-adult-train.csv`) and 16,281 test
records (in `data/uci-adult-test.csv`).

## Feature Engineering

## Results

The difference in predictive performance is large between the models: the post-LASSO achieves an AUC of 0.733 while the ensemble achieves an AUC of 0.892.

## References

Bach FR. 2008. BOLASSO: Model consistent LASSO estimation through the bootstrap.
In *Proceedings of the 25th International Conference on Machine Learning*,
Association for Computing Machinery, New York, NY, 2008, pp. 33–40.
doi:[10.1145/1390156.1390161](https://doi.org/10.1145/1390156.1390161)

Taddy M. 2017. One-Step Estimator Paths for Concave Regularization.
*Journal of Computational and Graphical Statistics* **26**(3): 525-536.
doi:[10.1080/10618600.2016.1211532](https://doi.org/10.1080/10618600.2016.1211532)

## License

Copyright 2021, Innovative Policy Lab (d/b/a Research Improving People's Lives),
Providence, RI. All Rights Reserved.

See [LICENSE](LICENSE) for more details.
