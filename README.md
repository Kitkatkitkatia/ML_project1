# EPFL Machine Learning Project 1

This repository contains implementations of the six baseline methods for the
EPFL Machine Learning Project 1, plus experiments with logistic regression for
binary classification. The implementation uses NumPy for the ML algorithms;
the notebooks also use pandas for CSV handling and Matplotlib/Seaborn for
plots. No external machine-learning library is used to train the models.

## Repository contents

- [`Code/implementations.py`](Code/implementations.py): baseline algorithms:
  mean-squared-error gradient descent, stochastic gradient descent, least
  squares, ridge regression, logistic regression, and regularized logistic
  regression.
- [`Code/costs.py`](Code/costs.py): MSE, MAE, and cross-entropy loss functions.
- [`Code/helpers.py`](Code/helpers.py): minibatch iteration and k-fold index
  generation.
- [`Code/evaluation.py`](Code/evaluation.py): accuracy and F1-score helpers.
- [`Code/Logistic_SGD.py`](Code/Logistic_SGD.py): logistic SGD and
  cross-validation helpers.
- [`Code/Logistic_SGD_balanced.py`](Code/Logistic_SGD_balanced.py): balanced
  minibatch logistic SGD and threshold evaluation.
- [`Logistic_SGD.ipynb`](Logistic_SGD.ipynb): an initial logistic-regression
  experiment.
- [`Logistic_SGD_improved.ipynb`](Logistic_SGD_improved.ipynb): the improved
  preprocessing, feature-selection, and cross-validation experiment.
- [`data/dataset/`](data/dataset/): training and test CSV files.
- [`submission_log_reg.csv`](submission_log_reg.csv): a saved prediction file
  in the competition submission format.

Other notebooks and image files contain exploratory analysis and plots.


## Customized model and experiment

The improved notebook explores logistic regression trained with stochastic
gradient descent. Its preprocessing workflow is:

1. Recode selected health-survey response values: `88` is treated as zero for
   the health-day questions, while `77` and `99` are treated as missing.
   Selected categorical fields are converted to binary values; their `7` and
   `9` responses are treated as missing.
2. Remove features with at least 80% missing training values.
3. Fill missing values using per-feature medians computed on the training set.
4. Standardize using training-set means and standard deviations (using a
   standard deviation of 1 for constant features), then clip values to
   `[-5, 5]`.
5. Rank features by the absolute correlation with the binary target and retain
   the top 100.

### Model architecture

The customized classifier is a linear logistic-regression model, not a
multi-layer neural network. For each example, it takes the 100 selected
preprocessed features $x$, computes a linear score $z = w^\top x$, and
applies the sigmoid function $\sigma(z) = 1/(1+e^{-z})$ to produce the
probability of the positive class. Training minimizes binary cross-entropy
with stochastic gradient descent. There are no hidden layers or nonlinear
feature transformations; the sigmoid is the only nonlinear operation.

The balanced-SGD implementation forms each minibatch by sampling half of its
examples from each class, helping the optimizer learn from the minority
class. L2 regularization is applied during weight updates, with the first
coefficient exempted by the implementation. In the current notebook workflow,
the selected 100-feature matrix is passed directly to the trainer without an
explicit constant (intercept) column, so that exemption applies to the first
selected feature rather than to a separate bias parameter.

For model selection, the notebook uses 10-fold cross-validation with a grid of
15 learning rates from `10^-3` to `1`, three regularization strengths
(`0`, `0.00005`, and `0.0001`), 1,500 SGD iterations, and minibatches of 512.
The custom balanced-SGD helper samples equal numbers from each class per
minibatch. On each validation fold, it evaluates decision thresholds from
`0.05` to `0.95`; the notebook selects the learning-rate/regularization pair
by mean validation F1 score and averages the fold-selected thresholds.

To explore the analysis, start Jupyter from the repository root and open
[`Logistic_SGD_improved.ipynb`](Logistic_SGD_improved.ipynb):

```bash
jupyter notebook Logistic_SGD_improved.ipynb
```

The notebook cells contain the data preparation and cross-validation
configuration. The initial workflow is in
[`Logistic_SGD.ipynb`](Logistic_SGD.ipynb).
