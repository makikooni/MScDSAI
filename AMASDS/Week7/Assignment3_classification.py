import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assignment 3: Classification

    In this exercise, we will work with two datasets. Perform all tasks below separately for either dataset.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task: Load data

    Load the two datasets and perform all tasks below on both datasets.

    Answer the following questions:
    - How many samples does the dataset have?
    - How many predictors?
    - How many classes?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task: Visualise data

    Plot the data and visualise the different classes (e.g. with different colour or markers)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task: Split the data

    Split the dataset into a training (80 % of sample) and a test set (20 % of samples).

    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task: Model optimisation

    Build both a logistic model and a k-NN classifier optimised on the training set.

    `logistic_model = sklearn.linear_model.SGDClassifier(loss='log_loss')`

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

    `knn_model = NearestNeighbors()`

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


    Perform the following tasks for all combinations of the two models and the two datasets

    For k-NN, choose the metric you want to optimise for (e.g. accuracy, specificity, sensitivity) and tune the model: find the optimal number of neighbours $k$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task: Confusion matrix

    Evaluate the models on the test set.

    Obtain and plot the confusion matrix as a colour figure.

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    Calculate accuracy, specificity and sensitivity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task: ROC curve and AUC

    Plot the ROC curve and calculate the area under the curve (AUC).

    For this, you will need to:
    1. Compute the probability estimates of the positive class ($y=1$):
        `model.predict_proba(X_test)[:, 1]`
    2. Compute the pairs of ROC values (false positive and true positive rates):
        `sklearn.metrics.roc_curve`
    3. Plot the ROC curve given the FPR and TPR
    4. Compute the AUC:
        `sklearn.metrics.roc_auc_score`

    Discuss which model is better on this dataset and why.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus: Other classification models (optional)

    Do the previous tasks for other classification models, for example:
    - Support vector machines (SVM)
    - Linear disciminant analysis (LDA)
    - Quadratic disciminant analysis (QDA)
    - Decision trees
    - Random forest

    Use the implementation in scikit-learn (`sklearn`).

    Compare the methods to logistic regression and k-NN.
    """)
    return


if __name__ == "__main__":
    app.run()
