import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Assignment 3: Classification

    In this exercise, we will work with two datasets. Perform all tasks below separately for either dataset.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.model_selection import train_test_split
    return mo, pd, plt, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task: Load data

    Load the two datasets and perform all tasks below on both datasets.

    Answer the following questions:
    - How many samples does the dataset have?
    - How many predictors?
    - How many classes?
    """
    )
    return


@app.cell
def _(pd):
    #Dataset 1
    df1 = pd.read_csv("data1.csv")
    df1.describe()
    print(f"The first dataset contains {df1.shape[0]} rows")
    print(f"The first dataset contains 2 columns of predictors")
    df1['y'].value_counts()
    print(f"The first dataset contains 2 classes: 0s and 1s")

    return (df1,)


@app.cell
def _(pd):
    #Dataset 2
    df2 = pd.read_csv("data2.csv")
    df2.describe()
    print(f"The second dataset contains {df2.shape[0]} rows")
    print(f"The second dataset contains 2 columns of predictors")
    df2['y'].value_counts()
    print(f"The second dataset contains 2 classes: 0s and 1s")


    return (df2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task: Visualise data

    Plot the data and visualise the different classes (e.g. with different colour or markers)
    """
    )
    return


@app.cell
def _(df1, plt):
    #Dataset1
    def vis_1():
        X1 = df1.iloc[:, 0]    # first predictor, all rows, first column
        X2 = df1.iloc[:, 1]    # second predictor, all rows second column
        y  = df1['y']          # label column
    
        # Plot: class 0
        plt.scatter(X1[y == 0], X2[y == 0], #x-axis,y-axis
                    color='red', label='Class 0', alpha=0.7)
    
        # Plot: class 1
        plt.scatter(X1[y == 1], X2[y == 1], 
                    color='violet', label='Class 1', alpha=0.7)
    
        plt.xlabel('Predictor 1 (x-axis)')
        plt.ylabel('Predictor 2 (y-axis)')
        plt.title('Dataset 1: Visualisation of Classes')
        plt.legend()
        plt.grid(True)
        plt.show()

    vis_1()
    return


@app.cell
def _(df2, plt):
    #Dataset2
    def vis_2():
        X1 = df2.iloc[:, 0]    # first predictor, all rows, first column
        X2 = df2.iloc[:, 1]    # second predictor, all rows second column
        y  = df2['y']          # label column
    
        # Plot: class 0
        plt.scatter(X1[y == 0], X2[y == 0], #x-axis,y-axis
                    color='blue', label='Class 0', alpha=0.7)
    
        # Plot: class 1
        plt.scatter(X1[y == 1], X2[y == 1], 
                    color='green', label='Class 1', alpha=0.7)
    
        plt.xlabel('Predictor 1 (x-axis)')
        plt.ylabel('Predictor 2 (y-axis)')
        plt.title('Dataset 2: Visualisation of Classes')
        plt.legend()
        plt.grid(True)
        plt.show()

    vis_2()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task: Split the data

    Split the dataset into a training (80 % of sample) and a test set (20 % of samples).

    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    )
    return


@app.cell
def _(df1, train_test_split):
    #Dataset 1
    def split_1():
        X = df1.iloc[:, :-1]   # predictors
        y = df1.iloc[:, -1]    # target
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,      
            random_state=42,    # to keep same for all script runs
            stratify=y          # preserve original ratio of targets
        )
    
        print("Dataset 1: training set size:", X_train.shape[0], "samples")
        print("Dataset 1: test set size:", X_test.shape[0], "samples")
    split_1()
    return


@app.cell
def _(df2, train_test_split):
    #Dataset 2
    def split_2():
        X = df2.iloc[:, :-1]   # predictors
        y = df2.iloc[:, -1]    # target
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,      
            random_state=42,    # to keep same for all script runs
            stratify=y          # preserve original ratio of targets
        )
    
        print("Dataset 2: training set size:", X_train.shape[0], "samples")
        print("Dataset 2: test set size:", X_test.shape[0], "samples")
    split_2()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task: Model optimisation

    Build both a logistic model and a k-NN classifier optimised on the training set.

    `logistic_model = sklearn.linear_model.SGDClassifier(loss='log_loss')`

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

    `knn_model = NearestNeighbors()`

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


    Perform the following tasks for all combinations of the two models and the two datasets

    For k-NN, choose the metric you want to optimise for (e.g. accuracy, specificity, sensitivity) and tune the model: find the optimal number of neighbours $k$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task: Confusion matrix

    Evaluate the models on the test set.

    Obtain and plot the confusion matrix as a colour figure.

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    Calculate accuracy, specificity and sensitivity.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Bonus: Other classification models (optional)

    Do the previous tasks for other classification models, for example:
    - Support vector machines (SVM)
    - Linear disciminant analysis (LDA)
    - Quadratic disciminant analysis (QDA)
    - Decision trees
    - Random forest

    Use the implementation in scikit-learn (`sklearn`).

    Compare the methods to logistic regression and k-NN.
    """
    )
    return


if __name__ == "__main__":
    app.run()
