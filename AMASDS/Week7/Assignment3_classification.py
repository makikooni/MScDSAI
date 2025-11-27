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
    from matplotlib import pyplot as plt
    import pandas as pd
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import (
        accuracy_score, confusion_matrix,
        roc_curve, roc_auc_score
    )
    from sklearn.neighbors import KNeighborsClassifier

    return (
        KNeighborsClassifier,
        SGDClassifier,
        StandardScaler,
        accuracy_score,
        confusion_matrix,
        make_pipeline,
        pd,
        plt,
        roc_auc_score,
        roc_curve,
        sns,
        train_test_split,
    )


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
        plt.figure(figsize=(10, 5))
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
        plt.figure(figsize=(10, 5))
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
    # Dataset 1
    def split_1():
        X = df1.iloc[:, :-1]   # predictors
        y = df1.iloc[:, -1]    # target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,      
            random_state=42, # to keep same for all script runs
            stratify=y # preserve original ratio of targets
        )
    
        print("Dataset 1: training set size:", X_train.shape[0], "samples")
        print("Dataset 1: test set size:", X_test.shape[0], "samples")
    
        return X_train, X_test, y_train, y_test


    X1_train, X1_test, y1_train, y1_test = split_1()

    return X1_test, X1_train, y1_test, y1_train


@app.cell
def _(df2, train_test_split):
    def split_2():
        X = df2.iloc[:, :-1]
        y = df2.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    
        print("Dataset 2: training set size:", X_train.shape[0], "samples")
        print("Dataset 2: test set size:", X_test.shape[0], "samples")
    
        return X_train, X_test, y_train, y_test

    X2_train, X2_test, y2_train, y2_test = split_2()

    return X2_test, X2_train, y2_test, y2_train


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


@app.cell
def _(
    KNeighborsClassifier,
    SGDClassifier,
    StandardScaler,
    make_pipeline,
    train_test_split,
):
    def train_logistic(X_train, y_train):
        # pipeline: standardisation + logistic regression
        # We standardise the predictors because SGD-based logistic regression and k-NN 
        # are sensitive to feature scale. StandardScaler ensures each feature has mean 0 
        # and unit variance, which stabilises optimisation and gives meaningful distances.
        model = make_pipeline(
            StandardScaler(),
            SGDClassifier(loss='log_loss', random_state=42)
        )
        model.fit(X_train, y_train)
        return model


    def tune_knn(X_train, y_train):
        # Validation split used to tune k while keeping the original test set untouched.
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    
        ks = [1, 3, 5, 7, 9]
        best_acc = 0.0
        best_ks = [] 
    
        for k in ks:
            model = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=k)
            )
            model.fit(X_tr, y_tr)
            acc = model.score(X_val, y_val)
        
            print(f"k={k}, validation accuracy={acc:.3f}")
        
            if acc > best_acc:
                best_acc = acc
                best_ks = [k]      
            elif acc == best_acc:
                best_ks.append(k)  # add another equally good k
    
        print(f"\nBest validation accuracy = {best_acc:.3f}")
        print(f"k values achieving this: {best_ks}")
        chosen_k = min(best_ks) 
        print(f"Chosen k = {chosen_k}")
    
        final_model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=chosen_k)
        )
        final_model.fit(X_train, y_train)
    
        return final_model, chosen_k
    return train_logistic, tune_knn


@app.cell
def _(X1_train, train_logistic, tune_knn, y1_train):
    # Dataset 1
    def optim_1():
        log1 = train_logistic(X1_train, y1_train)
        knn_model_1, final_k_1 = tune_knn(X1_train, y1_train)
        return log1, knn_model_1, final_k_1
    log1, knn_model_1, final_k_1 = optim_1()

    return final_k_1, knn_model_1, log1


@app.cell
def _(X2_train, train_logistic, tune_knn, y2_train):
    #Dataset 2
    def optim_2():
        log2 = train_logistic(X2_train, y2_train)
        knn_model_2, final_k_2 = tune_knn(X2_train, y2_train)
        return log2, knn_model_2, final_k_2
    log2, knn_model_2, final_k_2 = optim_2()
    return final_k_2, knn_model_2, log2


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


@app.cell
def _(accuracy_score, confusion_matrix, plt, sns):
    def evaluate_model(model, X_test, y_test, title):
        # Predict classes
        y_pred = model.predict(X_test)
    
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
    
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
    
        # Sensitivity (True Positive Rate for class 1)
        # TP / (TP + FN)
        sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
    
        # Specificity (True Negative Rate for class 0)
        # TN / (TN + FP)
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])

        # Print results
        print(f"\n=== {title} ===")
        print("Accuracy:    ", round(accuracy, 3))
        print("Sensitivity: ", round(sensitivity, 3))
        print("Specificity: ", round(specificity, 3))
        print("Confusion matrix:\n", cm)

        # Plot as colour heatmap
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")   # any colour map is fine
        plt.title(f"Confusion Matrix – {title}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.show()

    return (evaluate_model,)


@app.cell
def _(X1_test, evaluate_model, final_k_1, knn_model_1, log1, y1_test):
    #Dataset 1
    evaluate_model(log1, X1_test, y1_test, "Logistic Regression – Dataset 1")
    evaluate_model(knn_model_1, X1_test, y1_test, f"k-NN (k={final_k_1}) – Dataset 1")
    return


@app.cell
def _(X2_test, evaluate_model, final_k_2, knn_model_2, log2, y2_test):
    #Dataset 2
    evaluate_model(log2, X2_test, y2_test, "Logistic Regression – Dataset 2")
    evaluate_model(knn_model_2, X2_test, y2_test, f"k-NN (k={final_k_2}) – Dataset 2")
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


@app.cell
def _(plt, roc_auc_score, roc_curve):
    def plot_roc(model, X_test, y_test, title):
        # Probability estimates for the positive class (class 1)
        y_proba = model.predict_proba(X_test)[:, 1]
    
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
    
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0,1], [0,1], '--', color='grey')  # chance diagonal
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"ROC Curve – {title}")
        plt.legend()
        plt.grid(True)
        plt.show()
    
        print(f"{title} – AUC = {auc:.3f}")
        return auc

    return (plot_roc,)


@app.cell
def _(X1_test, final_k_1, knn_model_1, log1, plot_roc, y1_test):
    #Dataset1
    auc_log1 = plot_roc(log1, X1_test, y1_test, "Logistic Regression – Dataset 1")
    auc_knn1 = plot_roc(knn_model_1, X1_test, y1_test, f"k-NN (k={final_k_1}) – Dataset 1")
    return


@app.cell
def _(X2_test, final_k_2, knn_model_2, log2, plot_roc, y2_test):
    #Dataset2
    auc_log2 = plot_roc(log2, X2_test, y2_test, "Logistic Regression – Dataset 2")
    auc_knn2 = plot_roc(knn_model_2, X2_test, y2_test, f"k-NN (k={final_k_2}) – Dataset 2")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This assignment compared the performance of logistic regression and k-Nearest Neighbours (k-NN) on two synthetic binary classification datasets. The results demonstrate how differences in the underlying structure of the data strongly influence the relative performance of linear and non-linear classifiers.

    **For Dataset 1**, both models achieved perfect performance across all evaluation metrics: accuracy, sensitivity, specificity, and AUC were equal to 1.0. The scatter plot revealed that the dataset was almost perfectly linearly separable, with the two classes occupying distinct, non-overlapping regions of the feature space. Under these conditions, both logistic regression (which learns a global linear decision boundary) and k-NN (which classifies based on local neighbourhoods) were able to separate the classes without error. Dataset 1 therefore served as an perfect case illustrating how different models can perform equally well when the class structure is simple and cleanly separable.

    **Dataset 2** presented a more realistic and challenging classification problem. Although still broadly separable, the classes showed greater dispersion and partial overlap in the feature space as seen "in the middle" of scatter plot. In this situation, the performance gap between the models became more apparent. Logistic regression achieved higher accuracy (0.90), higher sensitivity (0.80), and a superior AUC (0.990), suggesting that its global linear boundary captured the overall structure of the data well. The tuned k-NN classifier (with k=5) still performed reasonably well overall, but showed lower sensitivity (0.60) and a reduced AUC (0.935), reflecting its struggle in the areas where the classes lie closer together This is consistent with k-NN reliance on local voting and density.

    Overall, these results highlight that when the decision boundary is roughly linear, as in Dataset 2, logistic regression can offer better generalisation than k-NN. Conversely, when the classes are cleanly separated, as in Dataset 1, both methods perform equally well. The findings reinforce the importance of visual exploration and model evaluation when choosing an appropriate classifier for a given dataset.
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


@app.cell
def _(
    StandardScaler,
    X1_test,
    X1_train,
    X2_test,
    X2_train,
    evaluate_model,
    make_pipeline,
    plot_roc,
    y1_test,
    y1_train,
    y2_test,
    y2_train,
):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    def train_svm(X_train, y_train):
        svm_model = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', probability=True, random_state=42)
        )
        svm_model.fit(X_train, y_train)
        return svm_model


    def train_random_forest(X_train, y_train):
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        return rf_model

    # Dataset 1
    svm1 = train_svm(X1_train, y1_train)
    rf1  = train_random_forest(X1_train, y1_train)
    evaluate_model(svm1, X1_test, y1_test, "SVM (RBF) – Dataset 1")
    auc_svm1 = plot_roc(svm1, X1_test, y1_test, "SVM (RBF) – Dataset 1")
    evaluate_model(rf1,  X1_test, y1_test, "Random Forest – Dataset 1")
    auc_rf1  = plot_roc(rf1,  X1_test, y1_test, "Random Forest – Dataset 1")

    # Dataset 2
    svm2 = train_svm(X2_train, y2_train)
    rf2  = train_random_forest(X2_train, y2_train)

    evaluate_model(svm2, X2_test, y2_test, "SVM (RBF) – Dataset 2")
    auc_svm2 = plot_roc(svm2, X2_test, y2_test, "SVM (RBF) – Dataset 2")

    evaluate_model(rf2,  X2_test, y2_test, "Random Forest – Dataset 2")
    auc_rf2  = plot_roc(rf2,  X2_test, y2_test, "Random Forest – Dataset 2")


    return


@app.cell
def _(mo):
    mo.md(
        r"""
    To extend the analysis, I also fitted two additional classifiers: an SVM with an RBF kernel and a Random Forest model, using exactly the same training/test splits and evaluation strategy.

    For **Dataset 1**, both SVM and Random Forest achieved perfect performance across all metrics (accuracy, sensitivity, specificity and AUC = 1.0). This matches what we saw before – Dataset 1 is so clearly separated which make it seem that any classifier can label the points perfectly.

    For **Dataset 2**, the results were more varied. The SVM achieved similar performance to k-NN, with accuracy 0.80, sensitivity 0.60, specificity 1.00, and AUC = 0.980. This suggests that, although the RBF kernel can model non-linear structure, the local overlap between the classes remains a limiting factor. In contrast, the Random Forest achieved perfect test performance (accuracy, sensitivity, specificity and AUC all equal to 1.0). According to my research, Random Forests are very adaptable and can pick up complex patterns in the data. In Dataset 2, this flexibility allows the model to follow the shape of the classes almost perfectly.

    Overall, these additional models reinforce the general pattern observed earlier. When the class boundary is simple and strongly separated (Dataset 1), all classifiers perform equally well. For the more complex and partially overlapping Dataset 2, linear logistic regression remains a competitive baseline, SVM and k-NN show moderate reductions in sensitivity, and the Random Forest comes out on top, as its flexibility allows it to handle the class structure better than the other models.
    """
    )
    return


@app.cell
def _(pd):
    results = {
        "Model": [
            "Logistic Regression (D1)", "k-NN (k=1) (D1)", "SVM (RBF) (D1)", "Random Forest (D1)",
            "Logistic Regression (D2)", "k-NN (k=5) (D2)", "SVM (RBF) (D2)", "Random Forest (D2)"
        ],
        "Accuracy":     [1.0, 1.0, 1.0, 1.0,   0.90, 0.80, 0.80, 1.0],
        "Sensitivity":  [1.0, 1.0, 1.0, 1.0,   0.80, 0.60, 0.60, 1.0],
        "Specificity":  [1.0, 1.0, 1.0, 1.0,   1.0,  1.0,  1.0,  1.0],
        "AUC":          [1.0, 1.0, 1.0, 1.0,   0.990, 0.935, 0.980, 1.0]
    }

    df_summary = pd.DataFrame(results)
    df_summary
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
