import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from matplotlib import pyplot as plt
    import numpy as np
    import scipy
    return mo, np, plt, scipy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data

    Some example data
    """
    )
    return


@app.cell
def _(np):
    x = np.array([0.3000, -0.7700, 0.9000, -0.0400, 0.7400, -0.5800, -0.9200, -0.2100, -0.5400, 0.6800])
    y = np.array([1.1492,  0.3582, 1.9013,  0.9487, 1.3096,  0.9646,  0.1079,  1.1262,  0.6131, 1.0951])
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 1: Data Visualisation

    Visualise the data as a scatter plot

    Use the pyplot function scatter
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    """
    )
    return


@app.cell
def _(plt, x, y):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=40, alpha=0.7)
    plt.title("Relationship between X and Y")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 2: Covariance and Correlation

    Calculate the covariance and the correlation between the two random variables

    You may use the function pearsonr from the scipy stats package
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    """
    )
    return


@app.cell
def _(np, x, y):
    #Covariance
    cov_matrix = np.cov(x, y)
    cov=cov_matrix[0,1]

    def interpret_cov(cov):
        if cov > 0:
            return "Positive covariance: x and y tend to increase together. Magnitude is not standardised."
        elif cov < 0:
            return "Negative covariance: as x increases, y tends to decrease. Magnitude is not standardised."
        else:
            return "Zero covariance: no linear co-movement."


    print(f"Covariance between x and y:")
    print(f"cov = {cov:.3f}\n")
    print(interpret_cov(cov))


    return


@app.cell
def _(scipy, x, y):
    #Correlation
    r, p = scipy.stats.pearsonr(x, y)

    def interpret_r(r):
        if r == 1:
            return "Perfect positive linear relationship (no scatter)."
        elif r >= 0.8:
            return "Strong positive linear relationship (little scatter)."
        elif r >= 0.4:
            return "Weak positive linear relationship (lots of scatter)."
        elif r > -0.4:
            return "No linear relationship (complete scatter)."
        elif r > -0.8:
            return "Weak negative linear relationship (lots of scatter)."
        elif r > -1:
            return "Strong negative linear relationship (little scatter)."
        else:  # r == -1
            return "Perfect negative linear relationship (no scatter)."


    print(f"Correlation between x and y:")
    print(f"r = {r:.3f}\n")
    print(interpret_r(r))

    return (interpret_r,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 3: Regression

    Find the best linear model

    Use the scipy stats function linregress to optimise the *intercept* and *slope* parameters
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
    """
    )
    return


@app.cell
def _(interpret_r, scipy, x, y):
    result = scipy.stats.linregress(x, y)

    slope = result.slope
    intercept = result.intercept
    r_value = result.rvalue
    p_value = result.pvalue

    def interpret_pvalue(p):
        if p < 0.001:
            return "Highly statistically significant p < 0.001 "
        elif p < 0.01:
            return "Very statistically significant p < 0.01 "
        elif p < 0.05:
            return "p < 0.05 - Statistically significant p < 0.05 - "
        else:
            return "Not statistically significant (p ≥ 0.05)"

    print("=== BEST LINEAR MODEL ===\n")
    print(f"Slope (w1): {slope:.4f}\n")
    print(f"Intercept (w0): {intercept:.4f}\n")
    print(f"Model Equation: y = {intercept:.4f} + {slope:.4f} * x\n")
    print(f"Correlation (r): {r_value:.4f} - {interpret_r(r_value)}\n")
    print(f"P-value: {p_value:.4f} - {interpret_pvalue(p_value)}\n")

    return intercept, slope


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 4: Model Visualisation

    Reuse the scatter plot from task 1 and plot the linear model as a line on top of the data

    You will need to:
    1. Use the range of dummy x-values `x2`
    2. Use the model to calculate the corresponding y-values `y2`
    3. Plot the line
    """
    )
    return


@app.cell
def _(intercept, np, plt, slope, x, y):
    x2 = np.arange(x.min(), x.max(), 0.1)  # Dummy x-values
    y2 = intercept + slope * x2 #regression equation

    # Continue with plotting
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=40, alpha=0.7, label='Data points')
    plt.plot(x2, y2, color='pink', linewidth=2, label='Linear model')
    plt.title("Linear Regression Model")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 5: Mean squared error

    Calculate the mean squared error for the linear model

    You will need to:
    1. Get the predicted values $\hat{y}$ from the model for each data point $x$
    2. Calculate the squared difference from the actual data $y$
    3. Sum the squared differences over all data points
    4. Normalise by the number of data points $n$
    """
    )
    return


@app.cell
def _(intercept, np, slope, x, y):
    # 1. Get the predicted values ŷ from the model for each data point x
    y_pred = intercept + slope * x

    # 2. Calculate the squared difference from the actual data y
    squared_errors = (y - y_pred) ** 2

    # 3. Sum the squared differences over all data points
    sum_squared_errors = np.sum(squared_errors)

    # 4. Normalise by the number of data points n
    mse = sum_squared_errors / len(x)

    # Numpy version
    #mse = np.mean((y - y_pred) ** 2)

    def interpret_mse(mse):
        if mse < 0.01:
            return "Excellent model fit (very low error)"
        elif mse < 0.1:
            return "Good model fit (low error)"
        elif mse < 0.5:
            return "Moderate model fit (reasonable error)"
        else:
            return "Poor model fit (high error)"


    print("=== MEAN SQUARED ERROR CALCULATION ===\n")
    print(f"Predicted values: {y_pred}\n")
    print(f"Squared errors: {squared_errors}\n")
    print(f"Sum of squared errors: {sum_squared_errors:.4f}\n")
    print(f"MSE (sum / {len(x)}): {mse:.4f} - {interpret_mse(mse)}")

    return


if __name__ == "__main__":
    app.run()
