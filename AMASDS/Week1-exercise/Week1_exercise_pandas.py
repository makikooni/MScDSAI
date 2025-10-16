import marimo

__generated_with = "0.16.4"
app = marimo.App()


@app.cell
def _():
    # Import packages
    import marimo as mo

    from math import sqrt  # to calculate square root
    from matplotlib import pyplot as plt  # pyplot to plot graphs
    import numpy as np  # numpy to manipulate numerical data
    import pandas as pd  # pandas to load, handle, and save data
    return mo, np, pd, plt, sqrt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Data

    As an example, let’s get a dataset of people’s heights in cm. Every row is an entry for a different person.

    ## Upload data file

    On the left-hand side, select the Files tab (icon of a folder).

    In the Files tab, click the upload button (icon of a document with an upwards arrow).

    Select the ‘heights.csv’ file and upload.

    You can now access the file in the notebook.
    """
    )
    return


@app.cell
def _(pd):
    # Load the CSV as a Pandas DataFrame
    df = pd.read_csv('heights.csv')
    return (df,)


@app.cell
def _(df):
    # Preview the first few rows of the dataset
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Descriptive statistics

    Let’s calculate the mean, variance and standard distribution of the sample. For this, we have three functions: `mean`, `var` and `std`.

    Note how the different functions call each other. For example, `var` calls the `mean` function twice with different arguments.
    """
    )
    return


@app.cell
def _(df):
    # Get the height data as a numpy array from the DataFrame
    heights = df['height'].to_numpy()
    return (heights,)


@app.cell
def _(heights):
    ## Mean
    def mean(data, bessels_correction: bool=False) -> float:
        dof = len(data)  # Degrees of freedom: number of items in the dataset
        if bessels_correction:  # Bessel's correction to calculate sample var/std dev
          dof = dof - 1  # Reduce degrees of freedom
        sum = 0  # Start the sum at zero
        for value in data:  # For every item in the dataset
          sum = sum + value  # Add the value to the sum
        output = sum / dof  # Mean: normalise the sum by the degrees of freedom
        return output  # Return the mean

    heights_mean = mean(heights)
    print(f"Mean height: {heights_mean:.2f} cm")
    return heights_mean, mean


@app.cell
def _(heights, mean):
    ## Variance
    def var(data, bessels_correction: bool=True) -> float:
        data_mean = mean(data)  # Get sample mean
        deviations = [(value - data_mean)**2 for value in data]  # Squared deviations from the mean
        output = mean(deviations, bessels_correction)  # Mean of squared deviations
        return output  # Return variance

    heights_variance = var(heights)  # Mean of squared deviations
    print(f"Variance height: {heights_variance:.2f} cm")
    return (var,)


@app.cell
def _(heights, sqrt, var):
    ## Standard deviation
    def std(data, bessels_correction: bool=True) -> float:
        variance = var(data, bessels_correction)  # Get sample variance
        output = sqrt(variance)  # Calculate square root of variance
        return output  # Return standard deviation

    heights_stddev = std(heights)
    print(f"Standard deviation height: {heights_stddev:.2f} cm")
    return heights_stddev, std


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Visualise

    Histrogram plots are a good way to easily visualise the distribution of a random variable
    """
    )
    return


@app.cell
def _(df):
    # Plot the height data as a histogram
    df['height'].plot(kind='hist', bins=5)  # 20 bins
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The `bins` argument defines how many bars the sample data is split into.

    Try to find a good value for this dataset.
    """
    )
    return


@app.cell
def _(df):
    # Try different values of the bins argument
    df['height'].plot(kind='hist', bins=100)  # 100 bins
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Normal distribution

    The height of people roughly follows the bell curve of a normal distribution. Can you see it in the histogram plot?

    Below is the probability density function of the normal distribution.
    """
    )
    return


@app.cell
def _(np):
    ## PDF of the normal distribution
    def normal(_x: np.ndarray, loc: float=0.0, scale: float=1.0) -> np.ndarray:
        _var = np.square(scale)
        return 1.0 / np.sqrt(2 * np.pi * _var) * np.exp(-np.square(_x - loc) / (2 * _var))
    return (normal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot the distribution

    Try different values for the parameters `loc` ($\mu$) and `scale` ($\sigma$) and observe the changes.

    Make sure to adjust the range of values for x accordingly.
    """
    )
    return


@app.cell
def _(normal, np, plt):
    # Calculate values of the normal distribution
    #------check explanation of loc and correct answer?
    _x = np.arange(-100, 100, 0.1)  # Get an even range of x values
    _y = normal(_x, loc=0, scale=30)  # Calculate the corresponding y values
    _fig = plt.figure()
    # Figure
    plt.plot(_x, _y)
    plt.show()  # Plot the normal distribtion
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Compare empirical to theoretical distribution

    Now, let’s compare the sample distribution to the normal distribution with the corresponding parameters $\mu=$ `heights_mean` and $\sigma=$ `heights_stddev`.
    """
    )
    return


@app.cell
def _(heights, heights_mean, heights_stddev, normal, np, plt):
    27  # Calculate values of the normal distribution
    _x = np.arange(heights.min(), heights.max(), 1)  # Get an even range of x values
    _y = normal(_x, heights_mean, heights_stddev)  # Calculate the corresponding y values
    _fig = plt.figure()
    # Figure
    plt.hist(heights, bins=50, density=True)
    plt.plot(_x, _y)  # Plot the histogram
    plt.show()  # Plot the normal distribtion
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Question

    What is the range of heights that 68 % of people fall into?
    """
    )
    return


app._unparsable_cell(
    r"""
    175 +- 9.98 cm # Write your answer in this cell
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Different groups of people

    Apart from the height, the dataset has another column `group` with two different values.

    As an exercise, repeat the same calculations and visualisations for the two different groups.

    How do the statistics of the subgroup samples differ from each other?
    """
    )
    return


@app.cell
def _(df, mean, normal, np, plt, std, var):
    # Get group of people
    df_group1 = df[df['group'] == 1]  # All rows where 'group' equals 1
    heights_group1 = df_group1['height'].to_numpy()  # All heights for group == 1
    df_group1.head()

    heights_mean1 = mean(heights_group1)
    print(f"Mean height for group 1: {heights_mean1:.2f} cm")

    heights_variance1 = var(heights_group1)  # Mean of squared deviations
    print(f"Variance height: {heights_variance1:.2f} cm")

    heights_stddev1 = std(heights_group1)
    print(f"Standard deviation height: {heights_stddev1:.2f} cm")

    27  # Calculate values of the normal distribution
    _x = np.arange(heights_group1.min(), heights_group1.max(), 1)  # Get an even range of x values
    _y = normal(_x, heights_mean1, heights_stddev1)  # Calculate the corresponding y values
    _fig = plt.figure()
    # Figure
    plt.hist(heights_group1, bins=50, density=True)
    plt.plot(_x, _y)  # Plot the histogram
    plt.show()  # Plot the normal distribtion

    return (heights_variance1,)


@app.cell
def _(df, heights_variance1, mean, normal, np, plt, std, var):
    df_group2 = df[df['group'] == 2]  # All rows where 'group' equals 2
    heights_group2 = df_group2['height'].to_numpy()  # All heights for group == 2
    df_group2.head()

    heights_mean2 = mean(heights_group2)
    print(f"Mean height for group 2: {heights_mean2:.2f} cm")

    heights_variance2 = var(heights_group2)  # Mean of squared deviations
    print(f"Variance height: {heights_variance1:.2f} cm")

    heights_stddev2 = std(heights_group2)
    print(f"Standard deviation height: {heights_stddev2:.2f} cm")

    27  # Calculate values of the normal distribution
    _x = np.arange(heights_group2.min(), heights_group2.max(), 1)  # Get an even range of x values
    _y = normal(_x, heights_mean2, heights_stddev2)  # Calculate the corresponding y values
    _fig = plt.figure()
    # Figure
    plt.hist(heights_group2, bins=50, density=True)
    plt.plot(_x, _y)  # Plot the histogram
    plt.show()  # Plot the normal distribtion
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Bonus

    Can you find the `scipy` functions to calculate a sample’s

    - quartiles
    - ranges
    - skew
    - kurtosis

    https://docs.scipy.org/doc/scipy/
    """
    )
    return


@app.cell
def _():
    #TO DO 
    return


if __name__ == "__main__":
    app.run()
