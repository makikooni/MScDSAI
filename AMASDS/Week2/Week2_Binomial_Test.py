import marimo

__generated_with = "0.16.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Collect data""")
    return


@app.cell(hide_code=True)
def _(mo, pd):
    #heads(1)

    df = pd.DataFrame(
        data={
            'observation': [],
        }, 
        columns=['Observation'],
        dtype=int,
    )

    editor = mo.ui.data_editor(data=df)
    editor
    return (editor,)


@app.cell
def _(editor):
    data = editor.value
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Observed data""")
    return


@app.cell
def _(data):
    # Count the different values in the data
    data.value_counts()
    return


@app.cell
def _(data):
    # Plot the discrete distribution over values
    data.value_counts().plot(kind='bar', title='Observed distribution', xlabel='Value', ylabel='Frequency')  # Bar plot of counted values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Expected data

    Which distribution did you expect to observe?
    """
    )
    return


@app.cell
def _(np, plt):
    # Fill in your expectations
    # Probabilities correspond to values

    expected_vals = np.array([0,1])  # Values
    expected_probs = np.array([0.5,0.5])  # Probabilities

    fig = plt.figure()
    plt.bar(expected_vals, expected_probs)
    plt.xticks(expected_vals)
    plt.xlabel('Values')
    plt.xlabel('Probability')
    plt.title('Expected')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Your hypotheses

    Write your hypothesis in the text cell below
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Null hypothesis: The success of getting Heads has 50% probability.
    Head is a succeess.

    Alternative hypothesis: The coin natural weightening and starting side makes it not fair.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Binomial test""")
    return


@app.cell
def _():
    from scipy.stats import binomtest
    return (binomtest,)


@app.cell
def _(binomtest):
    # One-sided test
    n = 30   # number of trials
    k = 13    # number of successes
    p = 1./2  # success probability of single trial

    test = binomtest(k, n, p, alternative="greater")
    print(test)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
