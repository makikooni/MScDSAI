import marimo

__generated_with = "0.16.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Propaganda Detection Tool

    A group of researchers developed an automated propaganda detection tool designed to nudge readers towards more critical news consumption. Using Large Language Models, the tool detects propaganda in news articles and provides context-rich explanations, enhancing users’ understanding and critical thinking.

    To collect evidence for the benefits of the tool, the researchers performed a study where participants used the tool while reading online news articles. The experiment was conducted following a between-subject design, meaning each participant was assigned to a single condition. The participants were randomly assigned to one of the three groups:

    1. No propaganda detection tool (Basic),
    2. Propaganda detection tool without explanations (Light), and
    3. Propaganda detection tool with explanations (Full)

    There are different number of datapoints for the conditions, because not all participants completed the study.

    Each participant read two news articles that contained elements of propaganda. Depending on the group (2 & 3), they used the propaganda detection tool that either only marks detected propaganda (Light) or also provides explanations of propaganda (Full). The control group (Basic) received unmarked texts. After reading each of the articles, participants completed an online survey.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _conditions = mo.image(src="conditions.png")
    mo.md(
        f"""
        {_conditions}
        Examples of news articles shown to participants in the three different experimental conditions. Left: Basic, no text highlights; Middle: Light, the tool highlights detected propaganda; Right: Full, the tool highlights detected propaganda and provides explanations.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Task 1: Package import

    Import the appropriate statistical tests from the Scipy stats package
    """
    )
    return


app._unparsable_cell(
    r"""
    from matplotlib import pyplot as plt
    from scipy.stats import   # complete this line
    import pandas as pd
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Task 2: Data import

    Upload the data file, import it and get the data for the different conditions from the columns of the dataframe.

    You might have to remove missing values (NaN) from the end of the column data.

    How many participants were there for each condition?
    """
    )
    return


@app.cell
def _(pd):
    # Data import
    filename = ''  # complete
    df = pd.read_csv(filename)

    df  # Show data
    return


@app.cell
def _():
    # Get data from columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Task 3: ANOVA

    Question: Is there a statistically significant difference between the three conditions?
    """
    )
    return


@app.cell
def _():
    # Perform a one-way ANOVA
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Task 4: Post-hoc t-tests

    The ANOVA can only tell us whether there is a signifcant different between groups or not. But it cannot tell between which groups.

    Question: Between which groups exists a significant difference?

    Form a hypothesis and perform the corresponding t-test.
    """
    )
    return


@app.cell
def _():
    # Post-hoc t-tests between groups
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Task 5: False Discovery Rate

    We are performing post-hoc t-tests on data that we have previously analysed with an ANOVA. There is a risk that we encounter significant results purely by chance. So, we need to adjust the p-values, for example with the Benjamini-Hochberg method.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html
    """
    )
    return


@app.cell
def _():
    # Adjustment of p-values to control the false discovery rate
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Task 6: Conclusions

    Based on the results, which conclusions do you draw about the effectiveness of the propaganda detection tool?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Bonus Task: Visualisation

    Find a good visualisation of the results that shows the differences in the score between the three experimental conditions (Basic, Light, Full).

    Hint: Which descriptive statistics are important in the calculation of the significance tests used above? How can you visualise these statistics?
    """
    )
    return


@app.cell
def _():
    # Plot the results
    return


if __name__ == "__main__":
    app.run()
