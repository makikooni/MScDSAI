import marimo

__generated_with = "0.17.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Effects of Music on VR Player Experience

    How much does music contribute to player experience in virtual reality (VR) games? A group of researchers conducted an exploration of how music affects time perception in a VR game.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _studyflow = mo.image(src="studyflow.png")
    _gameplay = mo.image(src="gameplay.png")

    mo.md(f"""
    {_studyflow}

    In a user study (N=64), they investigated the effects of music on players’ time perception (measured as retrospective time estimation). The study employed a between-subjects design with presence of music as the independent variable (with-music vs. no-music, see figure below). The primary research question was: *How does the presence of music affect time perception in a VR game?*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The researchers recruited a total of N=64 participants (median age of 22, IQR=20–25) from the university via mailing lists and flyers. Participants were evenly divided into the two sound conditions, balanced by self-reported gender (per group: one non-binary, 12 female, and 19 male participants).""")
    return


@app.cell(hide_code=True)
def _(mo):

    _gameplay = mo.image(src="gameplay.png")

    mo.md(f"""
    {_gameplay}

    Players were asked to play a bow-and-arrow tower defence VR game (screenshots in the figure below). In the game, players defended their position against orcs (a) and flying pegators (b), which dissolved upon being killed (scoring points) or as they reached the portal (c, d) behind the player (losing points).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Package imports""")
    return


@app.cell
def _():
    from matplotlib import pyplot as plt
    import numpy as np
    import scipy
    import pandas as pd
    return np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data

    We measured retrospective time estimation (RTE). The participants were not informed that they would later be asked to estimate the duration of play to assess the remembered duration. After the game, we asked participants to estimate the duration of play.

    Participants were asked to choose one of eleven RET intervals, starting from 61–90 seconds and ending with 361–390 seconds. With 272 seconds as the correct duration of the game, the eighth interval was the right choice. However, the seventh interval was also accepted as correct for the subsequent analysis because it was so close.

    Additionally, the researchers defined the relaxed absolute difference (RAD) as a metric for how close participants’ RTE was to the correct answer. An absolute measure was chosen as otherwise over vs. underestimations would skew each other, and we were interested in correct vs. incorrect time estimation. The measure was relaxed in the sense that both intervals 7 and 8 counted as correct (i.e., RAD = 0), intervals 6 and 9 both counted as RAD = 1, etc.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _table = mo.ui.table(
        data=[
            {
                "Time (sec)": "RET", 
                "61–90": 1, 
                "91–120": 2, 
                "121–150": 3, 
                "151–180": 4, 
                "181–210": 5, 
                "211–240": 6, 
                "241–270": 7, 
                "271–300": 8, 
                "301–330": 9, 
                "331–360": 10, 
                "361–390": 11,
            },
            {
                "Time (sec)": "RAD", 
                "61–90": 6, 
                "91–120": 5, 
                "121–150": 4, 
                "151–180": 3, 
                "181–210": 2, 
                "211–240": 1, 
                "241–270": 0, 
                "271–300": 0, 
                "301–330": 1, 
                "331–360": 2, 
                "361–390": 3,
            },
        ],
        label="Time Intervals",
    )

    mo.md(f"""
    {_table}

    The table above gives an overview of the time intervals and the corresponding RET and RAD values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Task: Data import

    Upload the data file, import it and get the data for the different conditions from the columns of the dataframe.
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
    ### Task: Convert RTE data to RAD scores

    Use the helper function to convert the data
    """
    )
    return


@app.cell
def _(np):
    # Helper function: conversion of RTE to RAD scores
    def rte_to_rad(rte):
        rad = np.zeros_like(rte)
        for i, score in enumerate(rte):
            if score < 7:  # Too low
                rad[i] = np.abs(score - 7)
            elif score > 8:  # Too high
                rad[i] = np.abs(score - 8)
        return rad
    return


@app.cell
def _():
    # Convert the data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Visualisation

    Visualise the RTE and RAD data in separate figures as boxplots, comparing the with-music condition to the without-music condition.

    *Question*:
    What are the median values of the different groups of samples?
    """
    )
    return


@app.cell
def _():
    # Plot RTE data
    return


@app.cell
def _():
    # Plot RAD data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Analysis

    ### Task: Significance Test

    Test the RTE and RAD data separately to answer the research question: *How does the presence of music affect time perception in a VR game?*

    Choose the appropriate significance test and justify your choice.

    Scipy documentation of statistical functions:
    https://docs.scipy.org/doc/scipy/reference/stats.html
    """
    )
    return


@app.cell
def _():
    # Significance testing RTE
    return


@app.cell
def _():
    # Significance testing RAD
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Your Report

    Answer the following questions:

    1. Which tests did you choose and why?
    2. What are the alternative and null hypotheses of the tests?
    3. Are the results of the test significant?
    4. Would you reject the null hypothesis or not?
    5. What are you conclusions about the research question?
    """
    )
    return


if __name__ == "__main__":
    app.run()
