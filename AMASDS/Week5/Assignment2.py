import marimo

__generated_with = "0.16.5"
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
    mo.md(f"""In a user study (N=64), they investigated the effects of music on players’ time perception (measured as retrospective time estimation). The study employed a between-subjects design with presence of music as the independent variable (with-music vs. no-music, see figure below). The primary research question was: *How does the presence of music affect time perception in a VR game?*""")
    return


@app.cell
def _(mo):
    studyflow = mo.image(src="AMASDS/Week5/studyflow.png")
    studyflow
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The researchers recruited a total of N=64 participants (median age of 22, IQR=20–25) from the university via mailing lists and flyers. Participants were evenly divided into the two sound conditions, balanced by self-reported gender (per group: one non-binary, 12 female, and 19 male participants).""")
    return


@app.cell(hide_code=True)
def _(mo):

    _gameplay = mo.image(src="AMASDS/Week5/gameplay.png")

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
    return np, pd, plt, scipy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data

    We measured retrospective time estimation (RTE). The participants were not informed that they would later be asked to estimate the duration of play to assess the remembered duration. After the game, we asked participants to estimate the duration of play.

    Participants were asked to choose one of eleven RTE intervals, starting from 61–90 seconds and ending with 361–390 seconds. With 272 seconds as the correct duration of the game, the eighth interval was the right choice. However, the seventh interval was also accepted as correct for the subsequent analysis because it was so close.

    Additionally, the researchers defined the relaxed absolute difference (RAD) as a metric for how close participants’ RTE was to the correct answer. An absolute measure was chosen as otherwise over vs. underestimations would skew each other, and we were interested in correct vs. incorrect time estimation. The measure was relaxed in the sense that both intervals 7 and 8 counted as correct (i.e., RAD = 0), intervals 6 and 9 both counted as RAD = 1, etc.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _table = mo.ui.table(
        data=[
            {
                "Time (sec)": "RTE", 
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

    The table above gives an overview of the time intervals and the corresponding RTE and RAD values.
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
    filename = 'AMASDS/Week5/vr_rte.csv'  # complete
    df = pd.read_csv(filename)

    df  # Show data
    return (df,)


@app.cell
def _(df):
    # Get data from columns
    rte_music = df['With music'].tolist()
    print(rte_music)
    rte_nomusic = df['Without music'].tolist()
    print(rte_nomusic)
    return rte_music, rte_nomusic


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
    return (rte_to_rad,)


@app.cell
def _(rte_music, rte_nomusic, rte_to_rad):
    # Convert the data
    rad_music=rte_to_rad(rte_music)
    rad_nomusic = rte_to_rad(rte_nomusic)
    return rad_music, rad_nomusic


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
def _(plt, rte_music, rte_nomusic):
    # Plot RTE data
    plt.figure(figsize=(6, 5))
    plt.boxplot([rte_music, rte_nomusic], tick_labels=['Music', 'No Music'])
    plt.title('Retrospective Time Estimation (RTE)')
    plt.ylabel('Chosen RTE Interval')
    plt.figtext(
        0.5, -0.15,
        "Boxplot comparing retrospective time estimations (RTE) between the with-music and without-music conditions. \n"
        "Values represent the interval chosen by participants (1–11), where higher interval numbers indicate longer perceived durations. \n"
        "The correct interval was 8 (271–300 seconds).",
        wrap=True, ha='center', fontsize=9
    )
    plt.savefig("rad_music.pdf")
    plt.show()
    return


@app.cell
def _(pd, rte_music, rte_nomusic):
    rte_music_series_median = (pd.Series(rte_music)).median()
    rte_nomusic_series_median = (pd.Series(rte_nomusic)).median()
    print("RTE median of Music group = " + str(rte_music_series_median))
    print("RTE median of Non-music group = " + str(rte_nomusic_series_median))
    return


@app.cell
def _(plt, rad_music, rad_nomusic):
    # Plot RAD data
    plt.figure(figsize=(6, 5))
    plt.boxplot([rad_music, rad_nomusic], tick_labels=['Music', 'No Music'])
    plt.title('Relaxed Absolute Difference (RAD)')
    plt.ylabel('Relaxed Absolute Difference (RAD)')

    plt.figtext(
        0.5,                
        -0.05,             
        "Lower RAD values indicate more accurate time estimation.\n "
        "RAD = 0 corresponds to correct intervals (7–8);\n higher values show greater deviation.",
        wrap=True,
        ha='center',        
        fontsize=9
    )
    plt.show()
    return


@app.cell
def _(pd, rad_music, rad_nomusic):
    rad_music_series_median = (pd.Series(rad_music)).median()
    rad_nomusic_series_median = (pd.Series(rad_nomusic)).median()
    print("RAD median of Music group = " + str(rad_music_series_median))
    print("RAD median of Non-music group = " + str(rad_nomusic_series_median))
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
def _(rte_music, rte_nomusic, scipy):
    # Significance testing RTE
    _U1, _p = scipy.stats.mannwhitneyu(rte_music, rte_nomusic,method='auto')
    print("U1: " + str(_U1))
    print("P:" + str(_p))
    if _p < 0.05:
        print("Reject H0 — significant difference")
    else:
        print("Fail to reject H0 — no significant difference")
    return


@app.cell
def _(rad_music, rad_nomusic, scipy):
    # Significance testing RAD
    U1, p = scipy.stats.mannwhitneyu(rad_music, rad_nomusic,method='auto')
    print("U1: " + str(U1))
    print("P:" + str(p))
    if p < 0.05:
        print("Reject H0 — significant difference")
    else:
        print("Fail to reject H0 — no significant difference")

    #RAD is already a “distance from correct interval” metric, which may produce more consistent differences between groups, hence smaller p (0.008).
    return


@app.cell
def _(np, rte_music, rte_nomusic, scipy):
    #Manual Check
    n1 = len(rte_music)
    n2 = len(rte_nomusic)
    U = 705

    mu_U = n1*n2 / 2
    sigma_U = np.sqrt(n1*n2*(n1+n2+1)/12)
    z = (U - mu_U -0.5) / sigma_U #with continuity correction
    print(z)
    print("Z check:")
    if abs(z) > 1.96:
        print("Reject H0 — significant difference\n")
    else:
        print("Fail to reject H0 — no significant difference\n")

    _p = 2 * (1 - scipy.stats.norm.cdf(abs(z)))  # two-tailed
    print(_p) #continuous normal approximation - slightly different result

    print("P check:")
    if _p < 0.05:
        print("Reject H0 — significant difference")
    else:
        print("Fail to reject H0 — no significant difference")
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


@app.cell
def _(mo):
    mo.md(
        r"""
    1. The analyses were conducted using non-parametric tests because the research question — *How does the presence of music affect time perception in a VR game?* — involves ordinal data that are not normally distributed, and population parameters could not be reliably estimated. Given that there are two independent groups (with-music vs. without-music), the **Mann–Whitney U test** was chosen to compare participants’ responses between conditions.
    2. **H₀**: Music has no effect on time perception in VR games.<br>
       **H₁**: Music affects time perception in VR games. <br>
       **IV**; Condition: Music vs No Music <br>
       **DV**; Time perception	Measured via RTE (Retrospective Time Estimation) and RAD (Relaxed Absolute Difference).
    3. The p-values for both RTE and RAD were below 0.05 (RTE: p = 0.038; RAD: p = 0.008), indicating that the differences between the music and no-music conditions were statistically **significant**. 
    4. Based on the results, **H₀ is rejected** for both measures.
    5. Conclusion: The findings suggest that background music significantly influences time perception in VR games. Participants’ estimated durations (RTE) and their accuracy (RAD) differed between the music and no-music conditions, supporting the hypothesis that presence of music affects perceived time.
    """
    )
    return


if __name__ == "__main__":
    app.run()
