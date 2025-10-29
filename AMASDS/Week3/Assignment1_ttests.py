import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Assignment 1: t-tests

    Week 3

    For this assignment, you are asked to perform significance tests for three different datasets:
    1. People’s heights
    2. Fashion Design Tool
    3. Image editing software

    You will have to choose the right test for every occasion, depending on the characteristics of the data and how it was collected.

    The specific instructions and questions are described in each section below. There are some bonus tasks which are not mandatory but can gain you extra points.
    """
    )
    return


@app.cell
def _():
    # Import packages
    import marimo as mo
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy
    import statsmodels.api as sm
    import seaborn as sns


    #Ref:https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    return mo, np, pd, plt, scipy, sm, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## People’s heights

    In week 1, we looked at people’s heights as example of data that follow the normal distribution (remember the central limit theorem).

    There are two groups of people. Is there a significant difference between the heights of the two groups?

    ### Instructions
    1. Upload the dataset to the notebook environment (`heights.csv`)
    2. Perform the adequate statistical test for the data
    3. Answer the questions below separately for both studies

    ### Questions
    1. How many datapoints do we have in the two groups and all together?
    2. Which test do you choose and why?
    3. What are the degrees of freedom of the test?
    4. What are the alternative and null hypotheses of the test?
    5. Is the result of the test significant?
    6. Would you reject the null hypothesis or not?
    7. What are you conclusions about the research question?

    ### Bonus (extra points)
    - Question: Which other test would be adequate for this data?
    - Perform the same analysis with the other tests that you think are adequate
    - Compare and discuss the results of the different tests
    """
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv("AMASDS/Week3/heights.csv")
    df.head()
    return (df,)


@app.cell
def _(df):
    #df.info()
    group1 = df.loc[df['group'] == 1]
    group1count = group1.shape[0]

    group2 = df.loc[df['group'] == 2]
    group2count = group2.shape[0]

    print("We have " + str(df.shape[0]) + " rows in total.")
    print("We have " + str(group1count) + " rows in group 1.")
    print("We have " + str(group2count) + " rows in group 2.")
    return group1, group1count, group2, group2count


@app.cell
def _(group1, group2, np, plt, sm, sns):
    #Groups preparation
    group1_heights = group1['height']
    mean1 = np.mean(group1_heights)
    std1 = np.std(group1_heights)
    group2_heights = group2['height']
    mean2 = np.mean(group2_heights)  
    std2 = np.std(group2_heights)
    print(f"Group 1: mean = {mean1:.1f}, std = {std1:.1f}")
    print(f"Group 2: mean = {mean2:.1f}, std = {std2:.1f}")

    #Distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=group1_heights, label='Group 1', alpha=0.7, kde=True)
    sns.histplot(data=group2_heights, label='Group 2', alpha=0.7, kde=True)
    plt.axvline(group1_heights.mean(), color='blue', linestyle='--', alpha=0.8, label=f'Group 1 Mean: {group1_heights.mean():.1f}cm')
    plt.axvline(group2_heights.mean(), color='orange', linestyle='--', alpha=0.8, label=f'Group 2 Mean: {group2_heights.mean():.1f}cm')
    plt.xlabel('Height (cm)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Heights: Group 1 vs Group 2')
    plt.legend()
    plt.savefig('height_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    #Q-Q plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Group 1 
    sm.qqplot(group1_heights, loc=mean1, scale=std1, line='45', ax=ax1)
    ax1.set_title('Q-Q Plot: Group 1 Heights')
    ax1.set_xlabel('Theoretical Quantiles')
    ax1.set_ylabel('Sample Quantiles')

    # Group 2 
    sm.qqplot(group2_heights, loc=mean2, scale=std2, line='45', ax=ax2)
    ax2.set_title('Q-Q Plot: Group 2 Heights') 
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles')

    plt.tight_layout()
    plt.savefig('qplots.png', dpi=150, bbox_inches='tight')
    plt.show()
    return group1_heights, group2_heights


@app.cell
def _(
    group1,
    group1_heights,
    group1count,
    group2,
    group2_heights,
    group2count,
    scipy,
):
    scipy.stats.ttest_ind(group1_heights, group2_heights)
    t_stat, p_val= scipy.stats.ttest_ind(group1['height'], group2['height'])
    print("t = " + str(t_stat))
    print("p = " + str(p_val))
    degrees = group1count + group2count - 2
    print("df = " + str(degrees))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    #Report on People’s heights
    1. How many datapoints do we have in the two groups and all together?
        This dataset contains 7912 datapoints in total divided into two groups with 3956 in each. It's people heights in cm.

    2. Which test do you choose and why?
        The chosen test to check for a significant difference between the two groups' mean heights is an independent two-sample, two-sided t-test. This test is appropriate because both groups have equal sample sizes (n=3,956), the variable (height) is continuous, and the data meet the key assumptions for this test. Specifically, the assumption of approximately normal distributions was verified through Q-Q plots, and the assumption of equal variances was supported by the similar standard deviations (Group 1 SD=7.6, Group 2 SD=7.1, ratio=1.07). The t-test is designed to answer one critical question: "How likely is it to see a difference this large (or larger) between two sample means, if the null hypothesis is true and there is actually no difference in the population?"

    3. What are the degrees of freedom of the test?
        In a two-sample t-test, the degrees of freedom are based on the number of data points in both groups df=n1​+n2​−2 therefore for this test it equals 7910.

    4. What are the alternative and null hypotheses of the test?
        The null hypothesis (H0) states that the two independent samples have identical population means. In other words, there is no significant difference between the average heights of the two groups. The alternative hypothesis (H1) states that the population means are not equal (two-sided test).

    5. Is the result of the test significant?
        The mean height for Group 1 is 178.29 cm, while the mean for Group 2 is 164.79 cm. These values differ substantially in practical terms. TThe t-test result produced a t-statistic of 81.78 and a p-value < 0.001. This provides overwhelming statistical evidence for a difference (significant even at a 0.1% level), which, combined with the substantial 13.5 cm difference in means, confirms a result of both statistical and great practical significance.

    6. Would you reject the null hypothesis or not?
        Given the extremely small p-value, I reject the null hypothesis and conclude that there is a statistically significant difference between the mean heights of the two groups.
    8. What are you conclusions about the research question?
        Based on the results of the independent two-sample t-test, we reject the null hypothesis. The analysis provides overwhelming evidence that there is a statistically significant difference in the mean heights between Group 1 and Group 2. This conclusion is supported by an exceptionally large t-statistic of 81.78 and a p-value of less than 0.001, which is significant at any conventional threshold (α = 0.05, 0.01, etc.). Furthermore, the observed difference of 13.5 cm is substantial in magnitude, confirming that the finding is not only statistically significant but also of considerable practical importance. Therefore, we conclude that group membership is strongly associated with a meaningful difference in average height.


    Figures:

    Figure 1: Distribution of heights for both groups, showing the substantial mean difference.
    {mo.image("AMASDS/Week3/height_distribution.png")}
    Figure 2: Q-Q plots assessing normality assumption for both groups.
    {mo.image("AMASDS/Week3/qplots.png")}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fashion Design Tool

    A group of researchers developed a generative AI system with a graphical interface to support the ideation process in a fashion design task. The question is if the system is actually beneficial to the ideation process.

    In two separate studies, several participants were invited to test the system (C) and compare it against two alternatives. In *Study 1*, the system is compared against Google Images (A). In *Study 2*, the system is compared against Stable Diffusion (B).

    The participants were asked to imagine working for a fashion design agency tasked with creating new dress designs for a client, and to “come up with a variety of styles with diferent colors, patterns, and textures… that should be creative, but not too impractical”.

    The study procedure was the following. First, participants would perform the task with the alternative system (A or B) for 10–15 minutes. They would then score their experience on the Creativity Support Index (CSI). Second, they would to the same task with the generative AI system (C). Afterwards, they score their experience again on the CSI.

    The datasets for the two studies consist of the scores on the Creativity Support Index (CSI).

    ### Instructions
    1. Upload the two datasets to the notebook environment (`fashion_study1.csv`, `fashion_study2.csv`)
    2. Perform the adequate statistical test separately for the data from the two studies
    3. Answer the questions below separately for both studies

    ### Questions
    1. How many people participated in the study?
    2. Which test do you choose and why?
    3. What are the degrees of freedom of the test?
    4. What are the alternative and null hypotheses of the test?
    5. Is the result of the test significant?
    6. Would you reject the null hypothesis or not?
    7. What are you conclusions about the research question?

    ### Bonus (extra points)
    - Create plots to separately visualise the results of the two studies
    """
    )
    return


@app.cell
def _(mo):
    mo.image("AMASDS/Week3/fashion_study.png")
    return


@app.cell
def _():
    # Write your code here ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Write your report and answers here

    [double-click to edit]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Image editing software

    A team of software developers built a new image editing software. They want to better understand who the software might benefit more: experts or amateurs. The question is if the software is better suited for experts, the general population or both.

    To test this, they run a human participant study. In the study they ask the participants to perform an image editing task with their software and measure the time it takes a participant to complete it. They recruit several participants for the study, but do not manage to find as many experts as non-experts.

    ### Instructions
    1. Upload the two datasets to the notebook environment (`imgedit_experts.csv`, `imgedit_nonexperts.csv`)
    2. Perform the adequate statistical test
    3. Answer the questions below

    ### Questions
    1. How many people participated in the study in the two groups?
    2. Which test do you choose and why?
    3. What are the degrees of freedom of the test?
    4. What are the alternative and null hypotheses of the test?
    5. Is the result of the test significant?
    6. Would you reject the null hypothesis or not?
    7. What are you conclusions about the research question?
    """
    )
    return


@app.cell
def _():
    # Write your code here ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Write your report and answers here

    [double-click to edit]
    """
    )
    return


if __name__ == "__main__":
    app.run()
