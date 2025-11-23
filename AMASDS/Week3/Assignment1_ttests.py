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
    # People’s heights

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
    df = pd.read_csv("heights.csv")
    df.head()
    return


@app.cell
def _(np, pd, plt, scipy, sm, sns):
    def task1():
    
        df = pd.read_csv("heights.csv")
        #df.head()
    
        group1 = df.loc[df['group'] == 1]
        group1count = group1.shape[0]
    
        group2 = df.loc[df['group'] == 2]
        group2count = group2.shape[0]
    
        print("We have " + str(df.shape[0]) + " rows in total.")
        print("We have " + str(group1count) + " rows in group 1.")
        print("We have " + str(group2count) + " rows in group 2.")
    
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
        plt.savefig('height_distribution_task1.png', dpi=150, bbox_inches='tight')
        plt.show()
    
        #Q-Q plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
        sm.qqplot(group1_heights, loc=mean1, scale=std1, line='45', ax=ax1)
        ax1.set_title('Q-Q Plot: Group 1 Heights')
        ax1.set_xlabel('Theoretical Quantiles')
        ax1.set_ylabel('Sample Quantiles')
     
        sm.qqplot(group2_heights, loc=mean2, scale=std2, line='45', ax=ax2)
        ax2.set_title('Q-Q Plot: Group 2 Heights') 
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Sample Quantiles')
    
        plt.tight_layout()
        plt.savefig('qplots_task1.png', dpi=150, bbox_inches='tight')
        plt.show()
    
        #Test
        scipy.stats.ttest_ind(group1_heights, group2_heights)
        t_stat, p_val= scipy.stats.ttest_ind(group1['height'], group2['height'])
        print("t = " + str(t_stat))
        print("p = " + str(p_val))
        degrees = group1count + group2count - 2
        print("df = " + str(degrees))
    
        alpha = 0.05
        t_critical = scipy.stats.t.ppf(1 - alpha/2, degrees)
        print("Two-tailed t-critical:", t_critical)
    
        if abs(t_stat) > t_critical:
            print("Reject H0")
        else:
            print("Fail to reject H0")
    task1()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    #Task 1) Report on People’s heights
    ##1. How many datapoints do we have in the two groups and all together?

        This dataset contains 7912 datapoints in total divided into two groups with 3956 in each. It's people heights in cm.

    ##2. Which test do you choose and why?

        The chosen test to check for a significant difference between the two groups' mean heights is an independent two-sample, two-sided t-test. This test is appropriate because both groups have equal sample sizes (n=3,956), the variable (height) is continuous, and the data met the key assumptions for this test. Specifically, the assumption of approximately normal distributions was verified through Q-Q plots, and the assumption of equal variances was supported by the similar standard deviations (Group 1 SD=7.6, Group 2 SD=7.1, ratio=1.07). The t-test is designed to answer one critical question: "How likely is it to see a difference this large (or larger) between two sample means, if the null hypothesis is true and there is actually no difference in the population?"

    ##3. What are the degrees of freedom of the test?

        In a two-sample t-test, the degrees of freedom are based on the number of data points in both groups df=n1​+n2​−2 therefore for this test it equals 7910.

    ##4. What are the alternative and null hypotheses of the test?

        The null hypothesis (H0) states that the two independent samples have identical population means. In other words, there is no significant difference between the average heights of the two groups. The alternative hypothesis (H1) states that the population means are not equal (two-sided test).

    ##5. Is the result of the test significant?

        The independent two-sample t-test produced a t-statistic of 81.78, compared to a two-tailed t-critical value of 1.96 using significance level of α = 0.05, and a p-value < 0.001. Both the enormous t-statistic and the tiny p-value provide overwhelming statistical evidence for a difference between the groups. This indicates that the result is highly significant statistically, and the magnitude of the difference confirms it is also of strong practical importance.

    ##7. Would you reject the null hypothesis or not?

        Given the extremely small p-value and the t-statistic far exceeding the critical value, I reject the null hypothesis. 


    ##8. What are you conclusions about the research question?

        Group 1 and Group 2 have a substantial difference in mean heights (13.5 cm), which is both statistically significant and practically meaningful. Therefore, group membership is associated with a meaningful difference in average height.


    ##Figures:

    Figure 1: Distribution of heights for both groups, showing the substantial mean difference.
    {mo.image("height_distribution_task1.png")}
    Figure 2: Q-Q plots assessing normality assumption for both groups.
    {mo.image("qplots_task1.png")}
    """
    )
    return


@app.cell
def _():
    """
    #Bonus: We could use the Welsch test as well but since the variance and distribution is similar there's no need for that, as the above test will give us statistically better results, but:

    def task1_bonus():
        test = scipy.stats.ttest_ind(group1_heights, group2_heights,equal_var=False,alternative='two-sided')
        #Using version 1.11.0 in order to receive calculated degrees of freedom
        print("t =", test.statistic)
        print("p =", test.pvalue)
        print("df =", test.df)
        
        alpha = 0.05
        t_critical = scipy.stats.t.ppf(1 - alpha/2, test.df)
        print("Two-tailed t-critical:", t_critical)
        
        if abs(test.statistic) > t_critical:
            print("Reject H0")
        else:
            print("Fail to reject H0")

    task1_bonus()
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Fashion Design Tool

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
    mo.image("fashion_study.png")
    return


@app.cell
def _(pd, plt, scipy, sm, sns):
    # Task 2A -----------------------
    def task2a():
        # Load data
        df_f1 = pd.read_csv("fashion_study1.csv")

        # Intiial calculations
        google_scores = df_f1["Google Images"]
        gen_scores = df_f1["generative.fashion"]
        degrees = len(google_scores) - 1
        diff = gen_scores - google_scores
        mean_google, std_google = google_scores.mean(), google_scores.std(ddof=1)
        mean_gen, std_gen = gen_scores.mean(), gen_scores.std(ddof=1)
        mean_diff, std_diff = diff.mean(), diff.std(ddof=1)

        # Distribution plot (histogram + KDE)
        plt.figure(figsize=(10,6))
        sns.histplot(google_scores, kde=True, alpha=0.6, label=f'Google Images (mean={mean_google:.2f})')
        sns.histplot(gen_scores, kde=True, alpha=0.6, label=f'Generative Fashion (mean={mean_gen:.2f})')
        plt.xlabel('CSI score')
        plt.ylabel('Count')
        plt.title('Distribution of CSI scores')
        plt.legend()
        plt.show()

        # Histogram of differences
        plt.figure(figsize=(8,5))
        sns.histplot(diff, kde=True, alpha=0.6, color='green', stat='count')
        plt.axvline(mean_diff, color='red', linestyle='--', label=f'Mean diff: {mean_diff:.2f}')
        plt.xlabel('Difference (Gen - Google)')
        plt.ylabel('Count')
        plt.title('Histogram of differences')
        plt.legend()
        plt.savefig('score_distribution_task2a.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Q-Q plots
        fig, axes = plt.subplots(1, 3, figsize=(15,4))
        sm.qqplot(google_scores, loc=mean_google, scale=std_google, line='45', ax=axes[0])
        axes[0].set_title(f'Q-Q: Google Images\nmean={mean_google:.2f} sd={std_google:.2f}')

        sm.qqplot(gen_scores, loc=mean_gen, scale=std_gen, line='45', ax=axes[1])
        axes[1].set_title(f'Q-Q: Generative Fashion\nmean={mean_gen:.2f} sd={std_gen:.2f}')

        sm.qqplot(diff, loc=mean_diff, scale=std_diff, line='45', ax=axes[2])
        axes[2].set_title(f'Q-Q: Difference (Gen - Google)\nmean={mean_diff:.2f} sd={std_diff:.2f}')

        plt.tight_layout()
        plt.savefig('qplots_task2a.png', dpi=150, bbox_inches='tight')
        plt.show()

           # Paired line plot (bonus)
        plt.figure(figsize=(10,6))
        plt.plot(google_scores, marker='o', linestyle='', alpha=0.4, label='Google Images')
        plt.plot(gen_scores, marker='o', linestyle='', alpha=0.4, label='Generative Fashion')
        for i in range(len(google_scores)):
            plt.plot([i, i], [google_scores[i], gen_scores[i]], color='gray', alpha=0.3)
        plt.xlabel('Participant index')
        plt.ylabel('CSI score')
        plt.title('Paired differences per participant (line plot)')
        plt.legend()
        plt.savefig('pairedlineplot_task2a.png', dpi=150, bbox_inches='tight')
        plt.show()
    

        #Test
        test = scipy.stats.ttest_rel(google_scores, gen_scores, axis=0, nan_policy='propagate', alternative='two-sided', keepdims=False)
        print("t =", test.statistic)
        print("p =", test.pvalue)
        print("df =", test.df)

        alpha = 0.05
        t_critical = scipy.stats.t.ppf(1 - alpha/2, test.df)
        print("Two-tailed t-critical:", t_critical)

        if abs(test.statistic) > t_critical:
            print("Reject H0")
        else:
            print("Fail to reject H0")
    task2a()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    #Task 2A) Report on Fashion Design Tool Study 1
    ##1. How many people participated in the study?

    The first study had 48 participants.

    ##2. Which test do you choose and why?

        I chose a paired t-test because the measurements are paired, meaning the same participants provided scores for both conditions (Google Images and Generative Fashion). The data are continuous and the sample size is equal for both sets of scores.

        Before performing the test, it is important to check the distribution of the data. Both groups’ scores and the differences between them appear approximately normally distributed, based on Q–Q plots (adjusted with loc and scale for better visualisation).(Figure 2) The histograms of individual scores are roughly bell-shaped, and the histogram of differences is also approximately symmetric, which supports the normality assumption required for the paired t-test. (Figure 1)

    The paired t-test is designed to answer one critical question: "How likely is it to see a difference this large (or larger) in the scores between the two conditions for the same participants, if the null hypothesis is true and there is actually no difference in the population?"

    ##3. What are the degrees of freedom of the test?

        In a paired t-test, the degrees of freedom are based on the number of pairs - 1:  therefore there are 47 degrees of freedom

    ##4. What are the alternative and null hypotheses of the test?
    Null hypothesis (H₀): The mean difference between the paired scores (Generative Fashion − Google Images) is zero. In other words, there is no significant difference in Creativity Support Index (CSI) scores between the two conditions.

    Alternative hypothesis (H₁): The mean difference between the paired scores is not zero. That is, there is a significant difference in CSI scores between the two conditions. (I am performing a two-sided test, I am not specyfying direction. )

    ##5. Is the result of the test significant?

    Yes. Using a significance level of α = 0.05, the paired t-test produced a t-statistic of 81.78, compared to a two-tailed t-critical value of 2.012, and a p-value < 0.001. Both values provide overwhelming statistical evidence of a difference between the two conditions. The result is highly statistically significant.

    ##6. Would you reject the null hypothesis or not?
    Given the extremely small p-value and the t-statistic far exceeding the critical value, I reject the null hypothesis.  There is strong evidence that the mean CSI scores for Generative Fashion differ from Google Images.

    ##7. What are you conclusions about the research question?
    The analysis shows that the Generative Fashion tool (mean = 71.10) produces significantly higher Creativity Support Index scores than Google Images (mean = 64.10). The substantial difference in means indicates that the tool provides measurable benefits for participants’ ideation process. Therefore, the research question is answered: the Generative Fashion tool improves creativity support relative to Google Images.

    Figures:

    Figure 1:  Distribution of differences between two sample groups.
    {mo.image("score_distribution_task2a.png")}

    Figure 2: Q-Q plots assessing normality assumption for sample groups and their differences.
    {mo.image("qplots_task2a.png")}

    Figure 3 (Bonus): Paired Line Plot
    {mo.image("pairedlineplot_task2a.png")}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Fashion Design Tool (B)""")
    return


@app.cell
def _(pd, plt, scipy, sm, sns):
    # Task 2B -----------------------
    def task2b():
        # Load data
        df_f2 = pd.read_csv("fashion_study2.csv")
        df_f2.describe()
    
        # Intiial calculations
        stab_scores = df_f2["Stable Diffusion"]
        gen_scores = df_f2["generative.fashion"]
        degrees = len(stab_scores) - 1
        diff = gen_scores - stab_scores
        mean_stab, std_stab = stab_scores.mean(), stab_scores.std(ddof=1)
        mean_gen, std_gen = gen_scores.mean(), gen_scores.std(ddof=1)
        mean_diff, std_diff = diff.mean(), diff.std(ddof=1)

        # Distribution plot (histogram + KDE)
        plt.figure(figsize=(10,6))
        sns.histplot(stab_scores, kde=True, alpha=0.6, label=f'Stable Diffusion (mean={mean_stab:.2f})')
        sns.histplot(gen_scores, kde=True, alpha=0.6, label=f'Generative Fashion (mean={mean_gen:.2f})')
        plt.xlabel('CSI score')
        plt.ylabel('Count')
        plt.title('Distribution of CSI scores')
        plt.legend()
        plt.show()

        # Histogram of differences
        plt.figure(figsize=(8,5))
        sns.histplot(diff, kde=True, alpha=0.6, color='green', stat='count')
        plt.axvline(mean_diff, color='red', linestyle='--', label=f'Mean diff: {mean_diff:.2f}')
        plt.xlabel('Difference (Gen - Stable Diffusion)')
        plt.ylabel('Count')
        plt.title('Histogram of differences')
        plt.legend()
        plt.savefig('score_distribution_task2b.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Q-Q plots
        fig, axes = plt.subplots(1, 3, figsize=(15,4))
        sm.qqplot(stab_scores, loc=mean_stab, scale=std_stab, line='45', ax=axes[0])
        axes[0].set_title(f'Q-Q: Stable Diffusion \nmean={mean_stab:.2f} sd={std_stab:.2f}')

        sm.qqplot(gen_scores, loc=mean_gen, scale=std_gen, line='45', ax=axes[1])
        axes[1].set_title(f'Q-Q: Generative Fashion\nmean={mean_gen:.2f} sd={std_gen:.2f}')

        sm.qqplot(diff, loc=mean_diff, scale=std_diff, line='45', ax=axes[2])
        axes[2].set_title(f'Q-Q: Difference (Gen - Stable Diffusion)\nmean={mean_diff:.2f} sd={std_diff:.2f}')

        plt.tight_layout()
        plt.savefig('qplots_task2b.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Paired line plot (bonus)
        plt.figure(figsize=(10,6))
        plt.plot(stab_scores, marker='o', linestyle='', alpha=0.4, label='Stable Diffusion')
        plt.plot(gen_scores, marker='o', linestyle='', alpha=0.4, label='Generative Fashion')
        for i in range(len(stab_scores)):
            plt.plot([i, i], [stab_scores[i], gen_scores[i]], color='gray', alpha=0.3)
        plt.xlabel('Participant index')
        plt.ylabel('CSI score')
        plt.title('Paired differences per participant (line plot)')
        plt.legend()
        plt.savefig('pairedlineplot_task2b.png', dpi=150, bbox_inches='tight')
        plt.show()

        #Test
        test = scipy.stats.ttest_rel(stab_scores, gen_scores, axis=0, nan_policy='propagate', alternative='two-sided', keepdims=False)
        print("t =", test.statistic)
        print("p =", test.pvalue)
        print("df =", test.df)

        alpha = 0.05
        t_critical = scipy.stats.t.ppf(1 - alpha/2, test.df)
        print("Two-tailed t-critical:", t_critical)

        if abs(test.statistic) > t_critical:
            print("Reject H0")
        else:
            print("Fail to reject H0")
    task2b()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    #Task 2B) Report on Fashion Design Tool Study 2
    ##1. How many people participated in the study?

    The second study had 39 participants.

    ##2. Which test do you choose and why?

        I chose a paired t-test because the measurements are paired, meaning the same participants provided scores for both conditions (Stable Diffusion and Generative Fashion). The data are continuous and the sample size is equal for both sets of scores.

        Before performing the test, it is important to check the distribution of the data. Both groups’ scores and the differences between them appear approximately normally distributed, based on Q–Q plots (adjusted with loc and scale for better visualisation).(Figure 2) The histograms of individual scores are roughly bell-shaped, and the histogram of differences is also approximately symmetric, which supports the normality assumption required for the paired t-test. (Figure 1)

    The paired t-test is designed to answer one critical question: "How likely is it to see a difference this large (or larger) in the scores between the two conditions for the same participants, if the null hypothesis is true and there is actually no difference in the population?"

    ##3. What are the degrees of freedom of the test?

        In a paired t-test, the degrees of freedom are based on the number of pairs - 1:  therefore there are 38 degrees of freedom

    ##4. What are the alternative and null hypotheses of the test?
    Null hypothesis (H₀): The mean difference between the paired scores (Generative Fashion − Stable Diffusion) is zero. In other words, there is no significant difference in Creativity Support Index (CSI) scores between the two conditions.

    Alternative hypothesis (H₁): The mean difference between the paired scores is not zero. That is, there is a significant difference in CSI scores between the two conditions. (I am performing a two-sided test, I am not specyfying direction. )

    ##5. Is the result of the test significant?

    Yes. Using a significance level of α = 0.05, the paired t-test produced a t-statistic of 81.78, compared to a two-tailed t-critical value of 2.02, and a p-value < 0.001. Both values provide overwhelming statistical evidence of a difference between the two conditions. The result is highly statistically significant.

    ##6. Would you reject the null hypothesis or not?
    Given the extremely small p-value and the t-statistic far exceeding the critical value, I reject the null hypothesis.  There is strong evidence that the mean CSI scores for Generative Fashion differ from Stable Diffusion.

    ##7. What are you conclusions about the research question?
    The analysis shows that the Generative Fashion tool (mean = 72.66) produces significantly higher Creativity Support Index scores than Stable Diffusion (mean = 62.03). The substantial difference in means indicates that the tool provides measurable benefits for participants’ ideation process. Therefore, the research question is answered: the Generative Fashion tool improves creativity support relative to Stable Diffusion.

    Figures:

    Figure 1:  Distribution of differences between two sample groups.
    {mo.image("score_distribution_task2b.png")}

    Figure 2: Q-Q plots assessing normality assumption for sample groups and their differences.
    {mo.image("qplots_task2b.png")}


    Figure 3 (Bonus): Paired Line Plot
    {mo.image("pairedlineplot_task2b.png")}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Image editing software

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
def _(pd, plt, scipy, sm, sns):
    # Task 3 -----------------------

    def task3():

        df_e1 = pd.read_csv("imgedit_experts.csv")
        df_e2 = pd.read_csv("imgedit_nonexperts.csv")
        #df_e1.describe()
        #df_e2.describe()

        # Intiial calculations
        exp_time = df_e1["Experts"]
        nonexp_time = df_e2["Non-experts"]
        mean_exp_time, std_exp_time = exp_time.mean(), exp_time.std(ddof=1)
        mean_nonexp_time, std_nonexp_time = nonexp_time.mean(), nonexp_time.std(ddof=1)

        # Distribution plot 
        plt.figure(figsize=(10, 6))
    
        sns.histplot(data=exp_time, label='Experts', alpha=0.7, kde=True)
        sns.histplot(data=nonexp_time, label='Non-experts', alpha=0.7, kde=True)
    
        plt.axvline(mean_exp_time, color='blue', linestyle='--', alpha=0.8,
                    label=f'Experts Mean: {mean_exp_time:.2f}')
        plt.axvline(mean_nonexp_time, color='orange', linestyle='--', alpha=0.8,
                    label=f'Non-experts Mean: {mean_nonexp_time:.2f}')
    
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Distribution of Times: Experts vs Non-experts')
        plt.legend()
    
        plt.savefig('time_distribution_task3.png', dpi=150, bbox_inches='tight')
        plt.show()

    
        #Q-Q plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
        sm.qqplot(exp_time, loc=mean_exp_time, scale=std_exp_time, line='45', ax=ax1)
        ax1.set_title('Q-Q Plot: Group 1 Experts')
        ax1.set_xlabel('Theoretical Quantiles')
        ax1.set_ylabel('Sample Quantiles')
   
        sm.qqplot(nonexp_time, loc=mean_nonexp_time, scale=std_nonexp_time, line='45', ax=ax2)
        ax2.set_title('Q-Q Plot: Group 2 Non-Experts') 
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Sample Quantiles')
    
        plt.tight_layout()
        plt.savefig('qplots_task3.png', dpi=150, bbox_inches='tight')
        plt.show()

        #Test
        test = scipy.stats.ttest_ind(exp_time, nonexp_time,equal_var=False,alternative='two-sided')
        #Using version 1.11.0 in order to receive calculated degrees of freedom
        print("t =", test.statistic)
        print("p =", test.pvalue)
        print("df =", test.df)
    
        alpha = 0.05
        t_critical = scipy.stats.t.ppf(1 - alpha/2, test.df)
        print("Two-tailed t-critical:", t_critical)
    
        if abs(test.statistic) > t_critical:
            print("Reject H0")
        else:
            print("Fail to reject H0")
    
    task3()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    #Task 3) Report on Image Editing Software
    ##1. How many people participated in the study in the two groups?
    The first group consisted of 11 experts participants. The second group consisted of 16 non-experts participants.


    ##3. Which test do you choose and why?
    A two-sample t-test (independent t-test) would be the standard choice, however because the group sizes are unequal (11 vs 16), and the difference at standart deviation of each (6.38 vs 10.38) insinuated unequal variances, the safest option is Welch’s t-test.

    The outcome (time) is roughly normally distributed within each group and due to small size of samples slightly wobble tails are to be expected. 
    The observations are independent.


    ##4. What are the degrees of freedom of the test?

    The degrees of freedom of this test are 24.789549648155898

    ##5. What are the alternative and null hypotheses of the test?

    Null hypothesis (H₀): The mean task-completion time is the same for experts and non-experts. In other words, there is no significant difference in completion times between the two groups.

    Alternative hypothesis (H₁): The mean task-completion time differs between experts and non-experts. That is, there is a significant difference in completion times between the two groups.
    (I am performing a two-sided test and am not specifying a direction.)

    ##6. Is the result of the test significant?
    No. Using a significance level of α = 0.05, the Welch's t-test produced a t-statistic of −2.03, compared to a two-tailed t-critical value of 2.06, and a p-value of 0.053. Because the p-value is slightly above 0.05 and the test statistic does not exceed the critical threshold, there is insufficient statistical evidence to conclude that experts and non-experts differ in their task-completion times. The result is not statistically significant.

    ##7. Would you reject the null hypothesis or not?
    Given the p-value above the threshold and t-stat not reaching the critical threshold, I have failed to reject the null hypothesis. 

    ##8. What are you conclusions about the research question?
    The experts completed the task faster on average (44.07) than non-experts (50.63), but this difference did not reach statistical significance. Therefore, we cannot conclude that the software benefits experts more than non-experts based on this sample. The evidence is suggestive but not strong enough at the 5% significance level, likely due to the small number of experts.

    ##Figures:

    Figure 1: Distribution of times for both groups.
    {mo.image("time_distribution_task3.png")}

    Figure 2: 
    {mo.image("qplots_task3.png")}
    """
    )
    return


if __name__ == "__main__":
    app.run()
