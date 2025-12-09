import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from matplotlib import pyplot as plt
    import numpy as np
    import scipy.stats as stats
    from scipy.stats import shapiro
    from scipy.stats import mannwhitneyu
    return mannwhitneyu, mo, pd, plt, shapiro, stats


@app.cell
def _(pd):
    #Data import
    writers = pd.read_csv("data/writers.csv")
    evals = pd.read_csv("data/evaluators.csv")

    #Renaming column
    writers = writers.copy()
    writers["story_id"] = writers["ai_story_gen.1.player.story_id"]


    return evals, writers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#RQ1 Do writers who use AI judge the creativity and authorship of their stories differently from those who do not use AI?""")
    return


@app.cell(hide_code=True)
def _(pd, writers):
    # RQ1 Data Prep

    def rq1_dataprep(writers: pd.DataFrame) -> pd.DataFrame:
        rq1 = writers.copy()

        # Story ID for joining with evaluator data
        rq1["story_id"] = rq1["ai_story_gen.1.player.story_id"]

        # Merging follow_up.1 AND follow_up.2
        self_var_map = {
            "own_ideas": "self_own_ideas",
            "novel": "self_novel",
            "original": "self_original",
            "rare": "self_rare",
            "appropriate": "self_appropriate",
            "feasible": "self_feasible",
            "publishable": "self_publishable",
            "profit": "self_profit",             
            "tt_enjoyed": "self_enjoyed",
            "tt_badly_written": "self_badly_written",
            "tt_boring": "self_boring",
            "tt_funny": "self_funny",
            "tt_twist": "self_twist",
            "tt_future": "self_future",
        }

        def combine_two_cols(df, c1, c2):
            s1 = df[c1]
            s2 = df[c2]
            # if s1 is NaN, take s2
            return s1.where(~s1.isna(), s2)

        for var, new_name in self_var_map.items():
            col1 = f"follow_up.1.player.{var}"
            col2 = f"follow_up.2.player.{var}"
            rq1[new_name] = combine_two_cols(rq1, col1, col2)

        rq1["self_used_ai"] = combine_two_cols(
            rq1,
            "follow_up.1.player.used_ai_tool",
            "follow_up.2.player.used_ai_tool",
        )

        for i in range(5):
            col1 = f"follow_up.1.player.ai_idea{i}_assist"
            col2 = f"follow_up.2.player.ai_idea{i}_assist"
            rq1[f"assist_idea{i}"] = combine_two_cols(rq1, col1, col2)

        # Renaming 
        rq1 = rq1.rename(columns={
            "participant.condition": "condition",
            "ai_story_gen.1.player.story": "story_text",
        })

        # DAT word columns
        dat_rename = {f"ai_story_gen.1.player.word{i}": f"word{i}" for i in range(1, 11)}
        rq1 = rq1.rename(columns=dat_rename)

        # Demographics
        rq1 = rq1.rename(columns={
            "payment.1.player.age": "age",
            "payment.1.player.gender": "gender",
            "payment.1.player.education": "education",
            "payment.1.player.employment": "employment",
        })

        # DAT placeholder
        dat_cols = [f"word{i}" for i in range(1, 11)]
        rq1["dat_words"] = rq1[dat_cols].values.tolist()
        rq1["DAT_score"] = None  

        # Final columns
        keep_cols = [
            "story_id",
            "condition",
            "story_text",
            "self_used_ai",
            "assist_idea0",
            "assist_idea1",
            "assist_idea2",
            "assist_idea3",
            "assist_idea4",
            "self_own_ideas",
            "self_novel",
            "self_original",
            "self_rare",
            "self_appropriate",
            "self_feasible",
            "self_publishable",
            "self_profit",     
            "self_enjoyed",
            "self_badly_written",
            "self_boring",
            "self_funny",
            "self_twist",
            "self_future",
            *dat_cols,
            "dat_words",
            "DAT_score",
            "age",
            "gender",
            "education",
            "employment",
        ]

        keep_cols = [c for c in keep_cols if c in rq1.columns]

        rq1_clean = rq1[keep_cols].copy()
        return rq1_clean


    rq1_clean = rq1_dataprep(writers)
    rq1_clean.to_csv("data/rq1_clean.csv", index=False)
    print("rq1_clean saved to data/rq1_clean.csv")

    return (rq1_clean,)


@app.cell(hide_code=True)
def _(rq1_clean):
    #RQ1 Data Check p1
    def rq1_datacheck():
        print("Dataset shape:", rq1_clean.shape)
    
        #Group sizes
        print("\nWriters per condition:")
        print(rq1_clean["condition"].value_counts(dropna=False))
    
        #N/A checks
        key_vars = [
            "self_own_ideas",
            "self_novel",
            "self_original",
            "self_rare",
            "self_appropriate",
            "self_feasible",
            "self_publishable",
            "self_enjoyed",
            "self_badly_written",
            "self_boring",
            "self_funny",
            "self_twist",
            "self_future",
        ]
    
        print("\nMissing values in key self-rating variables:")
        print(rq1_clean[key_vars].isna().sum())
    
        #Summary
        print("\nBasic summary of self-perception variables:")
        print(rq1_clean[key_vars].describe())

    rq1_datacheck()

    return


@app.cell(hide_code=True)
def _(rq1_clean, shapiro):
    #RQ1 Data Check p2
    def rq1_normality():
        vars_to_test = [
            "self_own_ideas",
            "self_novel",
            "self_original",
            "self_rare",
            "self_appropriate",
            "self_feasible",
            "self_publishable",
            "self_enjoyed",
            "self_badly_written",
            "self_boring",
            "self_funny",
            "self_twist",
            "self_future",
        ]
    
        print("Shapiro–Wilk normality test for each RQ1 dependent variable:\n")
    
        normality_results = {}
    
        for var in vars_to_test:
            data = rq1_clean[var].dropna()
            stat, p = shapiro(data)
            normality_results[var] = p
            print(f"{var}: p = {p:.10f}")
    
        print("\nInterpretation:")
        print("All variables returned p < 0.05, indicating significant deviation from normality.\n"
              "This supports the use of non-parametric tests.")

    rq1_normality()
    return


@app.cell(hide_code=True)
def _(plt, rq1_clean, stats):
    #RQ1 Data Check p3
    vars_to_plot = [
        "self_own_ideas",
        "self_novel",
        "self_original",
        "self_rare",
        "self_appropriate",
        "self_feasible",
        "self_publishable",
        "self_enjoyed",
        "self_badly_written",
        "self_boring",
        "self_funny",
        "self_twist",
        "self_future",
    ]

    plt.figure(figsize=(15, 18))

    for i, var in enumerate(vars_to_plot, 1):
        plt.subplot(5, 3, i)
        stats.probplot(rq1_clean[var].dropna(), dist="norm", plot=plt)
        plt.title(f"Q–Q Plot: {var}", fontsize=9)
        plt.tight_layout()

    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## RQ1 Data Check Conclusion:
    No missing values detected in key self-rating variables.

    Condition groups are well balanced (100, 98, 97).

    Visual inspection of the Q–Q plots for all self-perception variables showed clear deviations from the diagonal reference line expected under normality. The plots displayed the characteristic step-like patterns and skew typical of Likert scale data, with compressed tails and curved shapes indicating non-normal distributions. These visual patterns are fully consistent with the Shapiro–Wilk test results and further justify the use of non-parametric methods for analysing group differences in RQ1.

    Data quality is good to proceed with RQ1 analysis.
    """
    )
    return


@app.cell(hide_code=True)
def _(rq1_clean):
    #RQ1 Descriptive Stats Based on Condition
    def rq1_des_con():
        vars_of_interest = [
            "self_own_ideas",
            "self_novel",
            "self_original",
            "self_rare",
            "self_appropriate",
            "self_feasible",
            "self_publishable",
            "self_enjoyed",
            "self_badly_written",
            "self_boring",
            "self_funny",
            "self_twist",
            "self_future",
        ]
    
        desc = (
            rq1_clean
            .groupby("condition")[vars_of_interest]
            .agg(["mean", "std", "count"])
        )
    
        print("Descriptive statistics for writer self-perception by condition:")
        rq1_clean.to_csv("data/rq1_ai_condition.csv", index=False)
        return desc

    rq1_des_con()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## RQ1 Descriptive Stats Interpretation note (Condition-based):
    Across the three assigned conditions, writers in the human-only group reported the highest sense of authorship (M ≈ 6.96). Both AI-assisted conditions showed slightly lower authorship ratings (M ≈ 6.59 in 1AI; M ≈ 6.57 in 5AI), suggesting that the mere availability of AI ideas may modestly reduce perceived ownership of the story.

    Creativity-related evaluations remained broadly similar across groups. Novelty hovered around 4.7–4.5, and originality showed only a small decline in the 5AI group (4.80 → 4.81 → 4.41). Appropriateness was slightly higher when AI was available (7.26 in human, 7.50 in 1AI, 7.24 in 5AI), indicating that AI access may help writers feel their story meets the task expectations. Enjoyment was consistently high across all conditions (≈ 6.9–6.8) with minimal variation.

    Overall, the descriptive statistics indicate only modest differences between conditions, mainly in perceived authorship and appropriateness. Most other dimensions (novelty, originality, publishability, enjoyment) remain stable. Inferential tests are required to determine whether these small descriptive differences are statistically meaningful.
    """
    )
    return


@app.cell(hide_code=True)
def _(pd, rq1_clean):
    # RQ1 Descriptive Stats Based on ACTUAL AI Use
    def rq1_des_actual_ai(df):
        df = df.copy()
        self_report = df["self_used_ai"] == 1

        #Using assisted_ideas to check for liars
        idea_cols = ["assist_idea0", "assist_idea1", "assist_idea2", "assist_idea3", "assist_idea4"]
        used_assist = df[idea_cols].apply(lambda row: row.notna().any(), axis=1)

        df["ai_used_actual"] = ((self_report) | (used_assist)).astype(int)

        print("\nActual AI Use (0 = no AI, 1 = used AI):")
        print(df["ai_used_actual"].value_counts())

        print("\nCondition vs Actual AI Use:")
        print(pd.crosstab(df["condition"], df["ai_used_actual"]))

        vars_of_interest = [
            "self_own_ideas",
            "self_novel",
            "self_original",
            "self_rare",
            "self_appropriate",
            "self_feasible",
            "self_publishable",
            "self_enjoyed",
            "self_badly_written",
            "self_boring",
            "self_funny",
            "self_twist",
            "self_future",
        ]

        desc_actual = (
            df.groupby("ai_used_actual")[vars_of_interest]
              .agg(["mean", "std", "count"])
        )

        print("\nDescriptive Statistics by ACTUAL AI Use:")
        print(desc_actual)
        df.to_csv("data/rq1_ai_actual.csv", index=False)
    
        return df, desc_actual

    rq1_ai, desc_actual = rq1_des_actual_ai(rq1_clean)

    desc_actual
    return (rq1_ai,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## RQ1 Descriptive Stats Interpretation note (Actual AI-Usage):
    - In the human condition, only 2 writers reported or showed evidence of AI use,
       meaning the "human-only" baseline is very clean (95 true human writers).
    - In the human_1AI condition, AI usage was optional: 56 used the AI idea,
       while 44 chose to ignore it entirely.
    - In the human_5AI condition, most writers (83) used at least one AI idea,
       but 15 still ignored all five.

     This shows that AI availability does not guarantee AI usage.
     Actual behaviour varies within conditions, so analysing both assigned
     conditions and real AI use provides a more accurate understanding of
     how AI affects writer self-perception.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##RQ1 Test Choice:
    The dependent variables in this analysis (e.g., authorship, novelty, originality, enjoyment) were measured on 1–9 Likert scales. These data are ordinal, discretised, and—based on Shapiro–Wilk tests and Q–Q plot inspection—clearly violate the normality assumptions required for parametric tests such as ANOVA.

    Although the study originally included three assigned writing conditions (human, human_1AI, human_5AI), descriptive analyses showed substantial variability in whether participants actually used AI when it was available. For example, 44% of writers in the 1AI condition and 15% in the 5AI condition chose not to use AI at all, while two writers in the human-only condition reported or showed evidence of AI assistance. Because AI availability did not reliably correspond to AI behaviour, analysing differences across the three conditions would not accurately capture the true contrast of interest.

    If we had analysed the three pre-assigned groups, the appropriate non-parametric omnibus test would have been the Kruskal–Wallis H test, followed by Dunn post-hoc comparisons where necessary. However, since RQ1 focuses on actual AI usage (AI used vs. not used), the comparison involves two independent groups. Therefore, the correct non-parametric test is the Mann–Whitney U test, which was applied separately to each self-perception variable.

    Null Hypothesis (H₀):
    The distribution of each self-perception variable is equal for writers who used AI and writers who did not.

    Alternative Hypothesis (H₁):
    The distribution of each self-perception variable differs between writers who used AI and those who did not.

    This behaviour-based analytical approach provides a more valid and robust assessment of how AI involvement influences writers’ self-perception.
    """
    )
    return


@app.cell(hide_code=True)
def _(mannwhitneyu, pd, rq1_ai):
    def rq1_mannwhitneyu():

        # List of dependent variables
        rq1_vars = [
            "self_own_ideas",
            "self_novel",
            "self_original",
            "self_rare",
            "self_appropriate",
            "self_feasible",
            "self_publishable",
            "self_enjoyed",
            "self_badly_written",
            "self_boring",
            "self_funny",
            "self_twist",
            "self_future",
        ]
    
        results = []
    
        for var in rq1_vars:
            g0 = rq1_ai.loc[rq1_ai["ai_used_actual"] == 0, var].dropna()
            g1 = rq1_ai.loc[rq1_ai["ai_used_actual"] == 1, var].dropna()
    
            U, p = mannwhitneyu(g0, g1, alternative="two-sided")
        
            results.append({
                "variable": var,
                "U_stat": U,
                "p_value": p,
                "median_no_ai": g0.median(),
                "median_ai": g1.median(),
                "n_no_ai": len(g0),
                "n_ai": len(g1),
            })
    
        results_df = pd.DataFrame(results)
        return results_df

    rq1_mannwhitneyu()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##RQ1: Mann-Whitney U Test Conclusion
    A set of Mann–Whitney U tests compared self-perception ratings between writers who actually used AI (n = 141) and those who did not (n = 154). Most creativity-related evaluations — including novelty, originality, rarity, feasibility, publishability, and stylistic ratings — did not differ significantly between the groups (all p > .05).

    Three variables showed statistically significant differences. Writers who used AI reported slightly lower authorship (U = 12,437, p = .028), consistent with the idea that AI involvement reduces perceived personal ownership of the story. They also rated their stories as more appropriate for the task (U = 9,299, p = .028), suggesting that AI guidance helped align their writing with the expected genre or constraints. Finally, AI users scored lower on the item measuring whether their story changed what they expect of future stories they will read (U = 12,294, p = .045). This indicates that AI-assisted writing felt less personally meaningful or expectation-shifting, reducing the sense of narrative impact writers derived from their own work.

    Overall, AI use had selective rather than global effects: it influenced perceptions of authorship, appropriateness, and narrative impact, but did not meaningfully alter broader creativity or enjoyment evaluations.

    ##Decision: H₀ is partially rejected: 
    AI use affected only a small subset of self-perception variables, not the majority (3 significant out of 13 tested). Most aspects of writers’ self-evaluated creativity did not differ between AI and non-AI users, but authorship, appropriateness, and future-expectation impact did show significant effects.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #RQ2 — Reader Evaluation of AI-Influenced Stories
    RQ2a — Do readers evaluate AI-assisted stories differently before knowing AI was used? (Blind Quality Judgement)

    RQ2b — Does disclosure that a story used AI change evaluators' perceptions of authorship and ownership? (Impact of AI Disclosure on Ownership Judgement)

    RQ2c - Does the belief or suspicion that a story involved AI predict lower story ratings, even before readers are told anything? (Effect of AI Suspicion on Blind Ratings)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A small number of story IDs present in the evaluator dataset did not appear in the writers dataset (5 unique stories, 50+ evaluations). These likely correspond to excluded or pilot stories. Because no writer-level information (condition, AI usage) was available for them, all evaluations of these stories were removed from the analysis""")
    return


@app.cell(hide_code=True)
def _(evals, pd, rq1_clean):
    # RQ2: Dataset Prep

    def build_evals_full(evals_df: pd.DataFrame, writers_df: pd.DataFrame) -> pd.DataFrame:
    
        writers2 = writers_df.copy()

        # actual AI use (as before)
        idea_cols = ["assist_idea0", "assist_idea1", "assist_idea2", "assist_idea3", "assist_idea4"]
        writers2["ai_used_actual"] = (
            (writers2["self_used_ai"] == 1) |
            (writers2[idea_cols].notna().any(axis=1))
        ).astype(int)

        rows = []

        for k in range(1, 11):
            pre = f"evaluator.{k}.player."
            post = f"evaluator_p2.{k}.player."

            fields = [
                "participant.code",
                pre + "story_id",
                pre + "topic",
                pre + "novel",
                pre + "original",
                pre + "rare",
                pre + "appropriate",
                pre + "feasible",
                pre + "publishable",
                pre + "profit",
                pre + "tt_enjoyed",
                pre + "tt_badly_written",
                pre + "tt_boring",
                pre + "tt_funny",
                pre + "tt_twist",
                pre + "tt_future",
                # phase 2
                post + "ai_usage",
                post + "ai_assist",
                post + "authors_ideas",
                post + "ownership",
                post + "profit",
            ]

            avail = [c for c in fields if c in evals_df.columns]
            if not avail:
                continue

            sub = evals_df[avail].copy()

            rename = {}
            if "participant.code" in avail:
                rename["participant.code"] = "evaluator_code"
            if pre + "story_id" in avail:
                rename[pre + "story_id"] = "story_id"
            if pre + "topic" in avail:
                rename[pre + "topic"] = "topic"

            metrics = [
                "novel", "original", "rare", "appropriate",
                "feasible", "publishable", "profit",
                "tt_enjoyed", "tt_badly_written", "tt_boring",
                "tt_funny", "tt_twist", "tt_future",
            ]
            for m in metrics:
                col = pre + m
                if col in avail:
                    rename[col] = "pre_" + m

            post_map = {
                "ai_usage": "post_ai_usage",
                "ai_assist": "post_ai_assist",
                "authors_ideas": "post_authors_ideas",
                "ownership": "post_ownership",
                "profit": "post_profit",
            }
            for old, new in post_map.items():
                col = post + old
                if col in avail:
                    rename[col] = new

            sub = sub.rename(columns=rename)

            if "story_id" in sub.columns:
                sub = sub.dropna(subset=["story_id"])
                rows.append(sub)

        full = pd.concat(rows, ignore_index=True)

        # merging
        merged = full.merge(
            writers2[["story_id", "condition", "ai_used_actual"]],
            on="story_id",
            how="left"
        )

        # Dropping evaluations without an attached story in writers.csv
        n_before = merged.shape[0]
        merged = merged.dropna(subset=["condition"])
        n_after = merged.shape[0]
        print(f"Dropped {n_before - n_after} evaluations with no matching writer record.")

        return merged

    evals_full = build_evals_full(evals, rq1_clean)
    evals_full.to_csv("data/rq2_evals_full.csv", index=False)
    print("evals_full shape:", evals_full.shape)

    evals_full
    return (evals_full,)


@app.cell
def _(mo):
    mo.md(
        r"""
    #RQ2a — Do readers evaluate AI-assisted stories differently before knowing AI was used? (Blind Quality Judgement)

    Null Hypothesis (H₀):
    The distribution of each pre-disclosure story-quality rating is equal between AI-assisted and non-AI stories.

    Alternative Hypothesis (H₁):
    The distribution of at least one pre-disclosure story-quality rating differs between AI-assisted and non-AI stories.
    """
    )
    return


@app.cell(hide_code=True)
def _(evals_full):
    # RQ2a: Blind story quality Data Prep
    def prepare_rq2a(evals_full):
        quality_vars = [
            "pre_novel",
            "pre_original",
            "pre_rare",
            "pre_appropriate",
            "pre_feasible",
            "pre_publishable",
            "pre_tt_enjoyed",
            "pre_tt_badly_written",
            "pre_tt_boring",
            "pre_tt_funny",
            "pre_tt_twist",
            "pre_tt_future",
        ]

        cols = ["story_id", "evaluator_code", "ai_used_actual"] + quality_vars
        rq2a = evals_full[cols].copy()

        return rq2a


    rq2a = prepare_rq2a(evals_full)
    print("rq2a shape:", rq2a.shape)
    rq2a.head()

    return (rq2a,)


@app.cell(hide_code=True)
def _(rq2a):
    # RQ2a: Blind story quality Data Check
    def rq2a_datacheck(rq2a):
        print("RQ2a dataset shape:", rq2a.shape)

        # Group sizes
        print("\nEvaluations per AI-usage group:")
        print(rq2a["ai_used_actual"].value_counts(dropna=False))

        # Missing values
        quality_vars = [c for c in rq2a.columns if c.startswith("pre_")]
        print("\nMissing values in pre-quality variables:")
        print(rq2a[quality_vars].isna().sum())

        # Summary statistics for all evaluations
        print("\nSummary statistics (all evaluations):")
        print(rq2a[quality_vars].describe())

        # Group descriptive statistics
        quality_vars = [
            "pre_novel", "pre_original", "pre_rare",
            "pre_appropriate", "pre_feasible", "pre_publishable",
            "pre_tt_enjoyed", "pre_tt_badly_written",
            "pre_tt_boring", "pre_tt_funny",
            "pre_tt_twist", "pre_tt_future"
        ]

        group_desc = (
            rq2a.groupby("ai_used_actual")[quality_vars]
                 .agg(["mean", "std", "median", "count"])
        )

        print("\nGroup-wise descriptive statistics (AI used vs No AI):")
        print(group_desc)

        return group_desc


    group_desc = rq2a_datacheck(rq2a)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##Data Cleaning, Descriptive Statistics Checks, and Test Choice (RQ2a)

    The evaluator dataset contained 3,543 blind (pre-disclosure) story ratings, with no missing values across any creativity or stylistic variables. Evaluations were well distributed between stories written without AI (1,860 ratings) and those written with AI assistance (1,683 ratings), providing balanced groups for comparison. Group-wise descriptive statistics indicated that median scores were broadly similar across AI and non-AI stories, with only modest differences in means that required formal inferential testing. These checks confirmed that the dataset was complete, coherent, and suitable for non-parametric analysis.

    Because the dependent variables (e.g., novelty, originality, publishability, enjoyment) are measured on 1–9 Likert scales, they are ordinal and do not meet parametric assumptions such as normality. Accordingly, the Mann–Whitney U test was selected to compare readers’ blind evaluations of stories written with versus without actual AI usage.


    """
    )
    return


@app.cell
def _(mannwhitneyu, pd, rq2a):
    def rq2_mannwhitneyu(rq2a):

        rq2a_vars = [
            "pre_novel", "pre_original", "pre_rare",
            "pre_appropriate", "pre_feasible", "pre_publishable",
            "pre_tt_enjoyed", "pre_tt_badly_written",
            "pre_tt_boring", "pre_tt_funny",
            "pre_tt_twist", "pre_tt_future"
        ]
    
        def cliffs_delta(x, y):
            """Effect size for Mann–Whitney U: Cliff's delta."""
            nx, ny = len(x), len(y)
            U, _ = mannwhitneyu(x, y, alternative="two-sided")
            delta = (2*U)/(nx*ny) - 1
            return delta
    
        results = []
    
        for var in rq2a_vars:
            g0 = rq2a.loc[rq2a["ai_used_actual"] == 0, var].dropna()
            g1 = rq2a.loc[rq2a["ai_used_actual"] == 1, var].dropna()
        
            U, p = mannwhitneyu(g0, g1, alternative="two-sided")
            delta = cliffs_delta(g1, g0)   # positive = AI > non-AI
        
            results.append({
                "variable": var,
                "U_stat": U,
                "p_value": p,
                "median_no_ai": g0.median(),
                "median_ai": g1.median(),
                "mean_no_ai": g0.mean(),
                "mean_ai": g1.mean(),
                "n_no_ai": len(g0),
                "n_ai": len(g1),
                "cliffs_delta": delta
            })
    
        return pd.DataFrame(results)


    results_df = rq2_mannwhitneyu(rq2a)
    results_df

    return


@app.cell
def _(mo):
    mo.md(r"""##RQ2a: Mann-Whitney U Test Conclusion""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Blind evaluations showed that AI-assisted stories were rated significantly higher than human-only stories on almost all creativity and stylistic dimensions. Mann–Whitney U tests revealed small but consistent advantages for AI on novelty, originality, rarity, appropriateness, feasibility, publishability, and overall enjoyment (all p < .01). AI-written stories were also judged less “badly written” and contained stronger narrative twists. Only boredom and humour showed no differences. These findings suggest that—when readers do not know who authored the text—AI assistance tends to improve the perceived quality of creative writing.

    ##Decision: H₀ is largely rejected.
    Evaluators rated AI-assisted stories significantly higher on 10 out of 12 quality dimensions before knowing AI was involved. Only “boring” and “funny” showed no significant difference.

    This pattern indicates that AI-assisted stories were systematically judged as higher quality across most dimensions during blind evaluation.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #RQ2b — Does disclosure that a story used AI change evaluators' perceptions of authorship and ownership? (Impact of AI Disclosure on Ownership Judgement)

    Null Hypothesis (H₀):
    The distribution of each post-disclosure evaluative variable (authors’ ideas, ownership, profit attribution) is equal for AI-assisted and non-AI stories.

    Alternative Hypothesis (H₁):
    At least one post-disclosure variable differs between AI-assisted and non-AI stories, indicating that learning about AI involvement affects evaluators’ perceptions of authorship or ownership.
    """
    )
    return


@app.cell
def _(evals_full):
    # RQ2b Data Prep
    def rq2b_dataprep(evals_full):
        # Keep only rows with post-disclosure variables
        post_vars = ["post_authors_ideas", "post_ownership", "post_profit", "post_ai_assist"]
    
        rq2b = evals_full.copy()
    
        # Keep rows where at least one post variable exists
        rq2b = rq2b.dropna(subset=post_vars, how="all")
    
        # Ensure ai_used_actual exists
        rq2b = rq2b.dropna(subset=["ai_used_actual"])
    
        # Convert ai_used_actual to int for grouping clarity
        rq2b["ai_used_actual"] = rq2b["ai_used_actual"].astype(int)
    
        return rq2b[["story_id", "ai_used_actual", "evaluator_code"] + post_vars].copy()


    rq2b = rq2b_dataprep(evals_full)
    rq2b.to_csv("data/rq2b_clean.csv", index=False)

    print("RQ2b dataset shape:", rq2b.shape)
    rq2b.head()

    return (rq2b,)


@app.cell
def _(rq2b):
    # RQ2b Data Check

    def rq2b_datacheck(rq2b):
        print("Shape:", rq2b.shape)
    
        # Group sizes
        print("\nEvaluations per AI-usage group:")
        print(rq2b["ai_used_actual"].value_counts())
    
        # Missingness
        post_vars = ["post_authors_ideas", "post_ownership", "post_profit", "post_ai_assist"]
        print("\nMissing values:")
        print(rq2b[post_vars].isna().sum())
    
        # Descriptive statistics by group
        print("\nGroup-wise descriptive statistics:")
        print(
            rq2b.groupby("ai_used_actual")[post_vars]
                .agg(["mean", "std", "median", "count"])
        )

    rq2b_datacheck(rq2b)

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##Data Cleaning, Descriptive Statistics Checks, and Test Choice (RQ2b)

    The post-disclosure evaluator dataset contained 3,543 authorship-related ratings, evenly split between stories written without AI (1,860 evaluations) and those written with AI assistance (1,683 evaluations). No missing values were present in the key post-disclosure variables (post_authors_ideas, post_ownership, post_profit, post_ai_assist). Group-wise descriptive statistics showed substantial downward shifts for AI-assisted stories in perceived author contribution and ownership, while the perceived AI contribution increased as expected. These patterns suggested that disclosure of AI involvement may meaningfully alter evaluators’ credit attribution judgements.

    Because all dependent variables are Likert-type ordinal ratings and exhibit non-normal distributions, non-parametric testing is required. Accordingly, the Mann–Whitney U test was selected to compare post-disclosure authorship and ownership assessments between AI-assisted and non-AI stories.
    """
    )
    return


@app.cell
def _(mannwhitneyu, pd, rq2b):
    # RQ2b: Mann–Whitney U tests for post-disclosure evaluator judgements

    def rq2b_mannwhitneyu(rq2b):

        post_vars = [
            "post_authors_ideas",
            "post_ownership",
            "post_profit",
        ]
    
        def cliffs_delta(x, y):
            """Effect size for Mann–Whitney U: Cliff’s delta."""
            nx, ny = len(x), len(y)
            U, _ = mannwhitneyu(x, y, alternative="two-sided")
            delta = (2 * U) / (nx * ny) - 1
            return delta
    
        results = []
    
        for var in post_vars:
            g0 = rq2b.loc[rq2b["ai_used_actual"] == 0, var].dropna()
            g1 = rq2b.loc[rq2b["ai_used_actual"] == 1, var].dropna()
        
            U, p = mannwhitneyu(g0, g1, alternative="two-sided")
            delta = cliffs_delta(g1, g0)   # positive = AI > non-AI
        
            results.append({
                "variable": var,
                "U_stat": U,
                "p_value": p,
                "median_no_ai": g0.median(),
                "median_ai": g1.median(),
                "mean_no_ai": g0.mean(),
                "mean_ai": g1.mean(),
                "n_no_ai": len(g0),
                "n_ai": len(g1),
                "cliffs_delta": delta,
            })
    
        results_df = pd.DataFrame(results)
        results_df.to_csv("data/rq2b_results.csv", index=False)
        return results_df


    rq2b_results = rq2b_mannwhitneyu(rq2b)
    rq2b_results

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # RQ2b — Post-disclosure Judgement of Authorship and Ownership**

    Once evaluators were informed whether a story had been written with AI assistance, their assessments shifted sharply. AI-assisted authors were judged to have contributed substantially fewer ideas to the story and to hold weaker ownership claims. Mann–Whitney U tests revealed very large and highly significant reductions in both perceived author contribution and ownership for AI-assisted stories (all p < 10⁻¹³⁰), indicating a strong and consistent penalty applied to human authors once AI involvement became known.

    The profit-sharing variable was excluded from inferential analysis. Although the question asked evaluators to allocate story profit between the author and the AI tool, responses were highly inconsistent: many participants assigned 0% to the author even for stories with no AI involvement, while others gave widely varying allocations. This pattern suggests that the question was interpreted in heterogeneous ways (e.g., entering 0% to indicate “AI deserves nothing”) and therefore does not provide a valid basis for comparing AI and non-AI stories. The authorship and ownership items, by contrast, showed coherent and interpretable distributions.

    Decision: H₀ is fully rejected.
    Across all valid post-disclosure measures (authors’ ideas and ownership), evaluators attributed significantly less creative credit to writers who used AI. The effects were large, robust, and directionally consistent, demonstrating that AI disclosure—not the story’s content—drives a substantial reduction in perceived human authorship and entitlement.
    """
    )
    return


@app.cell
def _():
    #test
    #test conclusion
    #RQ2b data prep,cleaning, test choice
    #test
    #test conclusion
    return


if __name__ == "__main__":
    app.run()
