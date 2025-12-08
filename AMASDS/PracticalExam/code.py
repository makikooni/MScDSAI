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
              "This supports the use of non-parametric tests (Kruskal–Wallis + Dunn post-hoc).")

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
    The dependent variables in this analysis (e.g., authorship, novelty, originality) were measured on 1–9 Likert scales. These ratings are ordinal and do not meet the assumptions of normality required for parametric tests such as ANOVA.
    Because there are three independent writing conditions (human, human_1AI, human_5AI) and multiple ordinal ourtcome variables. the appropriate non-parametric alternative is the Kruskal–Wallis H test. This test determines whether the distributions of the self-perception scores differ across the three groups, without assuming normality or equal variances.
    When the Kruskal–Wallis test indicates a significant effect, Dunn post-hoc tests with Bonferroni correction are used to identify which specific pairs of conditions differ.

    Because writer self-perception was assessed through multiple Likert-scale items capturing different aspects of creativity and authorship, I conducted separate Kruskal–Wallis tests for each dependent variable. These tests examine whether at least one experimental condition differs from another. The global hypothesis was that AI exposure would affect at least one dimension of writer self-perception.

    Null Hypothesis (H₀): The distributions of each self-perception variable are equal for human, human_1AI, and human_5AI groups.

    Alternative Hypothesis (H₁): At least one condition differs from the others in the distribution of the self-perception variable.
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

    Three variables showed statistically significant differences. Writers who used AI reported slightly lower authorship (U = 12437, p = .028), consistent with the idea that AI involvement reduces perceived ownership. They also rated their stories as more appropriate for the task (U = 9299, p = .028), suggesting AI guidance helped align their writing with the expected genre or constraints. Finally, AI users scored lower on the item measuring whether their story changed what they expect of future stories they will read (U = 12,294, p = .045). This suggests that AI-assisted writing felt less personally meaningful or expectation-shifting, reducing the sense of narrative impact that writers derived from their own work.

    Overall, AI use had selective rather than global effects: it influenced perceptions of authorship and appropriateness but did not meaningfully alter broader creativity or enjoyment evaluations.
    """
    )
    return


@app.cell
def _(evals, pd, writers):
    #Building long format table
    #One row = one evaluation of story (pre/post)
    def build_evals_full(evals_df: pd.DataFrame, writers_df: pd.DataFrame) -> pd.DataFrame:
        writers2 = writers_df.copy()

        rows = []

        # each evaluator row can have up to 10 stories: evaluator.1 ... evaluator.10
        for k in range(1, 11):
            pre = f"evaluator.{k}.player."
            post = f"evaluator_p2.{k}.player."

            candidates = [
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
                # phase 2 (post-disclosure) columns
                post + "condition",
                post + "ai_usage",
                post + "ai_assist",
                post + "authors_ideas",
                post + "ownership",
                post + "profit",
            ]

            # keep only columns that actually exist in this export
            avail = [c for c in candidates if c in evals_df.columns]
            if not avail:
                continue

            sub = evals_df[avail].copy()

            # rename to clean names: story_id, evaluator_code, pre_*, post_*
            rename_map = {}

            if pre + "story_id" in avail:
                rename_map[pre + "story_id"] = "story_id"
            if "participant.code" in avail:
                rename_map["participant.code"] = "evaluator_code"
            if pre + "topic" in avail:
                rename_map[pre + "topic"] = "topic"

            # phase 1 metrics
            for name in [
                "novel",
                "original",
                "rare",
                "appropriate",
                "feasible",
                "publishable",
                "profit",
                "tt_enjoyed",
                "tt_badly_written",
                "tt_boring",
                "tt_funny",
                "tt_twist",
                "tt_future",
            ]:
                col = pre + name
                if col in avail:
                    rename_map[col] = "pre_" + name

            #phase 2 metrics
            post_map = {
                "condition": "post_condition",
                "ai_usage": "post_ai_usage",
                "ai_assist": "post_ai_assist",
                "authors_ideas": "post_authors_ideas",
                "ownership": "post_ownership",
                "profit": "post_profit",
            }
            for key, new in post_map.items():
                col = post + key
                if col in avail:
                    rename_map[col] = new

            sub = sub.rename(columns=rename_map)

            # keep only rows that have an actual story_id for this slot
            if "story_id" in sub.columns:
                sub = sub.dropna(subset=["story_id"])
                rows.append(sub)

        # long table
        full = pd.concat(rows, ignore_index=True)

        # merge writer info on story_id
        merged = full.merge(writers2, on="story_id", how="left")

        return merged


    evals_full = build_evals_full(evals, writers)

    print("evals_full shape:", evals_full.shape)
    print(evals_full.iloc[0, :25])

    return


if __name__ == "__main__":
    app.run()
