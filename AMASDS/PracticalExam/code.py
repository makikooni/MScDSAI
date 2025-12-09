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
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    return mannwhitneyu, mo, np, pd, plt, shapiro, sns, spearmanr, stats


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
    mo.md(
        r"""
    RESEARCH QUESTIONS (RQ1 + RQ2)

    This analysis investigates how AI assistance influences both writers' and readers' 
    evaluations of short stories. The study is structured into one writer-focused RQ 
    and one reader-focused RQ with three subcomponents.

    RQ1 – Writers’ Self-Perceptions
        Do writers who use AI evaluate their own stories differently from those who 
        do not? (Mann–Whitney U tests on all self_* variables comparing writers who used AI 
        versus those who did not.)

    RQ2 – Readers’ Evaluations of Stories
        RQ2a – Blind Pre-Disclosure Ratings:
            Do readers rate AI-assisted stories differently before knowing whether 
            AI was involved? (Mann–Whitney U tests on pre_* variables)

        RQ2b – Post-Disclosure Judgements:
            After readers learn which stories used AI, how does this affect their 
            judgements of authorship, ownership, and contribution? (Mann–Whitney U)

        RQ2c – Misattribution Bias:
            Does the amount of AI involvement that readers *believe* a story had 
            (0–100% ai_guess_percent) relate to the blind ratings they had previously 
            given? (Spearman correlations between ai_guess_percent and pre_* variables)

    Together, these analyses provide a full view of how AI assistance affects both 
    the creators' and the evaluators' perceptions before and after AI involvement 
    is revealed.
    """
    )
    return


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
        plt.savefig("images/qqplots_all_rq1.png", dpi=300, bbox_inches="tight")


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

    rq1_results = rq1_mannwhitneyu()
    rq1_results
    return (rq1_results,)


@app.cell
def _(mannwhitneyu, np, plt, rq1_ai, rq1_results, sns):
    def rq1_plots(rq1_ai, rq1_results, alpha=0.05):

        if isinstance(rq1_results, tuple):
            rq1_results = rq1_results[0]

        sns.set_style("whitegrid")

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

        pretty_names = {
            "self_own_ideas": "Own ideas",
            "self_novel": "Novel",
            "self_original": "Original",
            "self_rare": "Rare",
            "self_appropriate": "Appropriate",
            "self_feasible": "Feasible",
            "self_publishable": "Publishable",
            "self_enjoyed": "Enjoyed writing",
            "self_badly_written": "Badly written",
            "self_boring": "Boring",
            "self_funny": "Funny",
            "self_twist": "Has a twist",
            "self_future": "Would read more",
        }

        # ------------------ recompute Cliff's delta ------------------
        def cliffs_delta(x, y):
            """Effect size for Mann–Whitney U: Cliff’s delta."""
            nx, ny = len(x), len(y)
            U, _ = mannwhitneyu(x, y, alternative="two-sided")
            return (2 * U) / (nx * ny) - 1

        df = rq1_results.copy()
        deltas = []
        for var in rq1_vars:
            g0 = rq1_ai.loc[rq1_ai["ai_used_actual"] == 0, var].dropna()
            g1 = rq1_ai.loc[rq1_ai["ai_used_actual"] == 1, var].dropna()
            deltas.append(cliffs_delta(g1, g0))   # positive = AI > non-AI

        df["cliffs_delta"] = deltas
        df["label"] = df["variable"].map(pretty_names).fillna(df["variable"])
        df["significant"] = df["p_value"] < alpha

        # 1. EFFECT-SIZE FOREST PLOT
        ef = df.sort_values("cliffs_delta")

        plt.figure(figsize=(8, 6), dpi=150)
        sns.barplot(
            data=ef,
            x="cliffs_delta",
            y="label",
            hue="significant",
            dodge=False,
        )
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("Cliff's delta (AI users – non-AI users)")
        plt.ylabel("")
        plt.title(f"Self-ratings: effect sizes by item (RQ1, α = {alpha})")
        plt.legend(title="Significant?")
        plt.tight_layout()
        plt.savefig("images/cliff_rq1.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 2. MEDIAN COMPARISON BARPLOT
        med = df.copy()

        med_long = med.melt(
            id_vars=["label"],
            value_vars=["median_no_ai", "median_ai"],
            var_name="group",
            value_name="median_rating",
        )
        med_long["group"] = med_long["group"].map({
            "median_no_ai": "No AI",
            "median_ai": "AI used"
        })

        plt.figure(figsize=(10, 6), dpi=150)
        sns.barplot(
            data=med_long,
            x="label",
            y="median_rating",
            hue="group",
        )
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("")
        plt.ylabel("Median rating")
        plt.title("Self-ratings by item and AI usage (RQ1)")
        plt.tight_layout()
        plt.savefig("images/medians_rq1.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 3. GRID OF BOXPLOTS
        vars_list = rq1_vars
        n = len(vars_list)
        n_cols = 4
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 9), dpi=150)
        axes = axes.flatten()

        for ax, var in zip(axes, vars_list):
            sns.boxplot(
                data=rq1_ai,
                x="ai_used_actual",
                y=var,
                ax=ax,
            )
            ax.set_title(pretty_names.get(var, var), fontsize=9)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["No AI", "AI"], fontsize=8)
            ax.set_xlabel("")
            ax.set_ylabel("")

        for i in range(len(vars_list), len(axes)):
            axes[i].axis("off")

        plt.suptitle("Self-ratings for own stories (AI vs No AI)", fontsize=14)
        plt.tight_layout()
        plt.savefig("images/boxplots_rq1.png", dpi=300, bbox_inches="tight")
        plt.subplots_adjust(top=0.92)
        plt.show()


    rq1_plots(rq1_ai, rq1_results)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##RQ1: Mann-Whitney U Test Conclusion
    Self-ratings were compared between writers who used AI (n = 141) and those who did not (n = 154). Across most creativity-related dimensions — including novelty, originality, rarity, feasibility, publishability, enjoyment, humour, and twist — the two groups evaluated their own stories similarly. This is reflected in the median barplot and the boxplots, where the distributions for AI and non-AI participants overlap closely, and in the effect-size plot, where most Cliff’s delta values cluster near zero (all p > .05).

    Three variables showed statistically significant differences.
    First, writers who used AI reported lower authorship of ideas (U = 12,437, p = .028), consistent with the effect seen in RQ2b: authors using AI felt that fewer of the story’s ideas originated from themselves. Secondly, AI users gave higher ratings for appropriateness (U = 9,299, p = .028), suggesting that AI guidance may have helped align their writing more closely to the expected task, genre, or constraints. Finally, AI users scored lower on the ‘future expectations’ item (U = 12,294, p = .045), indicating that their AI-assisted stories felt less personally meaningful or less likely to influence what they expect from future stories they will read.

    One variable should be interpreted cautiously: “badly written.”
    Although the dataset included an item labelled self_badly_written, participant-facing materials instead displayed the item as “well written.” Because this reverses the conceptual direction, it is uncertain whether higher scores indicate better writing or worse writing. For this reason, and consistent with RQ2a, this item is excluded from substantive interpretation.

    Overall, the visualisations and statistical tests together indicate that AI use had selective rather than widespread effects on self-perception. Participants who used AI differed meaningfully from non-AI writers only on authorship, appropriateness, and future-expectation impact. Broader evaluations of creativity, stylistic quality, and enjoyment were largely unaffected.

    ##Decision: H₀ is partially rejected
    AI use produced significant differences in 3 out of 12 self-perception variables. ("Badly-written ignored")
    Most aspects of writers’ self-evaluated creativity and stylistic quality did not differ between groups, but authorship, appropriateness, and perceived future-impact showed reliable effects. This indicates a narrow and specific, rather than general, influence of AI assistance on how writers evaluate their own work.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #RQ2: Reader Evaluation of AI-Influenced Stories
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #RQ2a: Do readers evaluate AI-assisted stories differently before knowing AI was used? (Blind Quality Judgement)

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
    ##RQ2a: Data Cleaning, Descriptive Statistics Checks, and Test Choice

    The evaluator dataset contained 3,543 blind (pre-disclosure) story ratings, with no missing values across any creativity or stylistic variables. Evaluations were well distributed between stories written without AI (1,860 ratings) and those written with AI assistance (1,683 ratings), providing balanced groups for comparison. Group-wise descriptive statistics indicated that median scores were broadly similar across AI and non-AI stories, with only modest differences in means that required formal inferential testing. These checks confirmed that the dataset was complete, coherent, and suitable for non-parametric analysis.

    Because the dependent variables (e.g., novelty, originality, publishability, enjoyment) are measured on 1–9 Likert scales, they are ordinal and do not meet parametric assumptions such as normality. Accordingly, the Mann–Whitney U test was selected to compare readers’ blind evaluations of stories written with versus without actual AI usage.
    """
    )
    return


@app.cell(hide_code=True)
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
    
        return pd.DataFrame(results), 


    results_df = rq2_mannwhitneyu(rq2a)
    results_df

    return (results_df,)


@app.cell(hide_code=True)
def _(np, plt, results_df, rq2a, sns):
    def rq2_plots(rq2a, results_df, alpha=0.05):

        # Tuple issue fix
        if isinstance(results_df, tuple):
            results_df = results_df[0]

        sns.set_style("whitegrid")

        pretty_names = {
            "pre_novel": "Novel",
            "pre_original": "Original",
            "pre_rare": "Rare",
            "pre_appropriate": "Appropriate",
            "pre_feasible": "Feasible",
            "pre_publishable": "Publishable",
            "pre_tt_enjoyed": "Enjoyed",
            "pre_tt_badly_written": "Badly written",
            "pre_tt_boring": "Boring",
            "pre_tt_funny": "Funny",
            "pre_tt_twist": "Twist",
            "pre_tt_future": "Would read more",
        }


        # 1. EFFECT-SIZE (CLIFF’S DELTA) PLOT
        ef = results_df.copy()
        ef["label"] = ef["variable"].map(pretty_names).fillna(ef["variable"])
        ef["significant"] = ef["p_value"] < alpha
        ef = ef.sort_values("cliffs_delta")

        plt.figure(figsize=(8, 6), dpi=150)
        sns.barplot(
            data=ef,
            x="cliffs_delta",
            y="label",
            hue="significant",
            dodge=False
        )
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("Cliff's delta (AI users – non-AI users)")
        plt.ylabel("")
        plt.title(f"Effect sizes per item (Mann–Whitney U, α={alpha})")
        plt.legend(title="Significant?")
        plt.tight_layout()
        plt.savefig("images/cliff_rq2a.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 2. MEDIAN BARPLOT
        med = results_df.copy()
        med["label"] = med["variable"].map(pretty_names).fillna(med["variable"])

        med_long = med.melt(
            id_vars=["label"],
            value_vars=["median_no_ai", "median_ai"],
            var_name="group",
            value_name="median_rating"
        )
        med_long["group"] = med_long["group"].map({
            "median_no_ai": "No AI",
            "median_ai": "AI used"
        })

        plt.figure(figsize=(10, 6), dpi=150)
        sns.barplot(
            data=med_long,
            x="label",
            y="median_rating",
            hue="group"
        )
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("")
        plt.ylabel("Median rating")
        plt.title("Median ratings by item and AI usage")
        plt.tight_layout()
        plt.savefig("images/median_rq2a.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 3. GRID OF  BOXPLOTS FOR ALL ITEMS

        vars_list = results_df["variable"].tolist()
        n = len(vars_list)

        n_cols = 4
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 9), dpi=150)
        axes = axes.flatten()

        for ax, var in zip(axes, vars_list):
            sns.boxplot(
                data=rq2a,
                x="ai_used_actual",
                y=var,
                ax=ax
            )
            ax.set_title(pretty_names.get(var, var), fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["No AI", "AI"], fontsize=8)

        # Hiding empty subplots
        for i in range(len(vars_list), len(axes)):
            axes[i].axis("off")

        plt.suptitle("Distribution of ratings across all items (AI vs No AI)", fontsize=14)
        plt.tight_layout()
        plt.savefig("images/boxplots_rq2a.png", dpi=300, bbox_inches="tight")
        plt.subplots_adjust(top=0.93)
        plt.show()

    rq2_plots(rq2a, results_df)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##RQ2a: Mann-Whitney U Test Conclusion
    Blind evaluations showed that AI-assisted stories were generally rated more favourably than human-only stories across a wide range of creativity and stylistic dimensions. Mann–Whitney U tests revealed small but consistent advantages for the AI group on novelty, originality, rarity, appropriateness, feasibility, publishability, enjoyment, and perceived narrative twist (most p < .01). Descriptively, the largest median differences appeared for the “twist”, “feasible”, and “rare” items, where AI-assisted stories received noticeably higher central ratings — a pattern also reflected in the boxplots.

    Two variables—boring and funny—showed no meaningful differences between groups, indicating that AI assistance neither enhanced nor reduced the perceived humour or dullness of the texts.

    One additional variable requires caution: “badly written.”
    Although the dataset included an item labelled pre_tt_badly_written, the survey materials shown to participants used the wording “well written.” Because of this mismatch, it is unclear whether higher scores indicate better or worse writing quality. This variable is therefore excluded from interpretation.

    ##Decision: H₀ is largely rejected.
    The null hypothesis of no difference between groups is rejected for 9 out of the 11 interpretable dimensions ("Badly-written ignored").
    Evaluators rated AI-assisted stories significantly higher on the majority of quality measures, with only boring and funny showing no significant group differences.

    This pattern indicates that AI assistance systematically enhances perceived creative writing quality in blind evaluations.
    """
    )
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    ##RQ2b: Data Cleaning, Descriptive Statistics Checks, and Test Choice 

    The post-disclosure evaluator dataset contained 3,543 authorship-related ratings, evenly split between stories written without AI (1,860 evaluations) and those written with AI assistance (1,683 evaluations). No missing values were present in the key post-disclosure variables (post_authors_ideas, post_ownership, post_profit, post_ai_assist). Group-wise descriptive statistics showed substantial downward shifts for AI-assisted stories in perceived author contribution and ownership, while the perceived AI contribution increased as expected. These patterns suggested that disclosure of AI involvement may meaningfully alter evaluators’ credit attribution judgements.

    Because all dependent variables are Likert-type ordinal ratings and exhibit non-normal distributions, non-parametric testing is required. Accordingly, the Mann–Whitney U test was selected to compare post-disclosure authorship and ownership assessments between AI-assisted and non-AI stories.
    """
    )
    return


@app.cell(hide_code=True)
def _(mannwhitneyu, pd, rq2b):
    def rq2b_mannwhitneyu(rq2b):

        post_vars = [
            "post_authors_ideas",
            "post_ownership",
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

    return (rq2b_results,)


@app.cell(hide_code=True)
def _(plt, rq2b, rq2b_results, sns):

    def rq2b_plots(rq2b, results_df, alpha=0.05):
        if isinstance(results_df, tuple):
            results_df = results_df[0]

        sns.set_style("whitegrid")

        pretty_names = {
            "post_authors_ideas": "Authors' ideas (perceived source)",
            "post_ownership": "Ownership of ideas",
        }

        # 1. EFFECT-SIZE PLOT (Cliff’s delta)

        ef = results_df.copy()
        ef["label"] = ef["variable"].map(pretty_names)
        ef["significant"] = ef["p_value"] < alpha

        plt.figure(figsize=(6, 4), dpi=150)
        sns.barplot(
            data=ef,
            x="cliffs_delta",
            y="label",
            hue="significant",
            dodge=False
        )
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("Cliff's delta (AI users – non-AI users)")
        plt.ylabel("")
        plt.title("Effect sizes for authorship + ownership judgements (RQ2b)")
        plt.savefig("images/cliff_rq2b.png", dpi=300, bbox_inches="tight")
        plt.tight_layout()
        plt.show()

        # 2. MEDIAN COMPARISON BARPLOT

        med = results_df.copy()
        med["label"] = med["variable"].map(pretty_names)

        med_long = med.melt(
            id_vars=["label"],
            value_vars=["median_no_ai", "median_ai"],
            var_name="group",
            value_name="median_rating",
        )
        med_long["group"] = med_long["group"].map({
            "median_no_ai": "No AI",
            "median_ai": "AI used"
        })

        plt.figure(figsize=(6, 4), dpi=150)
        sns.barplot(
            data=med_long,
            x="label",
            y="median_rating",
            hue="group"
        )
        plt.xticks(rotation=0)
        plt.xlabel("")
        plt.ylabel("Median rating")
        plt.title("Median ratings by AI usage (RQ2b)")
        plt.tight_layout()
        plt.savefig("images/median_rq2b.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 3. BOXPLOTS SIDE-BY-SIDE

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

        for ax, var in zip(axes, results_df["variable"]):
            sns.boxplot(
                data=rq2b,
                x="ai_used_actual",
                y=var,
                ax=ax
            )
            ax.set_title(pretty_names[var], fontsize=10)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["No AI", "AI"], fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("Rating")

        plt.tight_layout()
        plt.savefig("images/boxplots_rq2b.png", dpi=300, bbox_inches="tight")
        plt.show()



    rq2b_plots(rq2b, rq2b_results)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## RQ2b: Mann-Whitney U Test Conclusion

    Once evaluators were informed whether a story had been written with AI assistance, they were asked additional questions about the perceived authorship and ownership of the ideas. These post-disclosure judgements showed clear and systematic differences between the AI and non-AI groups.
    Both the median comparison plot and the boxplots show the same strong pattern:
    stories written with AI assistance were attributed substantially fewer ideas and were judged to involve weaker ownership by their human authors. Median ratings dropped from 8 to 5 for authors’ ideas and from 8 to 6 for ownership, with noticeably lower distributions across the full range of responses. These graphical patterns mirror the inferential outcomes.

    The Mann–Whitney U tests confirmed this visually observed effect. Both variables showed very large and highly significant differences between groups (all p < 10⁻¹³⁰), indicating a strong evaluative penalty applied to authors who used AI. Cliff’s delta values were correspondingly large and negative, reflecting a consistent downward shift in judgements for the AI-assisted group.

    The profit-sharing variable was excluded from inferential analysis for two reasons. First, the profit question was displayed only for stories whose authors used AI, leaving no comparable data for human-only stories. Second, even within the AI group, responses were highly inconsistent and often conceptually contradictory, with some participants entering 0% to signal that the AI deserved no credit rather than to specify the human author’s share. For these reasons, the variable is considered statistically unreliable and conceptually ambiguous.

    ##Decision: H₀ is fully rejected.
    Across both valid post-disclosure measures—authors’ ideas and ownership—evaluators assigned significantly lower creative credit to authors who used AI. The effects were large, consistent across visual and inferential analyses, and unidirectional. These findings demonstrate that disclosing AI involvement prompts evaluators to substantially downgrade the human author’s perceived contribution and ownership of ideas.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #RQ2c - Does the belief or suspicion that a story involved AI predict lower story ratings, even before readers are told anything? (Effect of AI Suspicion on Blind Ratings)

    Null hypothesis (H₀)
    The analysis assumes that evaluators’ assumed level of AI involvement (ai_guess_percent) has no meaningful association with the blind pre-disclosure creativity and quality ratings (pre_* variables).
    Formally, across all correlations computed in rq2c_spearman, Spearman’s ρ = 0.
    Any observed correlations are expected to be trivially small and reflect sampling noise rather than genuine misattribution bias.

    Alternative hypothesis (H₁)
    The analysis proposes that evaluators’ assumptions about AI involvement (ai_guess_percent) are associated with the blind pre-disclosure ratings they assigned. If this were true, stories suspected of heavier AI use would show systematically higher or lower pre-ratings.
    Formally, at least one of the Spearman correlations computed in rq2c_spearman is non-zero.
    """
    )
    return


@app.cell
def _(evals_full, pd):
    def rq2c_dataprep(evals_full: pd.DataFrame) -> pd.DataFrame:

        df = evals_full.copy()

        # Alias for the 0–100% AI guess
        if "post_ai_assist" not in df.columns:
            raise KeyError("Column 'post_ai_assist' not found in evals_full.")
        df["ai_guess_percent"] = df["post_ai_assist"]

        pre_vars = [
            "pre_novel", "pre_original", "pre_rare",
            "pre_appropriate", "pre_feasible", "pre_publishable",
            "pre_tt_enjoyed", "pre_tt_badly_written",
            "pre_tt_boring", "pre_tt_funny",
            "pre_tt_twist", "pre_tt_future",
        ]

        base_cols = [
            "story_id",
            "evaluator_code",
            "condition",
            "ai_used_actual",
            "ai_guess_percent",
        ]

        cols = [c for c in base_cols + pre_vars if c in df.columns]

        rq2c = df[cols].copy()

        # Drop rows without an AI guess
        rq2c = rq2c.dropna(subset=["ai_guess_percent"])

        return rq2c

    rq2c = rq2c_dataprep(evals_full)
    rq2c.to_csv("data/rq2c_clean.csv", index=False)
    rq2c.head()
    return (rq2c,)


@app.cell
def _(pd, rq2c):
    def rq2c_datacheck(rq2c: pd.DataFrame) -> None:
        print("rq2c shape:", rq2c.shape)
        print("\nMissing values per column:")
        print(rq2c.isna().sum())

        print("\nAI guess (ai_guess_percent) summary:")
        print(rq2c["ai_guess_percent"].describe())

        # Check range of AI guess is within 0–100
        min_guess = rq2c["ai_guess_percent"].min()
        max_guess = rq2c["ai_guess_percent"].max()
        print(f"\nAI guess range: {min_guess:.1f} to {max_guess:.1f}")

    rq2c_datacheck(rq2c)

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #RQ2c: Data Cleaning, Descriptive Statistics Checks, and Test Choice
    The RQ2c dataset contained 3,543 pre-disclosure story evaluations, each paired with the evaluator’s later estimate of how much AI assistance they believed had been used (0–100%). All variables relevant to this analysis were fully complete, with no missing values across the AI-guess measure or any of the pre-disclosure creativity and stylistic ratings. This ensured that all cases could be included in the correlation analysis without imputation or casewise deletion.

    Descriptive statistics for the AI-guess variable indicated a wide and well-distributed range of perceived AI involvement (M = 46.24, SD = 31.29, range 0–100). The distribution showed substantial variability across evaluators, with the interquartile range spanning from 20% to 75%. This variation is crucial for RQ2c, as it demonstrates that participants did not uniformly assume stories were human-written or AI-written; instead, they expressed diverse levels of suspicion, providing sufficient spread to test for misattribution effects.

    Because the dependent variables (e.g., pre-novel, pre-original, pre-feasible, pre-twist) are measured on ordinal 1–9 Likert scales, they do not satisfy parametric assumptions such as normality or interval-level measurement. In addition, the AI-guess variable is continuous but non-normally distributed, a common property of subjective probability estimates. Accordingly, Spearman’s rank-order correlation was chosen as the appropriate inferential method. This non-parametric test evaluates whether higher perceived AI involvement is associated with systematic differences in blind creativity and quality ratings, without assuming linearity or homoscedasticity.

    This combination of complete data, broad variability in AI guesses, and ordinal outcome measures confirms that the dataset is suitable for a non-parametric correlation-based approach to assess misattribution bias.
    """
    )
    return


@app.cell
def _(pd, rq2c, spearmanr):
    def rq2c_spearman(rq2c: pd.DataFrame) -> pd.DataFrame:
        if "ai_guess_percent" not in rq2c.columns:
            raise KeyError("Expected column 'ai_guess_percent' in rq2c.")

        pre_vars = [c for c in rq2c.columns if c.startswith("pre_")]

        results = []
        for var in pre_vars:
            rho, p = spearmanr(
                rq2c[var],
                rq2c["ai_guess_percent"],
                nan_policy="omit"
            )
            results.append({
                "variable": var,
                "spearman_rho": rho,
                "p_value": p,
            })

        rq2c_results = pd.DataFrame(results)
        return rq2c_results

    rq2c_results = rq2c_spearman(rq2c)
    rq2c_results
    return (rq2c_results,)


@app.cell
def _(pd, plt, rq2c_results, sns):
    def rq2c_plots(rq2c_results: pd.DataFrame, alpha: float = 0.05) -> None:
        sns.set_style("whitegrid")

        pretty_names = {
            "pre_novel": "Novel",
            "pre_original": "Original",
            "pre_rare": "Rare",
            "pre_appropriate": "Appropriate",
            "pre_feasible": "Feasible",
            "pre_publishable": "Publishable",
            "pre_tt_enjoyed": "Enjoyed",
            "pre_tt_badly_written": "Badly written",
            "pre_tt_boring": "Boring",
            "pre_tt_funny": "Funny",
            "pre_tt_twist": "Twist",
            "pre_tt_future": "Would read more",
        }

        df = rq2c_results.copy()
        df["label"] = df["variable"].map(pretty_names).fillna(df["variable"])
        df["significant"] = df["p_value"] < alpha
        df = df.sort_values("spearman_rho")

        plt.figure(figsize=(7, 6), dpi=150)
        sns.barplot(
            data=df,
            x="spearman_rho",
            y="label",
            hue="significant",
            dodge=False,
        )
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("Spearman correlation (rho)")
        plt.ylabel("")
        plt.title("RQ2c: AI guess (%) vs pre-disclosure ratings")
        plt.legend(title=f"Significant (p < {alpha})?")
        plt.tight_layout()
        plt.savefig("images/spearman_rq2c.png", dpi=300, bbox_inches="tight")

        plt.show()

    rq2c_plots(rq2c_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##RQ2c: Spearman’s rank-order correlation Conclusion
    For RQ2c, the analysis tested whether evaluators’ assumptions about AI involvement were associated with the creativity and quality ratings they provided before learning whether a story had been written with AI assistance. After the disclosure phase, each evaluator indicated the extent to which they believed AI had contributed to the story (0–100%). This value was stored in the dataset as post_ai_assist and standardised as ai_guess_percent during data preparation (rq2c_dataprep).

    The key question was whether this AI-guess variable predicted earlier (blind) pre-disclosure ratings such as pre_novel, pre_original, pre_publishable, and related variables. To test this, the script rq2c_spearman computed a series of Spearman rank-order correlations between ai_guess_percent and all pre-disclosure rating variables. These correlations were then visualised in the rq2c_plots function using a forest-style barplot.

    The forest plot revealed a coherent but extremely small pattern. Across most creativity and stylistic dimensions—including novelty, originality, rarity, appropriateness, feasibility, enjoyment, narrative twist, and publishability—higher assumed AI involvement was associated with slightly lower pre-disclosure ratings. Several of these correlations were statistically significant (p < .05), but the effect sizes were tiny, with Spearman’s ρ values ranging from approximately –.10 to –.02. In practical terms, such values are negligible: a correlation of –0.05, for example, implies that even a 100-point difference in AI guess would alter the predicted creativity rating by less than one-tenth of a point on a 9-point scale. Thus, while the script detects these patterns numerically, they are too small to have any real-world interpretive value.

    The only variable showing a positive correlation was pre_tt_boring, where stories rated as more boring were slightly more likely to be subsequently misattributed to AI. Again, the effect was statistically detectable but too small to constitute meaningful bias.

    One item required the same caution applied in earlier analyses: pre_tt_badly_written. Although recorded under that name in the dataset, participants saw the opposite phrasing (“well written”) in the survey interface. Because of this mismatch, its scale direction cannot be interpreted with confidence. The value was included in the dataset for completeness but excluded from substantive conclusions.
    Overall, the statistical output (rq2c_results) and the visualisations confirm that evaluators’ later beliefs about AI involvement did not meaningfully influence the earlier blind ratings they provided. Pre-disclosure judgements appear to reflect genuine perceptions of story quality rather than hidden expectations about AI authorship. The presence of many statistically significant p-values is explained by the very large sample size (N = 3,543), which makes even microscopic effects detectable; however, their magnitude indicates that they are not practically significant.

    ##Decision: H₀ is retained
    Assumed AI involvement had no meaningful effect on blind creativity evaluations. Although the code identified several small correlations, all were trivial in size, and therefore do not support the presence of misattribution bias. Evaluative penalties only emerged after authorship was revealed (RQ2b), not during the initial blind evaluation phase.
    """
    )
    return


if __name__ == "__main__":
    app.run()
