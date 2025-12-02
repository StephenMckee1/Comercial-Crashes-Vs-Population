import argparse
import os
import re
import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression


def to_numeric(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    # remove commas, dollar signs, quotes, spaces
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s == "" or s == "-." or s == "-":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def detect_columns(df):
    # Look for population and crash columns for two years
    pop_cols = [c for c in df.columns if "Population" in c]
    total_crash_cols = [c for c in df.columns if "Total Crashes" in c]
    # fallback: any column with 'Commercial' in name
    commercial_cols = [c for c in df.columns if "Commercial" in c]
    return pop_cols, total_crash_cols, commercial_cols


def prepare_df(df):
    # Normalize column names by stripping
    df.columns = [c.strip() for c in df.columns]

    pop_cols, total_crash_cols, commercial_cols = detect_columns(df)

    # Try to pick the 2021 and 2022 population columns
    pop21 = None
    pop22 = None
    for c in pop_cols:
        if "'21" in c or "2021" in c:
            pop21 = c
        if "'22" in c or "2022" in c:
            pop22 = c

    crash21 = None
    crash22 = None
    for c in total_crash_cols:
        if "'21" in c or "2021" in c:
            crash21 = c
        if "'22" in c or "2022" in c:
            crash22 = c

    # If explicit change columns exist, use them as fallback
    change_pop_col = None
    change_crash_col = None
    for c in df.columns:
        if (
            "Change in Population" in c
            or "Change in Pop" in c
            or "Change in Population" in c
        ):
            change_pop_col = c
        if "Change in Total Crashes" in c or "Change in Crashes" in c:
            change_crash_col = c

    # Convert numeric columns
    for c in df.columns:
        df[c + "_num"] = df[c].apply(to_numeric)

    # Create standardized columns
    if pop21 and pop22:
        df["pop_21"] = df[pop21].apply(to_numeric)
        df["pop_22"] = df[pop22].apply(to_numeric)
        df["pop_change"] = df["pop_22"] - df["pop_21"]
    elif change_pop_col:
        df["pop_change"] = df[change_pop_col].apply(to_numeric)
        # try to reconstruct pop_21 if available
        if len(pop_cols) >= 1:
            df["pop_21"] = df[pop_cols[0]].apply(to_numeric)
            df["pop_22"] = df["pop_21"] + df["pop_change"]
    else:
        df["pop_change"] = np.nan

    if crash21 and crash22:
        df["crash_21"] = df[crash21].apply(to_numeric)
        df["crash_22"] = df[crash22].apply(to_numeric)
        df["crash_change"] = df["crash_22"] - df["crash_21"]
    elif change_crash_col:
        df["crash_change"] = df[change_crash_col].apply(to_numeric)
        if len(total_crash_cols) >= 1:
            df["crash_21"] = df[total_crash_cols[0]].apply(to_numeric)
            df["crash_22"] = df["crash_21"] + df["crash_change"]
    else:
        df["crash_change"] = np.nan

    # Percent changes where possible
    df["pop_pct_change"] = df.apply(
        lambda r: (
            (r["pop_change"] / r["pop_21"] * 100)
            if pd.notna(r.get("pop_change"))
            and pd.notna(r.get("pop_21"))
            and r.get("pop_21") != 0
            else np.nan
        ),
        axis=1,
    )
    df["crash_pct_change"] = df.apply(
        lambda r: (
            (r["crash_change"] / r["crash_21"] * 100)
            if pd.notna(r.get("crash_change"))
            and pd.notna(r.get("crash_21"))
            and r.get("crash_21") != 0
            else np.nan
        ),
        axis=1,
    )

    return df


def analyze(df, outdir, use_percent=True, annotate_top=5, percapita=False):
    # choose x and y
    xcol = "pop_pct_change" if use_percent else "pop_change"
    ycol = "crash_pct_change" if use_percent else "crash_change"

    # If percapita flag is set, compute crash rates per 100k people and analyze the change in rates
    if percapita:
        # require crash_21/crash_22 and pop_21/pop_22
        if (
            "crash_21" not in df.columns
            or "crash_22" not in df.columns
            or "pop_21" not in df.columns
        ):
            raise RuntimeError(
                "Per-capita analysis requires crash_21, crash_22 and pop_21/pop_22 columns"
            )
        df["crash_rate_21"] = df.apply(
            lambda r: (
                (r["crash_21"] / r["pop_21"] * 100000)
                if pd.notna(r["crash_21"])
                and pd.notna(r["pop_21"])
                and r["pop_21"] != 0
                else np.nan
            ),
            axis=1,
        )
        df["crash_rate_22"] = df.apply(
            lambda r: (
                (r["crash_22"] / r["pop_22"] * 100000)
                if pd.notna(r.get("crash_22"))
                and pd.notna(r.get("pop_22"))
                and r.get("pop_22") not in (0, None)
                else np.nan
            ),
            axis=1,
        )
        df["crash_rate_change"] = df["crash_rate_22"] - df["crash_rate_21"]
        df["crash_rate_pct_change"] = df.apply(
            lambda r: (
                (r["crash_rate_change"] / r["crash_rate_21"] * 100)
                if pd.notna(r.get("crash_rate_change"))
                and pd.notna(r.get("crash_rate_21"))
                and r.get("crash_rate_21") != 0
                else np.nan
            ),
            axis=1,
        )
        # swap ycol to use rate change
        ycol = "crash_rate_pct_change" if use_percent else "crash_rate_change"

    sub = df[["State", xcol, ycol]].copy()
    sub = sub.dropna(subset=[xcol, ycol])

    results = {}
    if len(sub) < 2:
        raise RuntimeError(
            "Not enough data rows with both population and crash changes to analyze"
        )

    x = sub[xcol].values
    y = sub[ycol].values

    # Pearson
    pearson_r, pearson_p = stats.pearsonr(x, y)
    # Spearman
    spearman_r, spearman_p = stats.spearmanr(x, y)

    # Linear regression (for reporting)
    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y)
    y_pred = lr.predict(x.reshape(-1, 1))
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    results["pearson_r"] = pearson_r
    results["pearson_p"] = pearson_p
    results["spearman_r"] = spearman_r
    results["spearman_p"] = spearman_p
    results["lr_coef"] = float(lr.coef_[0])
    results["lr_intercept"] = float(lr.intercept_)
    results["r2"] = float(r2)

    # Save merged table
    merged_path = os.path.join(outdir, "merged_changes.csv")
    sub.to_csv(merged_path, index=False)

    # Save summary
    txt = [
        f"Rows analyzed: {len(sub)}",
        f"X column: {xcol}",
        f"Y column: {ycol}",
        f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.4g}",
        f"Spearman rho = {spearman_r:.4f}, p = {spearman_p:.4g}",
        f"Linear regression slope = {lr.coef_[0]:.6f}, intercept = {lr.intercept_:.6f}, R^2 = {r2:.4f}",
    ]
    summary_path = os.path.join(outdir, "correlation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))

    # Plot
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x, y=y, ci=None, scatter_kws={"s": 60})
    plt.xlabel(
        "Population % change (2021->2022)" if use_percent else "Population change"
    )
    plt.ylabel("Crashes % change (2021->2022)" if use_percent else "Crashes change")
    plt.title("Population change vs Crash change by State")
    plt.grid(True, alpha=0.3)

    # Annotate top states by pop growth
    if annotate_top and annotate_top > 0:
        top = sub.sort_values(by=xcol, ascending=False).head(annotate_top)
        for _, row in top.iterrows():
            plt.text(row[xcol], row[ycol], row["State"], fontsize=9, weight="bold")

    plot_path = os.path.join(outdir, "pop_vs_crash.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return results, merged_path, summary_path, plot_path


def run_bootstrap(
    df,
    outdir,
    use_percent=True,
    percapita=False,
    extra_years=9,
    niter=1000,
    random_state=None,
):
    """Bootstrap observed (pop change, crash change) pairs to simulate extra years per state.

    Approach: treat each observed state-year pair in `df` as a unit. To simulate `extra_years` extra
    year-to-year observations per state, sample with replacement `len(states) * extra_years` pairs
    from the observed pairs and append to the observed set to form a larger dataset. Compute Pearson,
    Spearman and regression slope for each bootstrap iteration to obtain a sampling distribution.
    """
    rng = np.random.default_rng(random_state)

    xcol = "pop_pct_change" if use_percent else "pop_change"
    if percapita:
        ycol = "crash_rate_pct_change" if use_percent else "crash_rate_change"
    else:
        ycol = "crash_pct_change" if use_percent else "crash_change"

    observed = df[[xcol, ycol]].dropna()
    observed_pairs = observed.values
    n_states = len(observed_pairs)
    if n_states < 2:
        raise RuntimeError("Not enough observed pairs to bootstrap")

    # Prepare arrays to store results
    pearson_rs = np.zeros(niter)
    spearman_rs = np.zeros(niter)
    slopes = np.zeros(niter)

    sampled_count = n_states * extra_years

    for i in range(niter):
        # sample indices with replacement
        idx = rng.integers(0, n_states, size=sampled_count)
        sampled = observed_pairs[idx]
        # combine observed + sampled
        combined = np.vstack([observed_pairs, sampled])
        x = combined[:, 0]
        y = combined[:, 1]
        # compute stats
        try:
            pr, pp = stats.pearsonr(x, y)
        except Exception:
            pr, pp = np.nan, np.nan
        sr, sp = stats.spearmanr(x, y)
        # regression slope
        try:
            lr = LinearRegression().fit(x.reshape(-1, 1), y)
            slope = float(lr.coef_[0])
        except Exception:
            slope = np.nan

        pearson_rs[i] = pr
        spearman_rs[i] = sr
        slopes[i] = slope

    # Save bootstrap summary
    boot_df = pd.DataFrame(
        {
            "pearson_r": pearson_rs,
            "spearman_r": spearman_rs,
            "slope": slopes,
        }
    )
    boot_path = os.path.join(outdir, f"bootstrap_extra{extra_years}_iters{niter}.csv")
    boot_df.to_csv(boot_path, index=False)

    # Write percentiles
    summary_lines = []
    for col in ["pearson_r", "spearman_r", "slope"]:
        arr = boot_df[col].dropna().values
        pcts = (
            np.percentile(arr, [2.5, 25, 50, 75, 97.5])
            if len(arr) > 0
            else [np.nan] * 5
        )
        summary_lines.append(f"{col} percentiles (2.5,25,50,75,97.5): {pcts}")

    bs_summary_path = os.path.join(
        outdir, f"bootstrap_summary_extra{extra_years}_iters{niter}.txt"
    )
    with open(bs_summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # Plot histogram of Pearson r
    plt.figure(figsize=(7, 4))
    sns.histplot(pearson_rs[~np.isnan(pearson_rs)], bins=40, kde=True)
    plt.title(
        f"Bootstrap distribution of Pearson r (extra_years={extra_years}, iters={niter})"
    )
    plt.xlabel("Pearson r")
    plt.tight_layout()
    hist_path = os.path.join(
        outdir, f"bootstrap_pearson_hist_extra{extra_years}_iters{niter}.png"
    )
    plt.savefig(hist_path, dpi=150)
    plt.close()

    return boot_path, bs_summary_path, hist_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze relation between population change and crash change by state"
    )
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument(
        "--percent", action="store_true", help="Analyze percent changes (default)"
    )
    parser.add_argument(
        "--no-percent",
        dest="percent",
        action="store_false",
        help="Analyze absolute changes",
    )
    parser.set_defaults(percent=True)
    parser.add_argument(
        "--annotate-top",
        type=int,
        default=5,
        help="Number of top population-growth states to annotate on plot",
    )
    parser.add_argument(
        "--percapita",
        action="store_true",
        help="Normalize crashes to per-100k population and analyze change in crash rate",
    )
    parser.add_argument(
        "--bootstrap-years",
        type=int,
        default=0,
        help="Number of extra years per state to simulate via bootstrap (0 = no bootstrap)",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Number of bootstrap iterations",
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV file not found: {args.csv}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv, dtype=str)
    df = prepare_df(df)

    # If commercial columns exist, we could swap crash cols to those - currently we default to Total Crashes
    try:
        results, merged, summary, plot = analyze(
            df,
            args.outdir,
            use_percent=args.percent,
            annotate_top=args.annotate_top,
            percapita=args.percapita,
        )
    except Exception as e:
        print("Analysis failed:", e)
        raise

    # Optionally run bootstrap simulation to simulate extra years
    if args.bootstrap_years and args.bootstrap_years > 0:
        print(
            f"Running bootstrap: extra_years={args.bootstrap_years}, iters={args.bootstrap_iters} ..."
        )
        try:
            boot_csv, boot_summary, boot_hist = run_bootstrap(
                df,
                args.outdir,
                use_percent=args.percent,
                percapita=args.percapita,
                extra_years=args.bootstrap_years,
                niter=args.bootstrap_iters,
            )
            print("Bootstrap outputs:", boot_csv, boot_summary, boot_hist)
        except Exception as e:
            print("Bootstrap failed:", e)

    print("Analysis complete. Summary:")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("Outputs saved in", args.outdir)


if __name__ == "__main__":
    main()
