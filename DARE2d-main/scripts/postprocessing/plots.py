#!/usr/bin/env python3
"""
qazi - 09/12/2025

To run:
python scripts/postprocessing/shadow_realm.py `
  --summary "path_to_the_summary_csv_file" `
  --tiff "path_to_the_post_processed_movie" `
  --plots_dir "path_where_postprocess_results_were_saved"

Utilities to visualize the post-processed outputs:
 - Load CSV summary and make distribution plots for length, n_models, support_fraction
 - Compute a quality score for each event and produce plots and a small metrics CSV
 - Produce per-division timecourse scatter plots with error bars (frames on x-axis)

Usage:
    python postprocess_plots_and_quality.py --summary /path/to/post_processed_movie_summary.csv --tiff /path/to/post_processed_movie.tiff --plots_dir /path/to/out
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio

def compute_quality_score(df):
    """
    Compute a quality score for each detection row in df.
    Returns a numpy array of scores normalized to [0,1].

    Heuristic:
      - support_fraction (0..1) : more models supporting -> higher quality
      - pos_std (lower is better)
      - angle_std_deg (lower is better)
      - length_std (lower is better)
    We combine normalized terms:
      score = 0.45*support_frac + 0.25*(1 - norm_pos_std) + 0.2*(1 - norm_angle_std) + 0.1*(1 - norm_length_std)
    where norms are min-max normalized in the dataframe (clipped).
    """
    # copy to avoid modifying original
    dfc = df.copy().reset_index(drop=True)

    # normalize pos_std, angle_std_deg, length_std to 0..1 (0 best for these)
    def minmax_norm(series):
        s = series.copy().astype(float)
        mn = np.nanmin(s) if not np.all(np.isnan(s)) else 0.0
        mx = np.nanmax(s) if not np.all(np.isnan(s)) else mn + 1.0
        if mx == mn:
            return np.zeros_like(s, dtype=float)
        return (s - mn) / (mx - mn)

    # fill missing with zeros where appropriate
    norm_pos = minmax_norm(dfc['pos_std'].fillna(0.0))
    norm_ang = minmax_norm(dfc['angle_std_deg'].fillna(0.0))
    norm_len = minmax_norm(dfc['length_std'].fillna(0.0))
    support = dfc['support_fraction'].fillna(0.0).to_numpy().astype(float)

    # score weighting
    score = 0.45 * support + 0.25 * (1.0 - norm_pos) + 0.2 * (1.0 - norm_ang) + 0.10 * (1.0 - norm_len)
    # clip to [0,1]
    score = np.clip(score, 0.0, 1.0)
    return score

def plot_distributions(df, plots_dir, movie_stem):
    """
    Make distribution plots for length, length_std, n_models, support_fraction.
    Produces PNGs in plots_dir.
    (pos_std and angle_std_deg histograms were removed per request)
    """
    os.makedirs(plots_dir, exist_ok=True)
    cols = [
        ("length", "Rod length (px)"),
        ("length_std", "Rod length std (px)"),
        ("n_models", "Number of models (n_models)"),
        ("support_fraction", "Support fraction")
    ]
    for key, label in cols:
        if key not in df.columns:
            continue
        plt.figure(figsize=(6,4))
        series = df[key].dropna()
        if series.empty:
            plt.close()
            continue
        plt.hist(series, bins=40)
        plt.xlabel(label)
        plt.ylabel("Count")
        plt.title(f"{label} distribution ({movie_stem})")
        out_path = os.path.join(plots_dir, f"{movie_stem}_{key}_hist.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[INFO] Saved distribution plot: {out_path}")

def plot_quality_and_scatter(df, plots_dir, movie_stem):
    """
    Plot quality score histogram and scatter quality vs support_fraction and pos_std.
    """
    os.makedirs(plots_dir, exist_ok=True)
    if 'quality_score' not in df.columns:
        print("[WARN] 'quality_score' column missing; skipping quality plots.")
        return

    # quality histogram
    plt.figure(figsize=(6,4))
    plt.hist(df['quality_score'].dropna(), bins=40)
    plt.xlabel("Quality score")
    plt.ylabel("Count")
    plt.title(f"Quality score distribution ({movie_stem})")
    out_path = os.path.join(plots_dir, f"{movie_stem}_quality_hist.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved quality histogram: {out_path}")

    # scatter quality vs support_fraction
    if 'support_fraction' in df.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(df['support_fraction'], df['quality_score'], s=6, alpha=0.6)
        plt.xlabel("Support fraction")
        plt.ylabel("Quality score")
        plt.title(f"Quality vs Support fraction ({movie_stem})")
        out_path = os.path.join(plots_dir, f"{movie_stem}_quality_vs_support.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[INFO] Saved scatter: {out_path}")

    # scatter quality vs pos_std (lower pos_std hopefully higher quality) - keep if pos_std exists
    if 'pos_std' in df.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(df['pos_std'], df['quality_score'], s=6, alpha=0.6)
        plt.xlabel("pos_std (px)")
        plt.ylabel("Quality score")
        plt.title(f"Quality vs Positional std ({movie_stem})")
        out_path = os.path.join(plots_dir, f"{movie_stem}_quality_vs_posstd.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[INFO] Saved scatter: {out_path}")

def plot_division_timecourses(df, plots_dir, movie_stem):
    """
    Create two scatter/timecourse plots (with errorbars) for each division (grouped by cell_id):
      1) x and y vs frame (errorbars use pos_std)
      2) angle_deg and length vs frame (errorbars use angle_std_deg and length_std)
    Each division is plotted as a faint line+markers so you can see per-division dynamics.
    """
    os.makedirs(plots_dir, exist_ok=True)

    if 'cell_id' not in df.columns or 'frame' not in df.columns:
        print("[WARN] 'cell_id' or 'frame' missing; skipping division timecourse plots.")
        return

    groups = df.groupby('cell_id')
    # Plot 1: x and y
    plt.figure(figsize=(10,6))
    for cid, g in groups:
        g_sorted = g.sort_values('frame')
        frames = g_sorted['frame'].to_numpy()
        xvals = g_sorted['x'].to_numpy() if 'x' in g_sorted.columns else None
        yvals = g_sorted['y'].to_numpy() if 'y' in g_sorted.columns else None
        pos_err = g_sorted['pos_std'].to_numpy() if 'pos_std' in g_sorted.columns else None

        if xvals is not None:
            plt.errorbar(frames, xvals, yerr=pos_err, fmt='.-', alpha=0.25, markersize=3)
        if yvals is not None:
            plt.errorbar(frames, yvals, yerr=pos_err, fmt='.--', alpha=0.25, markersize=3)

    plt.xlabel("Frame")
    plt.ylabel("Position (px)")
    plt.title(f"Divisions: x (solid) and y (dashed) over frames ({movie_stem})")
    out_path = os.path.join(plots_dir, f"{movie_stem}_divisions_xy_timecourses.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved divisions XY timecourses: {out_path}")

    # Plot 2: angle_deg and length
    plt.figure(figsize=(10,6))
    for cid, g in groups:
        g_sorted = g.sort_values('frame')
        frames = g_sorted['frame'].to_numpy()
        ang = g_sorted['angle_deg'].to_numpy() if 'angle_deg' in g_sorted.columns else None
        ang_err = g_sorted['angle_std_deg'].to_numpy() if 'angle_std_deg' in g_sorted.columns else None
        length = g_sorted['length'].to_numpy() if 'length' in g_sorted.columns else None
        length_err = g_sorted['length_std'].to_numpy() if 'length_std' in g_sorted.columns else None

        if ang is not None:
            plt.errorbar(frames, ang, yerr=ang_err, fmt='.-', alpha=0.25, markersize=3)
        if length is not None:
            plt.errorbar(frames, length, yerr=length_err, fmt='.--', alpha=0.25, markersize=3)

    plt.xlabel("Frame")
    plt.ylabel("Angle (deg) / Length (px)")
    plt.title(f"Divisions: angle_deg (solid) and length (dashed) over frames ({movie_stem})")
    out_path = os.path.join(plots_dir, f"{movie_stem}_divisions_angle_length_timecourses.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved divisions angle/length timecourses: {out_path}")

def save_metrics_csv(metrics_dict, out_path):
    """
    Save a small two-column CSV with metric name and value.
    """
    dfm = pd.DataFrame(list(metrics_dict.items()), columns=['metric','value'])
    dfm.to_csv(out_path, index=False)
    print(f"[INFO] Wrote metrics CSV: {out_path}")

def main():
    p = argparse.ArgumentParser(description="Plotting and quality analysis for post-processed consensus outputs")
    p.add_argument("--summary", required=True, help="Path to post_processed_{movie_stem}_summary.csv")
    p.add_argument("--tiff", required=False, help="Path to post_processed_{movie_stem}.tiff (optional, for reference)")
    p.add_argument("--plots_dir", required=True, help="Directory where plots and derived CSV will be saved")
    args = p.parse_args()

    # root folder
    os.makedirs(args.plots_dir, exist_ok=True)

    # ALL results go in here
    inner_plots_dir = os.path.join(args.plots_dir, "plots")
    os.makedirs(inner_plots_dir, exist_ok=True)

    movie_stem = os.path.splitext(os.path.basename(args.summary))[0].replace("post_processed_", "").replace("_summary", "")

    # 1) Load summary CSV
    df = pd.read_csv(args.summary)
    print(f"[INFO] Loaded summary CSV with {len(df)} rows")

    # 2) Compute quality score (per-row)
    df['quality_score'] = compute_quality_score(df)

    # 3) Precision / F1 score definitions
    # NOTE: Without an external ground-truth 'recall' we treat support_fraction as a proxy for precision.
    # If you have true positives/negatives elsewhere, replace the following computations.
    if 'support_fraction' in df.columns:
        df['precision_score'] = df['support_fraction'].fillna(0.0).astype(float)
        # Assuming recall ~= support_fraction (no independent recall available), F1 becomes equal to support_fraction.
        # Keep this explicit so you can replace recall when true recall is available.
        df['f1_score'] = df['precision_score']  # harmonic_mean(precision, recall) with recall=precision => same value
    else:
        df['precision_score'] = 0.0
        df['f1_score'] = 0.0

    # 4) Compute aggregate metrics
    metrics = {}
    metrics['total_number_of_divisions'] = int(len(df))

    # Quality stats
    metrics['quality_mean'] = float(np.nanmean(df['quality_score'])) if 'quality_score' in df.columns else np.nan
    metrics['quality_median'] = float(np.nanmedian(df['quality_score'])) if 'quality_score' in df.columns else np.nan
    metrics['quality_std'] = float(np.nanstd(df['quality_score'])) if 'quality_score' in df.columns else np.nan
    metrics['quality_min'] = float(np.nanmin(df['quality_score'])) if 'quality_score' in df.columns else np.nan
    metrics['quality_max'] = float(np.nanmax(df['quality_score'])) if 'quality_score' in df.columns else np.nan

    # Precision / F1 summary (mean)
    metrics['precision_mean'] = float(np.nanmean(df['precision_score'])) if 'precision_score' in df.columns else np.nan
    metrics['f1_mean'] = float(np.nanmean(df['f1_score'])) if 'f1_score' in df.columns else np.nan

    # Best model for the movie (most common dominant_model)
    best_model = None
    if 'dominant_model' in df.columns:
        try:
            best_model = df['dominant_model'].dropna().mode().iloc[0]
            metrics['best_model_for_movie'] = str(best_model)
            metrics['best_model_count'] = int((df['dominant_model'] == best_model).sum())
        except Exception:
            metrics['best_model_for_movie'] = ""
            metrics['best_model_count'] = 0
    else:
        metrics['best_model_for_movie'] = ""
        metrics['best_model_count'] = 0

    # 5) Save aggregate metrics CSV (small)
    out_metrics_csv = os.path.join(inner_plots_dir, f"{movie_stem}_movie_metrics_summary.csv")
    save_metrics_csv(metrics, out_metrics_csv)

    # 6) Plots for distributions (note: pos_std and angle_std_deg histograms removed per request)
    plot_distributions(df, inner_plots_dir, movie_stem)

    # 7) Quality plots & scatter
    plot_quality_and_scatter(df, inner_plots_dir, movie_stem)

    # 8) Division timecourse scatter+errorbar plots
    plot_division_timecourses(df, inner_plots_dir, movie_stem)

    # Note: Per request, we do NOT save:
    #  - flow diagram
    #  - pos_std histogram
    #  - angle_std_deg histogram
    #  - post_processed_{movie}_summary_with_quality.csv
    #  - preview TIFF frame 0
    # Those files are intentionally not generated/saved here.

if __name__ == "__main__":
    main()
