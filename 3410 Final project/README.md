# Population vs Crashes analysis

This small Python tool analyzes whether states that gained population between 2021 and 2022 also saw increases in crashes (the dataset is `3410 Crash vs Population - Sheet2.csv`).

Assumption
- The CSV doesn't have a dedicated "Commercial accidents" column; by default the script uses the "Total Crashes '21" / "Total Crashes '22" columns as the crash metric. If you do have a separate column for commercial crashes, tell me the exact column name and I will update the script to use it instead.

Files
- `analysis.py` — main analysis script.
- `requirements.txt` — Python package list.

Usage
- Install requirements:

```cmd
python -m pip install -r requirements.txt
```

- Run the analysis (default uses percent changes):

```cmd
python analysis.py "3410 Crash vs Population - Sheet2.csv" --outdir results
```

- To analyze absolute changes instead of percent changes:

```cmd
python analysis.py "3410 Crash vs Population - Sheet2.csv" --outdir results --no-percent
```

Outputs (in `results/`)
- `merged_changes.csv` — per-state X (population change) and Y (crash change) used in analysis.
- `correlation_summary.txt` — Pearson/Spearman and regression results.
- `pop_vs_crash.png` — scatterplot with regression line and top state annotations.

Next steps / customization
- If you want a specific column used (for example, a dedicated commercial accident column), tell me the exact column name and I will update `analysis.py`.
- I can also add a small Jupyter notebook that walks through the results interactively.
