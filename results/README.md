# Experiment results

This folder holds every number reported in the paper. There is one workbook per task with every method, the eight agreement indices, the recorded
clustering time and the parameter string of each run, plus:

| file | contents |
|---|---|
| `P-value.xlsx` | paired comparisons |
| `Synthetic.xlsx` | the eight artificial benchmarks |
| `Ablation.xlsx` | the underlying values of every ablation figure and table, one sheet per analysis (guide below) |

## `Ablation.xlsx`

Task identifiers are `ecoli`, `ge_banknote`, `ge_htru2`, `ge_rice`, `ionosphere`, `libras`,
`pendigits`, `wdbc`, `wine` and `stl10`. ARI values are stored at full
precision.

| sheet | content | columns |
|---|---|---|
| `Geometry_50seeds` | ARI of the Full, Spherical and Without-ellipsoids variants, 10 tasks x 3 variants x 50 seeds | `dataName`, `variant`, `seed_index` (0-49), `ari` |
| `Component_controls` | paired contrasts over tasks: mean ARI difference with its 95% interval, exact two-sided Wilcoxon probability, Holm-adjusted probability and matched-pairs rank-biserial effect size | `family`, `contrast`, `n_pairs`, `mean_delta_ARI`, `ci_low`, `ci_high`, `p_exact`, `p_holm`, `rank_biserial` |
| `OAS_scan` | ARI for 12 values of the OAS shrinkage multiplier, 10 tasks x 12 values (one run each) | `dataName`, `oas_multiplier`, `ari`, `k_pred`, `k_ref` |
| `AlphaN_sweep` | neighborhood-control sweep (25 log-spaced grid points per task, 26 on Wine), alpha_M held at the task's adopted value; one row per grid point, aggregated over 21 seeds (fewer on three HTRU2 points) | `dataset`, `alpha_n`, `ari_mean`, `ari_sd`, `ari_ci_lo`, `ari_ci_hi` (95% interval over seeds), `n_seed` |
| `AlphaM_sweep` | graph- and shape-side sweep (21-22 grid points per task, the adopted value included), alpha_N held at the task's adopted value; one run per cell | `dataset`, `alpha_m`, `ari` |
| `Transfer_matrix` | ARI on a target task when the adopted (alpha_N, alpha_M) pair of a source task is applied, 10 x 10 runs | `target_task`, `source_task`, `ari`, `is_own_controls` (true on the diagonal) |
| `Global_pair_scan` | nine candidate pairs, each applied to every task, 9 x 10 runs (one run per cell): alpha_N log-spaced from 2.25e-4 to 1e-2 with alpha_M = 0.05; the pair at 1e-2 is the Wine pair, so its ten values equal the Wine column of `Transfer_matrix` | `pair_id`, `alpha_n`, `alpha_m`, `task`, `ari` |

The per-task workbooks have one row per method (`AEDPC-Full` is AEDPC) with `run_time`
(seconds, mean of three runs) and the eight indices `rand_score`, `adjusted_rand_score`,
`adjusted_mutual_info_score`, `normalized_mutual_info_score`, `homogeneity_score`,
`completeness_score`, `v_measure_score` and `fowlkes_mallows_score`. `P-value.xlsx` holds the
exact two-sided Wilcoxon probabilities of the nine-dataset paired comparison against each
external method, for the eight indices and for clustering time; `Synthetic.xlsx` lists reference and predicted cluster counts and the eight
indices of the artificial benchmarks.
