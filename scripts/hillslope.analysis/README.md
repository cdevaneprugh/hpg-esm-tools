# OSBS Hillslope Analysis

Plotting scripts for CTSM hillslope hydrology cases at OSBS, using
column-level (h1 stream) and gridcell (h0 stream) history output.

## Layout assumptions

OSBS cases use a single-aspect layout: 1 aspect, N HAND elevation bins
(+ lake column in Phase E.5+ cases) + 1 bareground column. Scripts
auto-detect columns from the NetCDF (`hillslope_index > 0 & < 9000`;
the rest is bareground) and sort by `hillslope_elev`. Zone grouping:
TAI < 0.5 m, Transition 0.5–5 m, Upland ≥ 5 m HAND. Averages are
area-weighted via `cols1d_wtgcell`.

Case definitions and scientific context live in `swenson/`
(`CLAUDE.md`, `STATUS.md`, `phases/`).

## Batch workflow

```bash
sbatch generate_case_plots.sh   # all plots for the OSBS branch cases:
                                # concat archive h0/h1 → bin annual → plot → plots/<case>/
```

Case list and archive paths are hardcoded at the top of the script.
Output lands in `plots/` (gitignored).

`generate_all_plots.py` is the local equivalent: runs the six main
plotters over pre-binned files in `data/` (gitignored; expects
`combined_h0_20yr.nc`, `combined_h1.nc`, `combined_h1_1yr.nc`,
`combined_h1_20yr.nc`). Edit its `VARIABLES` list to change coverage.

## Binning

```bash
./bin_temporal.sh <input.nc> <output.nc> [--years=N]   # NCO N-year averages (default 1)
```

Note: `mcdate` in binned files is the bin center, not the end.

## Plotting scripts

All take positional args; run with `-h`/no args for details.

| Script | Usage | Input |
|--------|-------|-------|
| `plot_timeseries_full.py` | `<in.nc> <out.png> <VAR>` — full-span gridcell time series | h0 20-yr binned |
| `plot_timeseries_last20.py` | `<in.nc> <out.png> <VAR> [--years=N]` — last N yr by HAND zone + per-column | h1 annual binned |
| `plot_col_timeseries.py` | `<in.nc> <out.png> <VAR>` — full-span column series by Lake/FZ/Upland group (Phase E.5+ files) | h1 annual binned |
| `plot_zwt_hillslope_profile.py` | `<in.nc> <out.png>` — water table vs elevation, early/recent panels | h1 binned |
| `plot_hillslope_cross_section.py` | `<in.nc> <out.png>` — filled cross-section (surface/water table/ponding) | h1 binned |
| `plot_tai_heatmap.py` | `<in.nc> <out.png> [VAR]` — time × HAND-elevation heatmap (default ZWT) | h1 annual binned |
| `plot_carbon_water_coupling.py` | `<in.nc> <out.png> [--carbon-var=TOTSOMC]` — ZWT vs carbon at 3 positions | h1 annual binned |
| `plot_vr_profile.py` | `<in.nc> <out.png> <VAR>` — depth profiles (needs a `levsoi`/`levgrnd`/`levdcmp` var; current cases emit none) | h1 binned |
| `get_gridcell.py` | no args — print 0.9x1.25° grid indices for OSBS (edit constants for other sites) | — |

`archive/` holds superseded 4-aspect-era plotters.

## Dependencies

`ctsm` conda env (repo-root `environment.yml`): Python 3 + xarray /
numpy / matplotlib, NCO tools.
