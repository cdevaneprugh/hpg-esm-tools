#!/usr/bin/env python
"""
Stage 4: Compare to Swenson's Published Hillslope Data

Compare our computed hillslope parameters to Swenson & Lawrence (2025)
published global dataset.

This script:
1. Loads Swenson's published hillslope data (or downloads if not present)
2. Extracts gridcells overlapping with our processed region
3. Computes comparison metrics (MAE, correlation, etc.)
4. Generates comparison plots

Published data:
- DOI: https://doi.org/10.5065/w01j-y441
- File: hillslopes_0.9x1.25_c240416.nc
- Resolution: 0.9° x 1.25° (288 x 192 gridcells globally)

Data paths:
- Stage 3 output: swenson/output/stage3/stage3_hillslope_params.json
- Published data: swenson/data/hillslopes_0.9x1.25_c240416.nc
- Output: swenson/output/stage4/

Expected runtime: ~5 minutes
"""

import os
import sys
import time
import json
import numpy as np

try:
    import netCDF4 as nc

    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Configuration
STAGE3_RESULTS = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/stage3/stage3_hillslope_params.json"
OUTPUT_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/stage4"

# Published data paths (try multiple locations)
PUBLISHED_DATA_PATHS = [
    "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/hillslopes_0.9x1.25_c240416.nc",
]

# Download URL for published data
PUBLISHED_DATA_URL = "https://doi.org/10.5065/w01j-y441"


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def find_published_data() -> str | None:
    """Find the published hillslope data file."""
    for path in PUBLISHED_DATA_PATHS:
        if os.path.exists(path):
            return path
    return None


def download_instructions() -> str:
    """Return instructions for downloading the published data."""
    return f"""
The published hillslope data is not found locally.

To download:
1. Visit: {PUBLISHED_DATA_URL}
2. Download: hillslopes_0.9x1.25_c240416.nc
3. Place in: {PUBLISHED_DATA_PATHS[0]}

Or use wget (if direct download link is available):
  mkdir -p /blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data
  cd /blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data
  # Download from NCAR RDA (requires account)
"""


def load_published_data(filepath: str, lon_range: tuple, lat_range: tuple) -> dict:
    """
    Load published hillslope data and extract gridcells within the specified region.

    Parameters
    ----------
    filepath : str
        Path to hillslopes NetCDF file
    lon_range : tuple
        (lon_min, lon_max) in degrees
    lat_range : tuple
        (lat_min, lat_max) in degrees

    Returns
    -------
    dict with gridcell parameters
    """
    ds = nc.Dataset(filepath, "r")

    # Get coordinates - published data uses LONGXY/LATIXY (2D arrays)
    if "LONGXY" in ds.variables:
        # Published data format: 2D arrays (lsmlat, lsmlon)
        longxy = ds.variables["LONGXY"][:]
        latixy = ds.variables["LATIXY"][:]
        # Extract 1D coordinate arrays
        lon = longxy[0, :]  # All columns at first row
        lat = latixy[:, 0]  # All rows at first column
    else:
        # Fallback to 1D coordinates
        lon = ds.variables["lon"][:]
        lat = ds.variables["lat"][:]

    # Handle longitude convention (0-360 vs -180 to 180)
    lon_min, lon_max = lon_range
    if lon_min < 0:
        lon_min += 360
    if lon_max < 0:
        lon_max += 360

    # Handle latitude range (may be reversed)
    lat_min = min(lat_range)
    lat_max = max(lat_range)

    # Find gridcells in region
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    lat_mask = (lat >= lat_min) & (lat <= lat_max)

    lon_indices = np.where(lon_mask)[0]
    lat_indices = np.where(lat_mask)[0]

    print(f"  Published data lon range: [{lon.min():.2f}, {lon.max():.2f}]")
    print(f"  Published data lat range: [{lat.min():.2f}, {lat.max():.2f}]")
    print(
        f"  Query region: lon [{lon_min:.2f}, {lon_max:.2f}], lat [{lat_min:.2f}, {lat_max:.2f}]"
    )
    print(
        f"  Matching gridcells: {len(lon_indices)} × {len(lat_indices)} = {len(lon_indices) * len(lat_indices)}"
    )

    # Variable names in published data
    # Typical names: nhillcolumns, hillslope_index, column_index,
    # col_h (height), col_d (distance), col_a (area), col_slope, col_aspect, col_width

    # List available variables
    print("\n  Available variables:")
    for name in ds.variables:
        var = ds.variables[name]
        if hasattr(var, "long_name"):
            print(f"    {name}: {var.shape} - {var.long_name}")
        else:
            print(f"    {name}: {var.shape}")

    result = {
        "lon": lon[lon_indices],
        "lat": lat[lat_indices],
        "lon_indices": lon_indices,
        "lat_indices": lat_indices,
        "variables": {},
    }

    # Extract hillslope parameters for the region
    # Structure depends on file format
    param_names = [
        # Published Swenson data variable names
        "hillslope_elevation",
        "hillslope_distance",
        "hillslope_area",
        "hillslope_slope",
        "hillslope_aspect",
        "hillslope_width",
        # Alternative variable names
        "col_h",
        "col_d",
        "col_a",
        "col_slope",
        "col_aspect",
        "col_width",
        "hand",
        "dtnd",
        "area",
        "slope",
        "aspect",
        "width",
    ]

    for name in param_names:
        if name in ds.variables:
            var = ds.variables[name]
            # Extract subset based on dimensions
            if var.ndim == 3:  # (column, lat, lon) or similar
                data = var[:, lat_indices, :][:, :, lon_indices]
            elif var.ndim == 2:  # (lat, lon)
                data = var[lat_indices, :][:, lon_indices]
            else:
                data = var[:]
            result["variables"][name] = data

    ds.close()
    return result


def compare_parameters(our_params: dict, published: dict) -> dict:
    """
    Compare our computed parameters to published values.

    Returns dict with comparison metrics.
    """
    metrics = {}

    our_elements = our_params["elements"]
    pub_vars = published["variables"]

    # Map our parameter names to published variable names
    param_map = {
        "height": ["hillslope_elevation", "col_h", "hand"],
        "distance": ["hillslope_distance", "col_d", "dtnd"],
        "area": ["hillslope_area", "col_a", "area"],
        "slope": ["hillslope_slope", "col_slope", "slope"],
        "aspect": ["hillslope_aspect", "col_aspect", "aspect"],
        "width": ["hillslope_width", "col_width", "width"],
    }

    for our_name, pub_names in param_map.items():
        # Find which published variable name is available
        pub_name = None
        for pn in pub_names:
            if pn in pub_vars:
                pub_name = pn
                break

        if pub_name is None:
            print(f"  Warning: No published data for '{our_name}'")
            continue

        # Extract our values
        our_values = np.array([elem[our_name] for elem in our_elements])

        # Get published values (average over region)
        pub_data = pub_vars[pub_name]
        pub_values = np.nanmean(pub_data, axis=tuple(range(1, pub_data.ndim)))

        # Compute metrics
        # Handle different array sizes
        n_compare = min(len(our_values), len(pub_values))

        if n_compare > 0:
            our_subset = our_values[:n_compare]
            pub_subset = pub_values[:n_compare]

            # Filter out zeros and nans
            valid = (
                (our_subset > 0)
                & (pub_subset > 0)
                & np.isfinite(our_subset)
                & np.isfinite(pub_subset)
            )

            if np.sum(valid) > 1:
                our_valid = our_subset[valid]
                pub_valid = pub_subset[valid]

                mae = np.mean(np.abs(our_valid - pub_valid))
                rmse = np.sqrt(np.mean((our_valid - pub_valid) ** 2))
                corr = np.corrcoef(our_valid, pub_valid)[0, 1]
                rel_error = np.mean(np.abs(our_valid - pub_valid) / pub_valid) * 100

                metrics[our_name] = {
                    "our_mean": float(np.mean(our_valid)),
                    "our_std": float(np.std(our_valid)),
                    "pub_mean": float(np.mean(pub_valid)),
                    "pub_std": float(np.std(pub_valid)),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "correlation": float(corr),
                    "relative_error_pct": float(rel_error),
                    "n_compared": int(np.sum(valid)),
                }

                print(f"\n  {our_name}:")
                print(
                    f"    Our mean: {np.mean(our_valid):.2f}, Published mean: {np.mean(pub_valid):.2f}"
                )
                print(f"    MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                print(f"    Correlation: {corr:.3f}")
                print(f"    Relative error: {rel_error:.1f}%")

    return metrics


def create_comparison_plots(
    our_params: dict, published: dict, metrics: dict, output_dir: str
) -> None:
    """Generate comparison plots."""
    if not HAS_MATPLOTLIB:
        print("  Skipping plots (matplotlib not available)")
        return

    print_section("Generating Comparison Plots")

    our_elements = our_params["elements"]

    # Bar chart comparing our parameters to published
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    param_names = ["height", "distance", "area", "slope", "aspect", "width"]
    param_units = ["m", "m", "m²", "m/m", "deg", "m"]

    for idx, (name, unit) in enumerate(zip(param_names, param_units)):
        ax = axes.flat[idx]

        our_values = [elem[name] for elem in our_elements]

        if name in metrics:
            m = metrics[name]
            ax.bar(0, m["our_mean"], yerr=m["our_std"], label="Our results", alpha=0.7)
            ax.bar(1, m["pub_mean"], yerr=m["pub_std"], label="Published", alpha=0.7)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Our", "Published"])
            ax.set_title(
                f"{name.capitalize()}\nCorr={m['correlation']:.2f}, RelErr={m['relative_error_pct']:.1f}%"
            )
        else:
            ax.bar(range(len(our_values)), our_values, alpha=0.7)
            ax.set_xlabel("Element")
            ax.set_title(f"{name.capitalize()} (no comparison)")

        ax.set_ylabel(f"{name.capitalize()} ({unit})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage4_parameter_comparison.png"), dpi=150)
    plt.close()

    print(f"  Saved comparison plots to {output_dir}")


def main():
    """Main processing function."""
    start_time = time.time()

    print_section("Stage 4: Compare to Published Data")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not HAS_NETCDF:
        print("ERROR: netCDF4 not available, cannot read published data")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 1: Load Stage 3 results
    # -------------------------------------------------------------------------
    print_section("Step 1: Loading Stage 3 Results")

    if not os.path.exists(STAGE3_RESULTS):
        print(f"ERROR: Stage 3 results not found at {STAGE3_RESULTS}")
        print("Please run Stage 3 first.")
        sys.exit(1)

    with open(STAGE3_RESULTS) as f:
        our_params = json.load(f)

    print(f"  Loaded {len(our_params['elements'])} hillslope elements")
    print(f"  Region: {our_params['metadata']['region_shape']}")
    print(f"  Lon range: {our_params['metadata']['lon_range']}")
    print(f"  Lat range: {our_params['metadata']['lat_range']}")

    # -------------------------------------------------------------------------
    # Step 2: Find and load published data
    # -------------------------------------------------------------------------
    print_section("Step 2: Loading Published Data")

    published_path = find_published_data()

    if published_path is None:
        print("WARNING: Published hillslope data not found.")
        print(download_instructions())

        # Create placeholder output
        comparison_results = {
            "status": "published_data_not_found",
            "instructions": download_instructions(),
            "our_params_summary": {
                "n_elements": len(our_params["elements"]),
                "region_lon": our_params["metadata"]["lon_range"],
                "region_lat": our_params["metadata"]["lat_range"],
            },
        }

        # Save partial results
        json_path = os.path.join(OUTPUT_DIR, "stage4_results.json")
        with open(json_path, "w") as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\n  Saved: {json_path}")

        # Create summary of our results without comparison
        print_section("Our Results Summary (No Comparison Available)")

        summary_path = os.path.join(OUTPUT_DIR, "stage4_summary.txt")
        with open(summary_path, "w") as f:
            f.write("Stage 4: Comparison Results\n")
            f.write("=" * 60 + "\n\n")
            f.write("STATUS: Published data not found\n\n")
            f.write(download_instructions())
            f.write("\n\nOur computed hillslope parameters:\n")
            f.write("-" * 40 + "\n")

            for elem in our_params["elements"]:
                f.write(f"\n{elem['aspect_name']} Bin {elem['hand_bin'] + 1}:\n")
                f.write(f"  Height: {elem['height']:.1f} m\n")
                f.write(f"  Distance: {elem['distance']:.0f} m\n")
                f.write(f"  Area: {elem['area'] / 1e6:.2f} km²\n")
                f.write(f"  Slope: {elem['slope']:.4f}\n")
                f.write(f"  Aspect: {elem['aspect']:.1f}°\n")
                f.write(f"  Width: {elem['width']:.0f} m\n")

        print(f"  Saved: {summary_path}")

        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.1f} seconds")
        print(
            "\nTo complete comparison, download the published data and re-run Stage 4."
        )
        return

    print(f"  Found published data: {published_path}")

    # Load published data for our region
    published = load_published_data(
        published_path,
        lon_range=tuple(our_params["metadata"]["lon_range"]),
        lat_range=tuple(our_params["metadata"]["lat_range"]),
    )

    # -------------------------------------------------------------------------
    # Step 3: Compare parameters
    # -------------------------------------------------------------------------
    print_section("Step 3: Computing Comparison Metrics")

    metrics = compare_parameters(our_params, published)

    # -------------------------------------------------------------------------
    # Step 4: Generate plots
    # -------------------------------------------------------------------------
    create_comparison_plots(our_params, published, metrics, OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # Step 5: Save results
    # -------------------------------------------------------------------------
    print_section("Step 5: Saving Results")

    comparison_results = {
        "status": "comparison_complete",
        "published_data_path": published_path,
        "our_params_summary": {
            "n_elements": len(our_params["elements"]),
            "region_lon": our_params["metadata"]["lon_range"],
            "region_lat": our_params["metadata"]["lat_range"],
        },
        "published_summary": {
            "n_lon_gridcells": len(published["lon"]),
            "n_lat_gridcells": len(published["lat"]),
            "lon_range": [float(published["lon"].min()), float(published["lon"].max())],
            "lat_range": [float(published["lat"].min()), float(published["lat"].max())],
        },
        "metrics": metrics,
    }

    json_path = os.path.join(OUTPUT_DIR, "stage4_results.json")
    with open(json_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Text summary
    summary_path = os.path.join(OUTPUT_DIR, "stage4_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Stage 4: Comparison Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Published data: {published_path}\n")
        f.write(f"Our region: lon {our_params['metadata']['lon_range']}, ")
        f.write(f"lat {our_params['metadata']['lat_range']}\n\n")

        f.write("Comparison Metrics:\n")
        f.write("-" * 40 + "\n")

        for param, m in metrics.items():
            f.write(f"\n{param.upper()}:\n")
            f.write(f"  Our mean ± std: {m['our_mean']:.2f} ± {m['our_std']:.2f}\n")
            f.write(
                f"  Published mean ± std: {m['pub_mean']:.2f} ± {m['pub_std']:.2f}\n"
            )
            f.write(f"  MAE: {m['mae']:.2f}\n")
            f.write(f"  RMSE: {m['rmse']:.2f}\n")
            f.write(f"  Correlation: {m['correlation']:.3f}\n")
            f.write(f"  Relative error: {m['relative_error_pct']:.1f}%\n")

        f.write(f"\nTotal processing time: {time.time() - start_time:.1f} seconds\n")

    print(f"  Saved: {summary_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section("Stage 4 Complete")

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.1f} seconds")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")

    print("\nKey comparison results:")
    for param, m in metrics.items():
        status = "GOOD" if m["relative_error_pct"] < 20 else "CHECK"
        print(
            f"  {param}: Corr={m['correlation']:.2f}, RelErr={m['relative_error_pct']:.1f}% [{status}]"
        )


if __name__ == "__main__":
    main()
