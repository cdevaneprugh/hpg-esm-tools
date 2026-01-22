#!/usr/bin/env python
"""
Validate our HillslopeGrid implementation against Swenson's published dataset.

This script:
1. Loads MERIT DEM sample (n30w095 - Mississippi region)
2. Runs our HillslopeGrid implementation
3. Loads Swenson's global hillslope dataset
4. Extracts the same region and compares statistics

The comparison is at the distribution level since:
- Our implementation: raw HAND/DTND per 90m pixel
- Swenson's data: discretized 16-column hillslope parameters at 0.9° x 1.25°
"""

import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Add pysheds fork to path
PYSHEDS_PATH = '/blue/gerber/cdevaneprugh/pysheds_fork'
sys.path.insert(0, PYSHEDS_PATH)

from pysheds.hillslope import HillslopeGrid

# Data paths
DATA_DIR = '/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data'
MERIT_DEM = os.path.join(DATA_DIR, 'MERIT_DEM_sample', 'n30w095_dem.tif')
SWENSON_NC = os.path.join(DATA_DIR, 'hillslopes_0.9x1.25_c240416.nc')

# MERIT sample region: N30-N35, W95-W90
REGION = {'lat_min': 30, 'lat_max': 35, 'lon_min': -95, 'lon_max': -90}


def run_our_implementation(dem_path: str, acc_threshold: int = 500) -> dict:
    """
    Run our HillslopeGrid on the MERIT DEM.

    Parameters
    ----------
    dem_path : str
        Path to MERIT DEM GeoTiff
    acc_threshold : int
        Accumulation threshold for channel identification

    Returns
    -------
    dict with HAND, DTND, hillslope arrays and statistics
    """
    print("Loading MERIT DEM...")
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    grid = HillslopeGrid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    print(f"  DEM shape: {dem.shape}")
    print(f"  DEM range: [{np.nanmin(dem):.1f}, {np.nanmax(dem):.1f}] m")

    # Process DEM
    print("Processing DEM (fill depressions, resolve flats)...")
    filled_dem = grid.fill_depressions(dem)
    inflated_dem = grid.resolve_flats(filled_dem)

    print("Computing flow direction and accumulation...")
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap, routing='d8')
    acc = grid.accumulation(fdir, dirmap=dirmap, routing='d8')

    # Create channel mask
    print(f"Creating channel mask (threshold={acc_threshold})...")
    channel_mask = acc > acc_threshold
    n_channel = np.sum(np.array(channel_mask))
    print(f"  Channel cells: {n_channel}")

    # Run our methods
    print("Running create_channel_mask...")
    channel_mask_out, channel_id, bank_mask = grid.create_channel_mask(
        fdir, channel_mask, dirmap=dirmap
    )

    print("Running compute_hand_extended...")
    hand, dtnd, drainage_id = grid.compute_hand_extended(
        fdir, inflated_dem, channel_mask_out, channel_id, dirmap=dirmap
    )

    print("Running compute_hillslope...")
    hillslope = grid.compute_hillslope(
        fdir, channel_mask_out, bank_mask, dirmap=dirmap
    )

    print("Running river_network_length_and_slope...")
    network_stats = grid.river_network_length_and_slope(
        fdir, inflated_dem, acc, channel_mask, dirmap=dirmap
    )

    # Convert to arrays
    hand_arr = np.array(hand)
    dtnd_arr = np.array(dtnd)
    hillslope_arr = np.array(hillslope)

    # Compute statistics (excluding nodata)
    valid = ~np.isnan(hand_arr) & (hand_arr > 0)

    results = {
        'hand': hand_arr,
        'dtnd': dtnd_arr,
        'hillslope': hillslope_arr,
        'network_stats': network_stats,
        'hand_mean': np.mean(hand_arr[valid]) if np.any(valid) else np.nan,
        'hand_median': np.median(hand_arr[valid]) if np.any(valid) else np.nan,
        'hand_max': np.max(hand_arr[valid]) if np.any(valid) else np.nan,
        'dtnd_mean': np.mean(dtnd_arr[valid]) if np.any(valid) else np.nan,
        'dtnd_median': np.median(dtnd_arr[valid]) if np.any(valid) else np.nan,
        'dtnd_max': np.max(dtnd_arr[valid]) if np.any(valid) else np.nan,
    }

    return results


def load_swenson_region(nc_path: str, region: dict) -> dict:
    """
    Load Swenson's dataset and extract the matching region.

    Parameters
    ----------
    nc_path : str
        Path to Swenson's NetCDF file
    region : dict
        Region bounds (lat_min, lat_max, lon_min, lon_max)

    Returns
    -------
    dict with hillslope parameters for the region
    """
    print(f"\nLoading Swenson's dataset...")
    ds = xr.open_dataset(nc_path)

    # Get coordinate arrays
    lon = ds['LONGXY'].values[0, :]  # lon varies along lsmlon
    lat = ds['LATIXY'].values[:, 0]  # lat varies along lsmlat

    print(f"  Global grid: {ds.dims['lsmlat']} x {ds.dims['lsmlon']}")
    print(f"  Lon range: [{lon.min():.1f}, {lon.max():.1f}]")
    print(f"  Lat range: [{lat.min():.1f}, {lat.max():.1f}]")

    # Find indices for our region
    # Note: lon in Swenson's data is 0-360, need to convert
    lon_min = region['lon_min'] + 360 if region['lon_min'] < 0 else region['lon_min']
    lon_max = region['lon_max'] + 360 if region['lon_max'] < 0 else region['lon_max']

    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    lat_idx = np.where((lat >= region['lat_min']) & (lat <= region['lat_max']))[0]

    print(f"  Region indices: lat[{lat_idx[0]}:{lat_idx[-1]+1}], lon[{lon_idx[0]}:{lon_idx[-1]+1}]")
    print(f"  Region size: {len(lat_idx)} x {len(lon_idx)} gridcells")

    # Extract region
    elevation = ds['hillslope_elevation'].values[:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    distance = ds['hillslope_distance'].values[:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    slope = ds['hillslope_slope'].values[:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    area = ds['hillslope_area'].values[:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    stream_slope = ds['hillslope_stream_slope'].values[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]

    ds.close()

    # Compute statistics (excluding zeros/nodata)
    valid_elev = elevation[elevation > 0]
    valid_dist = distance[distance > 0]
    valid_slope = slope[slope > 0]

    results = {
        'elevation': elevation,
        'distance': distance,
        'slope': slope,
        'area': area,
        'stream_slope': stream_slope,
        'elevation_mean': np.mean(valid_elev) if len(valid_elev) > 0 else np.nan,
        'elevation_median': np.median(valid_elev) if len(valid_elev) > 0 else np.nan,
        'elevation_max': np.max(valid_elev) if len(valid_elev) > 0 else np.nan,
        'distance_mean': np.mean(valid_dist) if len(valid_dist) > 0 else np.nan,
        'distance_median': np.median(valid_dist) if len(valid_dist) > 0 else np.nan,
        'distance_max': np.max(valid_dist) if len(valid_dist) > 0 else np.nan,
        'n_gridcells': len(lat_idx) * len(lon_idx),
    }

    return results


def compare_results(ours: dict, swenson: dict) -> None:
    """Print comparison of our results vs Swenson's."""

    print("\n" + "=" * 70)
    print("COMPARISON: Our Implementation vs Swenson's Published Data")
    print("=" * 70)

    print("\nNote: Direct comparison is approximate because:")
    print("  - Our data: raw 90m pixels")
    print("  - Swenson's: discretized 16-column parameters at 0.9° x 1.25°")
    print("  - Our HAND/DTND should have similar DISTRIBUTIONS to Swenson's elevation/distance")

    print("\n--- Height/Elevation Statistics ---")
    print(f"  Our HAND mean:      {ours['hand_mean']:.1f} m")
    print(f"  Swenson elev mean:  {swenson['elevation_mean']:.1f} m")
    print(f"  Our HAND median:    {ours['hand_median']:.1f} m")
    print(f"  Swenson elev median:{swenson['elevation_median']:.1f} m")
    print(f"  Our HAND max:       {ours['hand_max']:.1f} m")
    print(f"  Swenson elev max:   {swenson['elevation_max']:.1f} m")

    print("\n--- Distance Statistics ---")
    print(f"  Our DTND mean:      {ours['dtnd_mean']:.1f} m")
    print(f"  Swenson dist mean:  {swenson['distance_mean']:.1f} m")
    print(f"  Our DTND median:    {ours['dtnd_median']:.1f} m")
    print(f"  Swenson dist median:{swenson['distance_median']:.1f} m")
    print(f"  Our DTND max:       {ours['dtnd_max']:.1f} m")
    print(f"  Swenson dist max:   {swenson['distance_max']:.1f} m")

    print("\n--- River Network Statistics ---")
    ns = ours['network_stats']
    print(f"  Our network length: {ns['length']/1000:.1f} km")
    print(f"  Our network slope:  {ns['slope']*100:.3f}%")
    print(f"  Our main channel:   {ns['mch_length']/1000:.1f} km")
    print(f"  Swenson stream slope (region mean): {np.nanmean(swenson['stream_slope'])*100:.3f}%")

    print("\n--- Interpretation ---")
    # Check if statistics are in similar range
    elev_ratio = ours['hand_mean'] / swenson['elevation_mean'] if swenson['elevation_mean'] > 0 else np.nan
    dist_ratio = ours['dtnd_mean'] / swenson['distance_mean'] if swenson['distance_mean'] > 0 else np.nan

    print(f"  HAND/elevation ratio: {elev_ratio:.2f}")
    print(f"  DTND/distance ratio:  {dist_ratio:.2f}")

    if 0.5 < elev_ratio < 2.0 and 0.5 < dist_ratio < 2.0:
        print("\n  VALIDATION PASSED: Our results are in reasonable agreement with Swenson's")
        print("  (Differences expected due to resolution and discretization)")
    else:
        print("\n  VALIDATION WARNING: Large differences detected - investigate further")


def create_visualization(ours: dict, swenson: dict, output_path: str) -> None:
    """Create comparison visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Our HAND
    hand = ours['hand']
    hand[hand == 0] = np.nan
    im0 = axes[0, 0].imshow(hand, cmap='Blues', vmin=0, vmax=np.nanpercentile(hand, 99))
    axes[0, 0].set_title('Our HAND (90m)')
    plt.colorbar(im0, ax=axes[0, 0], label='Height (m)')

    # Our DTND
    dtnd = ours['dtnd']
    dtnd[dtnd == 0] = np.nan
    im1 = axes[0, 1].imshow(dtnd, cmap='Oranges', vmin=0, vmax=np.nanpercentile(dtnd, 99))
    axes[0, 1].set_title('Our DTND (90m)')
    plt.colorbar(im1, ax=axes[0, 1], label='Distance (m)')

    # Our hillslope classification
    hillslope = ours['hillslope']
    hillslope[hillslope == 0] = np.nan
    cmap = plt.colormaps.get_cmap('Set1').resampled(4)
    im2 = axes[0, 2].imshow(hillslope, cmap=cmap, vmin=0.5, vmax=4.5)
    axes[0, 2].set_title('Our Hillslope Class (90m)')
    cbar = plt.colorbar(im2, ax=axes[0, 2], ticks=[1, 2, 3, 4])
    cbar.ax.set_yticklabels(['Head', 'Right', 'Left', 'Channel'])

    # Swenson elevation (mean across columns)
    elev_mean = np.nanmean(swenson['elevation'], axis=0)
    elev_mean[elev_mean == 0] = np.nan
    im3 = axes[1, 0].imshow(elev_mean, cmap='Blues', vmin=0)
    axes[1, 0].set_title('Swenson Elevation (0.9° mean)')
    plt.colorbar(im3, ax=axes[1, 0], label='Height (m)')

    # Swenson distance (mean across columns)
    dist_mean = np.nanmean(swenson['distance'], axis=0)
    dist_mean[dist_mean == 0] = np.nan
    im4 = axes[1, 1].imshow(dist_mean, cmap='Oranges', vmin=0)
    axes[1, 1].set_title('Swenson Distance (0.9° mean)')
    plt.colorbar(im4, ax=axes[1, 1], label='Distance (m)')

    # Distribution comparison - HAND vs Swenson elevation
    ax5 = axes[1, 2]
    hand_valid = ours['hand'][~np.isnan(ours['hand']) & (ours['hand'] > 0)]
    elev_valid = swenson['elevation'][swenson['elevation'] > 0]

    ax5.hist(hand_valid, bins=50, alpha=0.5, label=f'Our HAND (n={len(hand_valid):,})', density=True)
    ax5.hist(elev_valid, bins=50, alpha=0.5, label=f'Swenson elev (n={len(elev_valid):,})', density=True)
    ax5.set_xlabel('Height above channel (m)')
    ax5.set_ylabel('Density')
    ax5.set_title('Distribution Comparison')
    ax5.legend()
    ax5.set_xlim(0, min(np.percentile(hand_valid, 99), np.percentile(elev_valid, 99)) * 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("SWENSON IMPLEMENTATION VALIDATION")
    print("Region: Mississippi (N30-N35, W95-W90)")
    print("=" * 70)

    # Run our implementation
    print("\n--- Running Our Implementation ---")
    ours = run_our_implementation(MERIT_DEM)

    # Load Swenson's data
    print("\n--- Loading Swenson's Published Data ---")
    swenson = load_swenson_region(SWENSON_NC, REGION)

    # Compare
    compare_results(ours, swenson)

    # Visualize
    output_path = os.path.join(DATA_DIR, 'validation_comparison.png')
    create_visualization(ours, swenson, output_path)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
