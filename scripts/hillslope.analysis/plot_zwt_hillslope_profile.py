#!/usr/bin/env python3
"""
Plot water table depth (ZWT) against hillslope elevation profile

Shows two 20-year time periods: early (years 1-20) and recent (last 20 years)
water table positions along hillslope.

ZWT = depth of water table below surface (m)
Water table elevation = hillslope_elev - ZWT

Usage:
    python3 plot_zwt_hillslope_profile.py [h1_20yr_file] [output_file] [hillslope]

Example:
    python3 plot_zwt_hillslope_profile.py data/combined_h1_20yr.nc plots/zwt_profile.png North
"""

import sys
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_zwt_profile(nc_file, output_file, hillslope='North'):
    """
    Plot ZWT vs hillslope profile for 3 time periods (20-year bins)

    Parameters:
    -----------
    nc_file : str
        Path to h1_20yr NetCDF file
    output_file : str
        Output PNG filename
    hillslope : str
        Hillslope to plot ('North', 'East', 'South', 'West')
    """

    # Load data
    ds = xr.open_dataset(nc_file, decode_times=False)

    zwt = ds['ZWT'].values
    elev = ds['hillslope_elev'].values
    dist = ds['hillslope_distance'].values
    mcdate = ds['mcdate'].values
    hillslope_index = ds['hillslope_index'].values

    # Get years from mcdate
    years = mcdate // 10000

    # Select hillslope columns
    hillslope_map = {'North': 1, 'East': 2, 'South': 3, 'West': 4}
    h_idx = hillslope_map[hillslope]

    # Get columns for this hillslope (4 columns per hillslope)
    h_cols = np.where(hillslope_index == h_idx)[0]

    # Sort by distance (outlet to ridge)
    sort_order = np.argsort(dist[h_cols])
    h_cols = h_cols[sort_order]

    h_elev = elev[h_cols]
    h_dist = dist[h_cols]

    # Select two time indices: first and last
    n_times = zwt.shape[0]
    time_indices = [0, n_times - 1]

    # Get year ranges for each bin (each bin is 20 years)
    # mcdate represents the average/center of the bin due to ncra averaging
    year_ranges = []
    for t_idx in time_indices:
        bin_center_year = years[t_idx]
        bin_start_year = bin_center_year - 9   # 20 year bin centered at mcdate
        bin_end_year = bin_center_year + 10
        year_ranges.append((bin_start_year, bin_end_year))

    time_labels = [
        f'Years {year_ranges[0][0]}-{year_ranges[0][1]} (Early, 20-yr avg)',
        f'Years {year_ranges[1][0]}-{year_ranges[1][1]} (Recent, 20-yr avg)'
    ]

    # Create figure with 2 snapshots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for ax, t_idx, label in zip(axes, time_indices, time_labels):
        # Water table elevation
        wt_elev = h_elev - zwt[t_idx, h_cols]

        # Plot surface elevation
        ax.plot(h_dist, h_elev, 'o-', linewidth=2.5, markersize=10,
                color='brown', label='Hillslope Surface', zorder=3)

        # Plot water table elevation
        ax.plot(h_dist, wt_elev, 'o-', linewidth=2.5, markersize=10,
                color='blue', label='Water Table', zorder=2)

        # Fill between surface and water table (unsaturated zone)
        ax.fill_between(h_dist, wt_elev, h_elev,
                         alpha=0.3, color='tan', label='Unsaturated Zone')

        # Fill below water table (saturated zone)
        ax.fill_between(h_dist, -10, wt_elev,
                         alpha=0.3, color='lightblue', label='Saturated Zone')

        # Stream level reference
        ax.axhline(0, color='darkblue', linestyle='--', linewidth=2,
                   label='Stream Level', zorder=1)

        # Add labels at each column
        pos_names = ['Outlet', 'Lower', 'Upper', 'Ridge']
        for i, (d, e, wt) in enumerate(zip(h_dist, h_elev, wt_elev)):
            ax.text(d, e + 0.5, pos_names[i], ha='center', fontsize=9, fontweight='bold')

            # Show ZWT value
            zwt_val = zwt[t_idx, h_cols[i]]
            ax.text(d, (e + wt)/2, f'ZWT={zwt_val:.1f}m',
                    ha='center', fontsize=8, style='italic',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Formatting
        ax.set_ylabel('Elevation above Stream (m)', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_ylim(-10, 9)

    axes[-1].set_xlabel('Distance from Stream (m)', fontsize=11)

    # Overall title
    first_year = years[0]
    last_year = years[-1]
    total_years = last_year - first_year

    fig.suptitle(f'Water Table Depth (ZWT) vs Hillslope Profile - {hillslope} Hillslope\n' +
                 f'Simulation Years {first_year}-{last_year} ({total_years} years total)',
                 fontsize=13, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    # Print summary info
    print()
    print("Time periods plotted:")
    for i, (t_idx, label) in enumerate(zip(time_indices, time_labels)):
        print(f"  {i+1}. {label}")
        print(f"     ZWT values: Outlet={zwt[t_idx, h_cols[0]]:.1f}m, Ridge={zwt[t_idx, h_cols[3]]:.1f}m")

    ds.close()

if __name__ == '__main__':
    # Default parameters
    nc_file = './data/combined_h1_20yr.nc'
    output_file = './plots/zwt_hillslope_profile.png'
    hillslope = 'North'

    # Check command line arguments
    if len(sys.argv) >= 2:
        nc_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    if len(sys.argv) >= 4:
        hillslope = sys.argv[3]

    print(f"Input file: {nc_file}")
    print(f"Output file: {output_file}")
    print(f"Hillslope: {hillslope}")
    print()

    # Generate plot
    plot_zwt_profile(nc_file, output_file, hillslope)
