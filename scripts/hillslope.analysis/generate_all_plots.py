#!/usr/bin/env python3
"""
Generate all hillslope analysis plots
"""

from plot_timeseries_full import plot_timeseries_full
from plot_timeseries_last20 import plot_timeseries_last20
from plot_zwt_hillslope_profile import plot_zwt_profile
from plot_elevation_width_overlay import plot_overlay_profiles
from plot_col_areas import plot_column_areas
from plot_pft_distribution import plot_pft_distribution

# Data files
h0_20yr = 'data/combined_h0_20yr.nc'
h1_1yr = 'data/combined_h1_1yr.nc'
h1_20yr = 'data/combined_h1_20yr.nc'
h1 = 'data/combined_h1.nc'

# Variables for timeseries plots
variables = ['GPP', 'TOTECOSYSC', 'ZWT']

# Full simulation timeseries (20-year bins)
for var in variables:
    plot_timeseries_full(h0_20yr, f'plots/{var}_full.png', var)

# Last 20 years timeseries (annual bins)
for var in variables:
    plot_timeseries_last20(h1_1yr, f'plots/{var}_last20.png', var)

# Water table profiles
plot_zwt_profile(h1_20yr, f'plots/ZWT_hillslope_profile.png', 'North')

# Other plots
plot_overlay_profiles(h1, 'plots/elevation_width_overlay.png')
plot_column_areas(h1, 'plots/column_areas.png')
plot_pft_distribution(h1, 'plots/pft_distribution.png')
