#!/usr/bin/env python3
"""
Plot PFT (Plant Functional Type) distribution for hillslope columns
"""

import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Parameter file: /blue/gerber/earth_models/inputdata/lnd/clm2/paramdata/ctsm60_params.c250311.nc
PFT_NAMES = {
    0: 'Not vegetated',
    1: 'Needleleaf evergreen temperate tree',
    2: 'Needleleaf evergreen boreal tree',
    3: 'Needleleaf deciduous boreal tree',
    4: 'Broadleaf evergreen tropical tree',
    5: 'Broadleaf evergreen temperate tree',
    6: 'Broadleaf deciduous tropical tree',
    7: 'Broadleaf deciduous temperate tree',
    8: 'Broadleaf deciduous boreal tree',
    9: 'Broadleaf evergreen shrub',
    10: 'Broadleaf deciduous temperate shrub',
    11: 'Broadleaf deciduous boreal shrub',
    12: 'C3 arctic grass',
    13: 'C3 non-arctic grass',
    14: 'C4 grass',
    15: 'C3 crop',
    16: 'C3 irrigated',
    17: 'Temperate corn',
    18: 'Irrigated temperate corn',
    19: 'Spring wheat',
    20: 'Irrigated spring wheat',
    21: 'Winter wheat',
    22: 'Irrigated winter wheat',
    23: 'Temperate soybean',
    24: 'Irrigated temperate soybean',
    25: 'Barley',
    26: 'Irrigated barley',
    27: 'Winter barley',
    28: 'Irrigated winter barley',
    29: 'Rye',
    30: 'Irrigated rye',
    31: 'Winter rye',
    32: 'Irrigated winter rye',
    33: 'Cassava',
    34: 'Irrigated cassava',
    35: 'Citrus',
    36: 'Irrigated citrus',
    37: 'Cocoa',
    38: 'Irrigated cocoa',
    39: 'Coffee',
    40: 'Irrigated coffee',
    41: 'Cotton',
    42: 'Irrigated cotton',
    43: 'Date palm',
    44: 'Irrigated date palm',
    45: 'Fodder grass',
    46: 'Irrigated fodder grass',
    47: 'Grapes',
    48: 'Irrigated grapes',
    49: 'Groundnuts',
    50: 'Irrigated groundnuts',
    51: 'Millet',
    52: 'Irrigated millet',
    53: 'Oil palm',
    54: 'Irrigated oil palm',
    55: 'Potatoes',
    56: 'Irrigated potatoes',
    57: 'Pulses',
    58: 'Irrigated pulses',
    59: 'Rapeseed',
    60: 'Irrigated rapeseed',
    61: 'Rice',
    62: 'Irrigated rice',
    63: 'Sorghum',
    64: 'Irrigated sorghum',
    65: 'Sugar beet',
    66: 'Irrigated sugar beet',
    67: 'Sugarcane',
    68: 'Irrigated sugarcane',
    69: 'Sunflower',
    70: 'Irrigated sunflower',
    71: 'Miscanthus',
    72: 'Irrigated miscanthus',
    73: 'Switchgrass',
    74: 'Irrigated switchgrass',
    75: 'Tropical corn',
    76: 'Irrigated tropical corn',
    77: 'Tropical soybean',
    78: 'Irrigated tropical soybean',
}

def plot_pft_distribution(nc_file, output_file):
    """
    Plot PFT distribution for a single hillslope column
    
    Parameters:
    -----------
    nc_file : str
        Path to h1 NetCDF file
    output_file : str
        Output PNG filename
    """
    
    # Open NetCDF file
    ds = xr.open_dataset(nc_file)
    
    # Get PFT data
    pft_column = ds['pfts1d_ci'].values
    pft_types = ds['pfts1d_itype_veg'].values
    pft_weights = ds['pfts1d_wtcol'].values
    
    # All hillslope columns have same PFT distribution, so just use first
    col_mask = pft_column == 1
    col_types = pft_types[col_mask]
    col_weights = pft_weights[col_mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create pie chart
    #colors = plt.cm.Set3(np.arange(len(col_types)))
    wedges, texts, autotexts = ax.pie(col_weights,
                                        labels=[PFT_NAMES.get(t, f'PFT {t}') for t in col_types],
    #                                    colors=colors,
                                        autopct='%1.2f%%',
                                        startangle=90)

    # Improve text visibility
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)
        autotext.set_weight('bold')

    for text in texts:
        text.set_fontsize(10)

    ax.set_title('OSBS Hillslope PFT Distribution',
                 fontsize=14, fontweight='bold')
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    ds.close()

if __name__ == '__main__':
    # Default parameters
    nc_file = './data/combined_h1.nc'
    output_file = './plots/pft_distribution.png'
    
    # Check command line arguments
    if len(sys.argv) >= 2:
        nc_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    
    print(f"Input file: {nc_file}")
    print(f"Output file: {output_file}")
    
    # Generate plot
    plot_pft_distribution(nc_file, output_file)
