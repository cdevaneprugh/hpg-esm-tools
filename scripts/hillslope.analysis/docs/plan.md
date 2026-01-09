## Files
1. combined_h0.nc
2. combined_annual_h0.nc
3. combined_h1.nc

### Still need
1. 20 year binned h0 file. - DONE
2. 20 year binned h1 file that preserves column data (bin per column). - DONE

## Plots
1. hillslope elevation and width profile. - DONE
2. column areas - DONE
    * use column weight column rather than calculating ourselves? - NO
3. pft distribution - DONE
    * make sure labels are correct - DONE
    * double check that all column types and weights are equal (besides col 16). - DONE
4. zwt hillslope profile
    * keep same basic idea
    * look at 20 year averages at 3 different points in simulation (start, middle, end)
    * Clean up shaded portions and labels

### Still need
1. Time series of full simulation
    * GPP, TOTECOSYSC, ZWT
    * At gridcell level (cant see per column difference at this time scale)
    * Binned every 20 years
2. Time series short
    * GPP, TOTECOSYSC, ZWT
    * Binned annually and every month (compare at gridcell level)
    * Last 20 years of simulation (last 20 full calendar years)
    * also look at columns using h1 file (are they so similar that plots are messy?).
NOTE: Need to bin by elevation and cardinal direction.

## Other
1. Look at carbon production at different soil depths --> compare to mean water table height.
    * We should see carbon peak at mean water table height.
    * This is lower priority.
2. Look at topographical or LIDAR data of site
    * clever ways to display it?
    * how to get an average hillslope to use for the model?
    * ideally get one hillslope that can work for each cardinal direction.
3. Sanity check that each column is what I think. - DONE
    * Maybe make a table or csv with values to compare directly.
    * Have a list or separate column of vals that are identical?
