#!/bin/bash

module purge
#module load subversion/1.14.2
module load perl/5.24.1   
module load cmake/3.26.4 
module load python/3.12
module load gcc/14.2.0    
module load openmpi/5.0.7
module load netcdf-c/4.9.3  
module load netcdf-f/4.6.2
module load hdf5/1.14.6
module load lapack/3.11.0
module load esmf/8.8.1
module save ctsm-modules
