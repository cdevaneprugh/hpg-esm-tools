#!/bin/bash

cmake \
  -DNetCDF_C_PATH=/apps/gcc/14.2.0/openmpi/5.0.7/netcdf-c/4.9.3 \
  -DNetCDF_Fortran_PATH=/apps/gcc/14.2.0/openmpi/5.0.7/netcdf-f/4.6.2 \
  -DWITH_PNETCDF=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX=`pwd`/bld \
  -DCMAKE_INSTALL_RPATH=`pwd`/src/gptl \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \

make

make install
