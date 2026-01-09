#--- COMPILER QUIRKS ---------------------------------------------------
# These fortran flags are needed for gcc >= 10. Should be handled in gnu.cmake file. If flag is removed in future, uncomment here.
#string(APPEND FFLAGS " -fallow-argument-mismatch -fallow-invalid-boz")

# In netcdf-c 4.9.3, _FillValue was changed to NC_FillValue to follow best practices.
# This line ensures legacy support for code that uses netcdf-c and the original convention. netcdf-f not affected.
string(APPEND CPPDEFS " -DNETCDF_ENABLE_LEGACY_MACROS")


#--- LINK LIBRARIES ---------------------------------------------------
# Execute nf-config and store output as a variable.
# Outputs netcdf-c and f paths/libraries that need to be linked to build model.
execute_process(COMMAND ${NETCDF_FORTRAN_PATH}/bin/nf-config --flibs OUTPUT_VARIABLE SHELL_CMD_OUTPUT_BUILD_INTERNAL_IGNORE0 OUTPUT_STRIP_TRAILING_WHITESPACE)


# link netcdf and lapack libraries
string(APPEND SLIBS " ${SHELL_CMD_OUTPUT_BUILD_INTERNAL_IGNORE0}")
string(APPEND SLIBS " -L$(LAPACK_LIBDIR) -llapack -lblas")


#--- MISC -------------------------------------------------------------
# Tells parallelio to optimize build for lustre file system
set(PIO_FILESYSTEM_HINTS "lustre")
