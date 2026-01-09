# CTSM Case Summary Report

This document summarizes the configuration and run status of custom and test cases.

## Custom Cases

### custom.I1850Clm50BgcCru.f09_g17.250926-101933
- **Compset**: 1850_DATM%CRUv7_CLM50%BGC_SICE_SOCN_MOSART_SGLC_SWAV_SESP
- **Resolution**: f09_g17 (0.9x1.25 atmosphere/land, gx1v7 ocean, r05 river)
- **Run Status**: SUCCESS
- **Completion Date**: 2025-09-26 11:35:21

### custom.I1850Clm60BgcCru.f09_g17.250926-101954
- **Compset**: 1850_DATM%CRUv7_CLM60%BGC_SICE_SOCN_MOSART_SGLC_SWAV_SESP
- **Resolution**: f09_g17 (0.9x1.25 atmosphere/land, gx1v7 ocean, r05 river)
- **Run Status**: SUCCESS
- **Completion Date**: 2025-09-26 12:45:56

## Test Cases

### test.I1850Clm50BgcCropCru.f09_g17.250926-100507
- **Compset**: 1850_DATM%CRUv7_CLM50%BGC-CROP_SICE_SOCN_MOSART_SGLC_SWAV_SESP
- **Resolution**: f09_g17 (0.9x1.25 atmosphere/land, gx1v7 ocean, r05 river)
- **Run Status**: SUCCESS
- **Completion Date**: 2025-09-30 10:23:48

### test.I1850Clm50Bgc.f09_g17.250926-054939
- **Compset**: 1850_DATM%GSWP3v1_CLM50%BGC_SICE_SOCN_MOSART_SGLC_SWAV_SESP
- **Resolution**: f09_g17 (0.9x1.25 atmosphere/land, gx1v7 ocean, r05 river)
- **Run Status**: SUCCESS
- **Completion Date**: 2025-09-26 19:15:08

### test.I1850Clm50SpCru.f09_g17.250926-054839
- **Compset**: 1850_DATM%CRUv7_CLM50%SP_SICE_SOCN_MOSART_SGLC_SWAV_SESP
- **Resolution**: f09_g17 (0.9x1.25 atmosphere/land, gx1v7 ocean, r05 river)
- **Run Status**: SUCCESS
- **Completion Date**: 2025-09-26 06:45:30

### test.I1850Clm60BgcCropCru.f09_g17.250930-113918
- **Compset**: 1850_DATM%CRUv7_CLM60%BGC-CROP_SICE_SOCN_MOSART_SGLC_SWAV_SESP
- **Resolution**: f09_g17 (0.9x1.25 atmosphere/land, gx1v7 ocean, r05 river)
- **Run Status**: SUCCESS
- **Completion Date**: 2025-09-30 12:41:49

### test.I1850Clm60Bgc.f09_g17.250926-054949
- **Compset**: 1850_DATM%GSWP3v1_CLM60%BGC_SICE_SOCN_MOSART_SGLC_SWAV_SESP
- **Resolution**: f09_g17 (0.9x1.25 atmosphere/land, gx1v7 ocean, r05 river)
- **Run Status**: SUCCESS
- **Completion Date**: 2025-09-26 20:12:33

### test.I1850Clm60SpCru.f09_g17.250926-054854
- **Compset**: 1850_DATM%CRUv7_CLM60%SP_SICE_SOCN_MOSART_SGLC_SWAV_SESP
- **Resolution**: f09_g17 (0.9x1.25 atmosphere/land, gx1v7 ocean, r05 river)
- **Run Status**: SUCCESS
- **Completion Date**: 2025-09-26 06:57:12

## Summary

- **Total Cases**: 8 (2 custom, 6 test)
- **Successful Runs**: 8/8 (100%)
- **Common Resolution**: All cases use f09_g17 grid
- **Model Versions Tested**: CLM50 and CLM60
- **Configuration Types**: BGC, BGC-CROP, SP (Satellite Phenology)
- **Forcing Data**: CRUv7 and GSWP3v1
