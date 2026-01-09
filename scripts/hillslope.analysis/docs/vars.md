Carbon State Variables

Source: src/soilbiogeochem/SoilBiogeochemCarbonStateType.F90

| Variable     | Description           | Units |
|--------------|-----------------------|-------|
| SOILC_vr     | Total soil C          | gC/m³ |
| LIT_MET_C_vr | Metabolic litter C    | gC/m³ |
| LIT_CEL_C_vr | Cellulose litter C    | gC/m³ |
| LIT_LIG_C_vr | Lignin litter C       | gC/m³ |
| SOM_ACT_C_vr | Active SOM C          | gC/m³ |
| SOM_SLO_C_vr | Slow SOM C            | gC/m³ |
| SOM_PAS_C_vr | Passive SOM C         | gC/m³ |
| CWD_C_vr     | Coarse woody debris C | gC/m³ |

Note: C13/C14 isotope variants also exist (e.g., C13_SOILC_vr, C14_SOILC_vr) but are inactive by default.

---
Carbon Flux Variables

Source: src/soilbiogeochem/SoilBiogeochemCarbonFluxType.F90

| Variable      | Description                     | Units    |
|---------------|---------------------------------|----------|
| HR_vr         | Total heterotrophic respiration | gC/m³/s  |
| FPHR_vr       | Potential HR fraction (for CH4) | unitless |
| LIT_MET_HR_vr | HR from metabolic litter        | gC/m³/s  |
| LIT_CEL_HR_vr | HR from cellulose litter        | gC/m³/s  |
| LIT_LIG_HR_vr | HR from lignin litter           | gC/m³/s  |
| SOM_ACT_HR_vr | HR from active SOM              | gC/m³/s  |
| SOM_SLO_HR_vr | HR from slow SOM                | gC/m³/s  |
| SOM_PAS_HR_vr | HR from passive SOM             | gC/m³/s  |
| CWD_HR_vr     | HR from CWD                     | gC/m³/s  |

Carbon Transfer Fluxes:
| Variable                  | Description                   | Units   |
|---------------------------|-------------------------------|---------|
| LIT_MET_C_TO_SOM_ACT_C_vr | Metabolic litter --> Active SOM | gC/m³/s |
| LIT_CEL_C_TO_SOM_SLO_C_vr | Cellulose litter --> Slow SOM   | gC/m³/s |
| LIT_LIG_C_TO_SOM_SLO_C_vr | Lignin litter --> Slow SOM      | gC/m³/s |
| SOM_ACT_C_TO_SOM_SLO_C_vr | Active --> Slow SOM             | gC/m³/s |
| SOM_SLO_C_TO_SOM_PAS_C_vr | Slow --> Passive SOM            | gC/m³/s |
| CWD_C_TO_LIT_CEL_C_vr     | CWD --> Cellulose litter        | gC/m³/s |
| CWD_C_TO_LIT_LIG_C_vr     | CWD --> Lignin litter           | gC/m³/s |

---
Environmental Scalars

Source: src/soilbiogeochem/SoilBiogeochemCarbonFluxType.F90

| Variable | Description                          | Units    |
|----------|--------------------------------------|----------|
| T_SCALAR | Temperature limitation on decomp     | unitless |
| W_SCALAR | Moisture limitation on decomp        | unitless |
| O_SCALAR | Oxygen (anoxia) limitation on decomp | unitless |

