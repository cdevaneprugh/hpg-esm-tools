# Script & Documentation Review - 2025-01-10

Comprehensive review of all scripts and documentation in hpg-esm-tools.

---

## Executive Summary

| Directory | Quality | Status | Action |
|-----------|---------|--------|--------|
| scripts/case.analyzer/ | EXCELLENT | Production | Keep as-is |
| scripts/hillslope.analysis/ | GOOD | Production | Minor refactoring |
| scripts/inputdata.downloading/ | EXCELLENT | Production | Keep as-is |
| scripts/porting/ | POOR-FAIR | Dormant | Consolidate to docs |
| scripts/deploy.custom.files/ | EXCELLENT | Legacy | Keep (transition planned) |
| docs/ | EXCELLENT | Active | Keep all |

---

## 1. scripts/case.analyzer/ - KEEP AS-IS

**Score: 8.6/10 - Production Ready**

Fast hybrid toolkit for analyzing CTSM case outputs. Combines Bash orchestration, NCO NetCDF manipulation, and Python plotting.

### Scripts
| Script | Lines | Purpose | Quality |
|--------|-------|---------|---------|
| analyze_case | 143 | Main orchestrator | EXCELLENT |
| generate_case_summary.sh | 148 | XML extraction + file catalog | VERY GOOD |
| concat_hist_stream | 99 | NetCDF concatenation | EXCELLENT |
| bin_temporal | 138 | Annual averaging | VERY GOOD |
| plot_variables.py | 66 | Plotting engine | EXCELLENT |
| default.conf | 48 | Configuration | EXCELLENT |

### Strengths
- Defensive programming (`set -euo pipefail`)
- Modular design (each script standalone)
- No hardcoded paths (queries CIME XML)
- Excellent documentation (README, PROJECT_PLAN)
- Smart features (incomplete file detection, memory-safe large file handling)

### Recommendations
- **Keep as-is** - No changes needed
- Optional: Add year detection validation to bin_temporal
- Optional: Add missing variable handling to plot_variables.py

---

## 2. scripts/hillslope.analysis/ - MINOR REFACTORING

**Score: 7/10 - Good with improvement opportunities**

Specialized scripts for CTSM hillslope hydrology analysis with column-level (h1 stream) data.

### Scripts
| Script | Purpose | Quality | Action |
|--------|---------|---------|--------|
| bin_1yr.sh | Annual binning | GOOD | Keep |
| bin_20yr.sh | N-year binning | GOOD | Consolidate with bin_1yr.sh |
| plot_timeseries_full.py | Long-term gridcell averages | GOOD | Keep |
| plot_timeseries_last20.py | Recent period breakdown | EXCELLENT | Keep |
| plot_zwt_hillslope_profile.py | Water table profiles | VERY GOOD | Implement 3-period feature |
| plot_elevation_width_overlay.py | Hillslope geometry | GOOD | Enable color scheme |
| plot_col_areas.py | Column area distribution | GOOD | Standardize styling |
| plot_pft_distribution.py | PFT composition | GOOD | Keep |
| plot_vr_profile.py | Vertical soil profiles | EXCELLENT | Parameterize hillslope |
| get_gridcell.py | Grid index calculator | FUNCTIONAL | Convert to utility or remove |
| generate_all_plots.py | Batch plotting | GOOD | Add config support |

### Key Issues
1. **Code duplication**: bin_1yr.sh and bin_20yr.sh are 95% identical
2. **Commented-out styling**: Multiple scripts have disabled color specifications
3. **Hardcoded parameters**: Some scripts fixed to specific hillslopes/periods

### Recommended Refactoring
1. Consolidate bin_1yr.sh + bin_20yr.sh into single `bin_temporal.sh --years=N`
2. Enable color schemes in overlay and area plots
3. Parameterize plot_vr_profile.py for any hillslope
4. Implement 3-period ZWT profile (documented but not implemented)

---

## 3. scripts/inputdata.downloading/ - KEEP AS-IS

**Score: 9/10 - Production Ready**

Comprehensive toolkit for downloading CTSM input data from NCAR SVN servers.

### Scripts
| Script | Lines | Purpose | Quality |
|--------|-------|---------|---------|
| download_files.sh | 196 | Main download orchestrator | EXCELLENT |
| get_file_info.sh | 108 | SVN metadata scanner | GOOD |
| parse_svn_xml.py | 154 | SVN XML parser | EXCELLENT |
| ncar_speed_test.sh | 83 | Network testing | GOOD |

### Strengths
- Resumable downloads with 3 retries
- Smart caching (checks local before downloading)
- Zero-size file detection
- Comprehensive logging
- SLURM recommendations for large downloads

### Evidence of Use
- 4,109 files indexed in logs (Sept 26, 2025)
- Active GSWP3 atmospheric forcing data downloads

### Recommendations
- **Keep as-is** - Production ready
- Minor: Add timestamps to generated files
- Minor: Make test URL configurable in speed test

---

## 4. scripts/porting/ - CONSOLIDATE TO DOCS

**Score: 4/10 - Minimal utility**

Minimal collection of environment setup scripts. Mostly obsolete given conda environment.

### Scripts
| Script | Lines | Purpose | Quality | Action |
|--------|-------|---------|---------|--------|
| module_env_setup.sh | 15 | Module loading | FAIR | Keep as reference |
| get_upstream_changes.sh | 3 | Git changelog | POOR | REMOVE |
| build_shared_pio.sh | 14 | PIO build | FAIR | ARCHIVE |

### Issues
- No error handling or validation
- Incomplete implementations
- Superseded by conda environment (`ctsm`)

### Recommendations
1. **Remove** get_upstream_changes.sh (too minimal, document as git recipe)
2. **Archive** build_shared_pio.sh (pre-built PIO exists)
3. **Keep** module_env_setup.sh as reference with validation notes
4. Document module requirements in CLAUDE.md instead

---

## 5. scripts/deploy.custom.files/ - KEEP (TRANSITION PLANNED)

**Score: 9/10 - Excellent but being replaced**

Sophisticated deployment system for propagating custom CTSM modifications.

### Files
| File | Purpose | Quality |
|------|---------|---------|
| deploy.sh | Main deployment orchestrator (367 lines) | EXCELLENT |
| config_machines.xml | HiPerGator machine config | EXCELLENT |
| config_batch.xml | SLURM batch directives | EXCELLENT |
| Various SourceMods | Custom CTSM modifications | CURRENT |

### Strengths
- Production-grade error handling
- Smart mode (only changed files) vs force mode
- Git-style diff logging
- Comprehensive help text

### Status
Per CLAUDE.md: "Being replaced by proper CTSM fork strategy"

### Recommendations
- **Keep for now** - Still functional and valuable
- Create migration guide when fork strategy is ready
- Archive configs as reference for new workflow

---

## 6. docs/ - KEEP ALL

**Score: 9/10 - Excellent documentation**

High-quality technical documentation with strong current relevance.

### Documents
| Document | Lines | Purpose | Quality | Status |
|----------|-------|---------|---------|--------|
| CTSM_Deterministic_Testing_Analysis.md | 404 | Hash comparison study | EXCELLENT | Current |
| CTSM_CPRNC_Deterministic_Analysis.md | 464 | CPRNC validation | EXCELLENT | Current |
| NCO_TEMPORAL_ANALYSIS.md | 542 | NCO workflow guide | VERY GOOD | Actionable |
| SPILLHEIGHT_IMPLEMENTATION.md | 100+ | Hillslope feature docs | GOOD | Active |
| TEST_Cases.md | 65 | Case summary | FAIR | Reference |
| clm.hist.names/ | - | Field reference | GOOD | Reference |

### Recommendations
- **Keep all** - Excellent documentation
- Consider implementing bin_temporal.sh as suggested in NCO_TEMPORAL_ANALYSIS.md
- Expand TEST_Cases.md with more detail if time permits

---

## Action Items Summary

### Immediate (Quick Wins)
- [ ] Remove scripts/porting/get_upstream_changes.sh
- [ ] Enable color schemes in hillslope plotting scripts
- [ ] Fix docstring mismatch in plot_zwt_hillslope_profile.py

### Short-term (Refactoring)
- [ ] Consolidate bin_1yr.sh + bin_20yr.sh into parameterized script
- [ ] Parameterize plot_vr_profile.py for any hillslope
- [ ] Add config file support to generate_all_plots.py
- [ ] Archive build_shared_pio.sh with documentation

### Medium-term (Enhancements)
- [ ] Implement 3-period ZWT profile functionality
- [ ] Create shared utility module for common matplotlib patterns
- [ ] Document module requirements in CLAUDE.md (replace porting/ scripts)

### Not Recommended
- Converting scripts to Claude agents (current CLI interface is optimal)
- Major refactoring of case.analyzer (already production-ready)
- Removing deploy.custom.files (still actively used)

---

## Dependencies

### External Tools Required
- **NCO**: ncdump, ncra, ncrcat (conda: nco>=5.1.0)
- **Git**: changelog, version control
- **wget**: file downloads
- **CIME**: xmlquery (in case directory)

### Python Libraries
- xarray, matplotlib, numpy (conda: ctsm environment)
- netcdf4 (conda: ctsm environment)

---

*Review completed: 2025-01-10*
*Next review: After refactoring items completed*
