# Swenson Representative Hillslope Implementation

Implementation of Swenson & Lawrence (2025) representative hillslope methodology for OSBS.

## Background

@../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md

## Key Resources

| Resource | Location |
|----------|----------|
| Progress tracking | `progress-tracking.md` (this directory) |
| Paper summary | `../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md` |
| Swenson's codebase | `/blue/gerber/cdevaneprugh/Representative_Hillslopes/` |
| Our pysheds fork | `$BLUE/pysheds_fork` |
| pysheds documentation | https://mattbartos.com/pysheds/ |
| Processing scripts | `scripts/` (this directory) |

## Directory Structure

```
swenson/
├── CLAUDE.md              # This file - context loader
├── progress-tracking.md   # Progress tracking and reference docs
└── scripts/               # Processing scripts
```

## pysheds Setup

Our pysheds fork: `$BLUE/pysheds_fork` (env var: `$PYSHEDS_FORK`)

**Branches:**
- `master` - synced with upstream
- `uf-development` - our development branch

**To use:** Run `pysheds-env` before running scripts (adds to PYTHONPATH).

```bash
pysheds-env
python -c "from pysheds.sgrid import sGrid; print('OK')"
```

**Remotes:**
- `origin` - `git@github.com:cdevaneprugh/pysheds.git`
- `upstream` - `https://github.com/mdbartos/pysheds.git`
