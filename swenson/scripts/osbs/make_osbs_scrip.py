#!/usr/bin/env python3
"""
Create a single-cell SCRIP-format NetCDF for the OSBS hillslope case.

The output is the input to ESMF_Scrip2Unstruct, which produces the
ESMF unstructured mesh CTSM consumes via LND_DOMAIN_MESH /
ATM_DOMAIN_MESH / MASK_MESH. This workflow replaces the PTS_LAT /
PTS_LON shortcut and is the Phase H workaround for CTSM Issue #1432
(grc%area = spval in NUOPC single-point mode).

Default behavior reads center lat/lon and gridcell area directly
from the production hillslope NetCDF, so the common case is:

    python scripts/osbs/make_osbs_scrip.py --verbose

CLI flags override the defaults. The same script handles option (c')
from Phase H B1 — just pass --area 36.81 to target the NEON
terrestrial sampling boundary.

Corner ordering follows the CTSM-shipped reference implementation
in tools/mkmapgrids/mkscripgrid.ncl (Erik Kluzek, 2011): CCW from
SW, i.e. positions [SW, SE, NE, NW]. Area in steradians follows the
spherical-cap formula (sin(lat_N) - sin(lat_S)) * dLon_rad.

Usage:
    python scripts/osbs/make_osbs_scrip.py [--hillslope FILE]
        [--lat L] [--lon M] [--area A_KM2]
        [--output OUT] [--name NAME] [--verbose]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
SWENSON_DIR = SCRIPT_DIR.parent.parent
DEFAULT_HILLSLOPE = (
    SWENSON_DIR
    / "output"
    / "osbs"
    / "2026-05-05_production"
    / "hillslopes_osbs_production_c260505.nc"
)
DEFAULT_OUTPUT_DIR = SWENSON_DIR / "output" / "mesh"

EARTH_RADIUS_KM = 6371.0
KM_PER_DEG_LAT = math.pi * EARTH_RADIUS_KM / 180.0  # 111.195 km/°


def read_defaults_from_hillslope(hillslope_path: Path) -> tuple[float, float, float]:
    """Pull center lat/lon and gridcell area from the hillslope NetCDF."""
    with xr.open_dataset(hillslope_path) as ds:
        lat = float(ds["LATIXY"].values.flatten()[0])
        lon = float(ds["LONGXY"].values.flatten()[0])
        area_km2 = float(ds["AREA"].values.flatten()[0])
    return lat, lon, area_km2


def compute_square_cell_geometry(
    lat_c: float, lon_c: float, area_km2: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute corners and area in steradians for a square-on-sphere cell.

    Returns
    -------
    corner_lats : ndarray of shape (4,)
        CCW from SW: [SW_lat, SE_lat, NE_lat, NW_lat]
    corner_lons : ndarray of shape (4,)
        CCW from SW: [SW_lon, SE_lon, NE_lon, NW_lon]
    area_sr : float
        Cell area in steradians (= radian²).
    """
    side_km = math.sqrt(area_km2)
    cos_lat = math.cos(math.radians(lat_c))
    if cos_lat < 0.1:
        raise ValueError(
            f"Latitude {lat_c}° is too close to a pole "
            "(cos(lat) < 0.1); square-cell approximation breaks down."
        )
    d_lat = side_km / KM_PER_DEG_LAT
    d_lon = side_km / (KM_PER_DEG_LAT * cos_lat)
    half_lat = 0.5 * d_lat
    half_lon = 0.5 * d_lon

    lat_S = lat_c - half_lat
    lat_N = lat_c + half_lat
    lon_W = lon_c - half_lon
    lon_E = lon_c + half_lon

    # CCW from SW (matches mkscripgrid.ncl:135-150 convention)
    corner_lats = np.array([lat_S, lat_S, lat_N, lat_N], dtype=np.float64)
    corner_lons = np.array([lon_W, lon_E, lon_E, lon_W], dtype=np.float64)

    # Spherical-cap area (exact for a lat/lon-aligned cell)
    area_sr = (math.sin(math.radians(lat_N)) - math.sin(math.radians(lat_S))) * math.radians(d_lon)

    return corner_lats, corner_lons, area_sr


def write_scrip(
    output_path: Path,
    lat_c: float,
    lon_c: float,
    area_km2: float,
    name: str,
    imask: int = 1,
    source_hillslope: Path | None = None,
) -> None:
    """Write a single-cell SCRIP NetCDF to output_path."""
    corner_lats, corner_lons, area_sr = compute_square_cell_geometry(lat_c, lon_c, area_km2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with Dataset(output_path, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("grid_size", 1)
        nc.createDimension("grid_corners", 4)
        nc.createDimension("grid_rank", 2)

        v_dims = nc.createVariable("grid_dims", "i4", ("grid_rank",))
        v_dims[:] = np.array([1, 1], dtype=np.int32)

        v_clat = nc.createVariable("grid_center_lat", "f8", ("grid_size",))
        v_clat.units = "degrees"
        v_clat[:] = np.array([lat_c], dtype=np.float64)

        v_clon = nc.createVariable("grid_center_lon", "f8", ("grid_size",))
        v_clon.units = "degrees"
        v_clon[:] = np.array([lon_c], dtype=np.float64)

        v_klat = nc.createVariable("grid_corner_lat", "f8", ("grid_size", "grid_corners"))
        v_klat.units = "degrees"
        v_klat[:] = corner_lats.reshape(1, 4)

        v_klon = nc.createVariable("grid_corner_lon", "f8", ("grid_size", "grid_corners"))
        v_klon.units = "degrees"
        v_klon[:] = corner_lons.reshape(1, 4)

        v_imask = nc.createVariable("grid_imask", "i4", ("grid_size",))
        v_imask[:] = np.array([imask], dtype=np.int32)

        v_area = nc.createVariable("grid_area", "f8", ("grid_size",))
        v_area.units = "steradian"
        v_area[:] = np.array([area_sr], dtype=np.float64)

        nc.Conventions = "SCRIP"
        nc.title = f"SCRIP grid for {name}"
        nc.source = "Custom 1x1 SCRIP for OSBS hillslope routing (Phase H, Issue #1432 workaround)"
        nc.history = (
            f"{_dt.datetime.now().isoformat()}: created by "
            f"scripts/osbs/make_osbs_scrip.py"
        )
        if source_hillslope is not None:
            nc.derived_from = str(source_hillslope)
        nc.center_lat_deg = lat_c
        nc.center_lon_deg = lon_c
        nc.area_km2 = area_km2
        nc.area_steradian = area_sr


def verify_scrip(output_path: Path) -> None:
    """Re-open the SCRIP file and assert schema correctness."""
    with xr.open_dataset(output_path) as ds:
        assert ds.sizes == {"grid_size": 1, "grid_corners": 4, "grid_rank": 2}, (
            f"unexpected dims: {dict(ds.sizes)}"
        )
        required = [
            "grid_dims",
            "grid_center_lat",
            "grid_center_lon",
            "grid_corner_lat",
            "grid_corner_lon",
            "grid_imask",
            "grid_area",
        ]
        for var in required:
            assert var in ds.variables, f"missing required variable: {var}"
        assert ds.grid_center_lat.attrs.get("units") == "degrees"
        assert ds.grid_center_lon.attrs.get("units") == "degrees"
        assert ds.grid_corner_lat.attrs.get("units") == "degrees"
        assert ds.grid_corner_lon.attrs.get("units") == "degrees"
        assert ds.grid_area.attrs.get("units") == "steradian"
        # CCW ordering check: SW.lat = SE.lat (south) and NE.lat = NW.lat (north)
        klat = ds.grid_corner_lat.values[0]
        klon = ds.grid_corner_lon.values[0]
        assert klat[0] == klat[1] and klat[2] == klat[3], "corner lat not in CCW-from-SW order"
        assert klon[0] == klon[3] and klon[1] == klon[2], "corner lon not in CCW-from-SW order"
        assert klat[0] < klat[2], "south corner not south of north corner"
        assert klon[0] < klon[1], "west corner not west of east corner"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--hillslope",
        type=Path,
        default=DEFAULT_HILLSLOPE,
        help=f"Hillslope NetCDF to read default lat/lon/area from (default: {DEFAULT_HILLSLOPE.relative_to(SWENSON_DIR)})",
    )
    p.add_argument("--lat", type=float, default=None, help="Override center latitude (degrees north)")
    p.add_argument("--lon", type=float, default=None, help="Override center longitude (degrees east, 0-360)")
    p.add_argument("--area", type=float, default=None, help="Override gridcell area in km²")
    p.add_argument("--name", type=str, default="OSBS", help="Site name embedded in metadata (default: OSBS)")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SCRIP NetCDF path (default: output/mesh/osbs_scrip_<area>km2_c<YYMMDD>.nc)",
    )
    p.add_argument("--verbose", action="store_true", help="Print progress")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    lat_d, lon_d, area_d = read_defaults_from_hillslope(args.hillslope)
    lat_c = args.lat if args.lat is not None else lat_d
    lon_c = args.lon if args.lon is not None else lon_d
    area_km2 = args.area if args.area is not None else area_d

    if args.output is None:
        date_str = _dt.date.today().strftime("%y%m%d")
        area_str = f"{area_km2:.0f}" if area_km2 == int(area_km2) else f"{area_km2:.2f}".replace(".", "p")
        args.output = DEFAULT_OUTPUT_DIR / f"osbs_scrip_{area_str}km2_c{date_str}.nc"

    if args.verbose:
        print(f"Center: ({lat_c:.6f}°N, {lon_c:.6f}°E)")
        print(f"Area:   {area_km2:.3f} km²")
        print(f"Source: {args.hillslope}")
        print(f"Output: {args.output}")

    write_scrip(
        args.output,
        lat_c=lat_c,
        lon_c=lon_c,
        area_km2=area_km2,
        name=args.name,
        source_hillslope=args.hillslope,
    )
    verify_scrip(args.output)

    if args.verbose:
        with xr.open_dataset(args.output) as ds:
            print()
            print("=== SCRIP file summary ===")
            print(f"  grid_size:        {int(ds.sizes['grid_size'])}")
            print(f"  grid_corners:     {int(ds.sizes['grid_corners'])}")
            print(f"  grid_rank:        {int(ds.sizes['grid_rank'])}")
            print(f"  grid_center_lat:  {float(ds.grid_center_lat.values[0]):.6f}°")
            print(f"  grid_center_lon:  {float(ds.grid_center_lon.values[0]):.6f}°")
            klat = ds.grid_corner_lat.values[0]
            klon = ds.grid_corner_lon.values[0]
            print(f"  grid_corner_lat:  SW={klat[0]:.6f}  SE={klat[1]:.6f}  NE={klat[2]:.6f}  NW={klat[3]:.6f}")
            print(f"  grid_corner_lon:  SW={klon[0]:.6f}  SE={klon[1]:.6f}  NE={klon[2]:.6f}  NW={klon[3]:.6f}")
            print(f"  grid_imask:       {int(ds.grid_imask.values[0])}")
            print(f"  grid_area:        {float(ds.grid_area.values[0]):.4e} sr")
            print(f"  -> implied km²:   {float(ds.grid_area.values[0]) * 4 * math.pi * EARTH_RADIUS_KM**2 / (4 * math.pi):.3f} km²")
        print()
        print(f"Wrote {args.output}")
        print(f"Next step: ESMF_Scrip2Unstruct {args.output} <mesh_output.nc> 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
