"""
DEM resolution validation against a high-resolution DSM reference.

Compares DSM, DTM and CHM products gridded at several resolutions against the
native-resolution DSM, aggregated onto each target grid with a matched
statistic.

Expected directory layout (Perma-X style)
-----------------------------------------
Gridded products, one folder per resolution, flat per product:

    <BASE>/
        Perma-X_2023_1m/   DSM/ DTM/ CHM/ reports/
        Perma-X_2023_2m/   ...
        Perma-X_2023_5m/   ...
    filenames: WC_<Site>_<YYYYMMDD>_<gsd>_<flight>_<PRODUCT>.tif

Native-resolution reference, one folder per scene:

    <REF_ROOT>/
        WC_<Site>_<YYYYMMDD>_<gsd>_<flight>/
            WC_..._DSM.tif          <- merged mosaic, used as the reference
            WC_..._Ortho.tif, ..._Hillshade.tif, *.vrt, *.png
            DSM/ Ortho/             <- per-tile products, ignored
            PointClouds/            <- .las / .copc.laz, ignored here
            processing_info/

The gsd token ("15cm") is the source flight GSD and stays constant across the
resolution roots, so scene keys match directly between them and the reference
folder names. Only .tif is read; .png quicklooks, .vrt, tiled subfolders and
temp/ directories are ignored.

Design notes
------------
* The product raster at each resolution defines the target grid, so alignment
  is exact by construction and no product is ever resampled.
* The reference is aggregated (not interpolated) onto that grid using the same
  statistic that built the coarse DSM. A mismatched statistic produces a bias
  that scales with cell size and masquerades as a resolution effect.
* The vertical offset is estimated ONCE per scene at the finest resolution and
  applied unchanged to all resolutions. Estimating it per resolution would
  absorb exactly the systematic bias that coarsening introduces.
* CHM is computed two ways:
    - "fixed"   : DTM held at the finest resolution. Isolates the DSM effect.
                  NOTE: algebraically identical to the DSM error, since the DTM
                  cancels on both sides. Consistency check, not a result.
    - "matched" : DTM at the same resolution as the DSM. The realistic combined
                  effect, and the only CHM variant carrying new information.
* AGGREGATION_FLOOR rows give the resolvability limit: pure aggregation
  arithmetic with no sensor difference. Errors below it are not measurable
  with this reference.

Results are written per scene, so an interrupted run resumes where it stopped.

Assumes north-up, axis-aligned, projected rasters in metric units.
Requires: numpy, pandas, rasterio
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE = Path("/isipd/projects/p_planetdw/data/lidar/05_results")

# Resolution (m) -> folder name under BASE.
RES_DIRS = {
    1.0: "Perma-X_2023_1m",
    2.0: "Perma-X_2023_2m",
    5.0: "Perma-X_2023_5m",
}
RESOLUTIONS = [1.0, 2.0, 5.0]
REF_RES = 1.0  # finest target grid: defines the fixed DTM and the offset

# Reference: root containing one folder per scene, each holding the merged
# native-resolution DSM as <scene>/<scene>_DSM.tif. Tiled DSM/ and Ortho/
# subfolders, .vrt mosaics and PointClouds/ are ignored.
# Note: this sits on a different share from BASE (projects-noreplica, not projects).
REF_ROOT = Path("/isipd/projects-noreplica/p_macsprocessing/PermaX_MACS/PermaX_2023/data_products")
REF_DSM_SUFFIX = "_DSM.tif"

# Product subfolder names inside each resolution root.
DSM_SUBDIR = "DSM"
DTM_SUBDIR = "DTM"

# Restrict the run. None = all scenes. Otherwise a regex matched against the
# scene key, e.g. r"TukDriftwood|YukonCoast".
SCENE_REGEX = None

# Statistic used when the coarse DSM was built from the point/dense cloud.
# Must match, or the comparison measures the statistic, not the resolution.
#   "max"  - highest return per cell
#   "p95"  - 95th percentile; robust to outliers, recommended as primary
#   "mean" - arithmetic mean
AGG_STAT_PRIMARY = "p95"
AGG_STAT_SENSITIVITY = None   # "max" to re-run; p95 vs max differed by <0.03 m, so off by default

MIN_COVER_FRAC = 0.5   # min fraction of a target cell covered by valid reference pixels

# Stratification. Slope separates terrain effects; vegetation height separates
# the canopy effect, which the difference maps show is the dominant driver:
# error is confined to shrub networks and grows with cell size because a
# max-type lidar aggregate keeps climbing with more samples per cell while the
# SfM canopy envelope does not. Without this split the two are confounded.
SLOPE_BINS = [0.0, 2.0, 5.0, 10.0, 20.0, 90.0]
VEG_BINS = [-1.0, 0.5, 2.0, 5.0, 50.0]
VEG_LABELS = ["bare <0.5 m", "low shrub 0.5-2 m", "tall shrub 2-5 m", "trees >5 m"]

# Erode the comparison mask by this many target cells. Corridor and ribbon
# footprints (Tsiigehtchic02, PeelSlumpsHighway) have most of their pixels
# within a few cells of a scalloped edge, where the two products disagree for
# geometric reasons. 0 disables.
ERODE_EDGE_CELLS = 3

# Scenes excluded outright. These show straight-edged blocks, stripes or comb
# artifacts in the difference maps - mosaic seams or flight-line disagreement,
# not terrain. No statistic fixes them; they only contaminate the baseline.
EXCLUDE_SCENES = [
    "WC_DempsterHW01_20230709_15cm_01",     # rectangular blocks, -20 m patch
    "WC_PeelSlumpsHighway_20230715_15cm_01",  # comb artifact, ME +13 m at 5 m
]

OFFSET_MAX_SLOPE_DEG = 2.0
OFFSET_MAX_CHM_M = 0.25    # near-bare ground only, where DSM ~ DTM
OFFSET_MODE = "per_scene"  # "per_scene" (recommended) | "per_resolution" | "none"

# Blunder handling. Undeclared sentinels (-9999, -32767, 3.4e38) and SfM
# failures over water/low-texture were inflating RMSE to 180 m and LE95 to
# 600 m while NMAD stayed sane - the signature of outliers, not resolution.
# A physical elevation gate plus a fixed error threshold is more defensible
# than a percentile trim, which always discards a fixed fraction whether or
# not anything is wrong. Both are counted and reported per row.
ELEV_VALID_RANGE = (-100.0, 2000.0)  # metres; None to disable
MAX_ABS_ERROR_M = 50.0               # |error| above this is a blunder; None to keep
ERROR_CLIP_PCT = None                # percentile trim; superseded by the above

# Scenes whose product and reference footprints barely overlap give metrics
# from a small, possibly unrepresentative corner. Applied when summarising and
# plotting, not when processing, so nothing is silently lost from the CSV.
MIN_VALID_FRAC = 0.1

# Per-scene spatial difference maps (DSM - reference) at every resolution.
# The single most useful diagnostic: tells you *where* the disagreement is.
WRITE_DIFF_MAPS = True
DIFF_MAP_MAX_PX = 1200      # decimation for display only
DIFF_MAP_DPI = 130
DIFF_MAP_CLIP_M = 2.0       # fixed symmetric colour limit; None = 98th pct

# Decision threshold: the vertical error you are willing to accept. Drawn on
# the decision figure so the choice of grid is read off directly.
TOLERANCE_NMAD_M = 0.25

# Tolerance when checking that a resolution root really has the cell size its
# folder name claims. Loose enough for grids written at e.g. 0.9999999 m.
CELL_SIZE_TOL = 1e-3

OUT_DIR = BASE / "_resolution_validation"
MAKE_PLOTS = True           # write figures to OUT_DIR/plots
PLOT_ONLY = False           # True: skip processing, replot from the CSV
WRITE_DIFF_RASTERS = False  # 140 scenes x 3 res x 4 products is a lot of GeoTIFFs
INVENTORY_ONLY = False      # True: write the scene inventory and stop
RESUME = True               # skip scenes that already have a per-scene CSV
MAX_WORKERS = 32             # >1 runs scenes in parallel processes

# Aggregation runs in bands of target rows so that a full-flight 0.15 m mosaic
# never has to be held in memory. Lower this if RAM is tight, raise it for speed.
BAND_TARGET_ROWS = 16

SCENE_PATTERN = re.compile(
    r"^(?P<region>[A-Za-z]{2,3})_(?P<site>.+?)_(?P<date>\d{6}[\dXx]{2})"
    r"_(?P<gsd>\d+cm)_(?P<flight>[0-9A-Za-z]+)$"
)

# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------


def scene_key(path: Path, product: str) -> str:
    """Strip the trailing _<PRODUCT> from the stem to get the scene key."""
    stem = path.stem
    suffix = "_" + product
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def list_products(root: Path, subdir: str, product: str) -> dict[str, Path]:
    """Map scene key -> GeoTIFF path for one product folder (non-recursive)."""
    d = root / subdir
    if not d.is_dir():
        return {}
    out = {}
    for p in d.iterdir():
        if p.is_dir() or p.suffix.lower() not in (".tif", ".tiff"):
            continue
        out[scene_key(p, product)] = p
    return out


def list_reference(root: Path) -> dict[str, Path]:
    """
    Map scene key -> merged reference DSM for the per-scene folder layout:
    <root>/<scene>/<scene>_DSM.tif. Everything else in the scene folder
    (tiled DSM/, Ortho/, PointClouds/, .vrt, .png) is ignored.
    """
    out = {}
    if not root.is_dir():
        return out
    for d in root.iterdir():
        if not d.is_dir():
            continue
        p = d / f"{d.name}{REF_DSM_SUFFIX}"
        if p.is_file():
            out[d.name] = p
    return out


def check_roots() -> list[str]:
    """Report each configured directory and how many GeoTIFFs it holds."""
    lines = []
    d = REF_ROOT
    n = len(list_reference(d)) if d.is_dir() else 0
    lines.append(f"  REF_ROOT        {d}\n"
                 f"                  exists={d.is_dir()}  scenes_with_DSM={n}")
    for r in RESOLUTIONS:
        root = BASE / RES_DIRS[r]
        for label, sub in (("DSM", DSM_SUBDIR), ("DTM", DTM_SUBDIR)):
            p = root / sub
            n = (len([x for x in p.iterdir()
                      if x.suffix.lower() in (".tif", ".tiff")]) if p.is_dir() else 0)
            lines.append(f"  {r:g} m {label:3s}        {p}\n"
                         f"                  exists={p.is_dir()}  tif={n}")
    return lines


def availability(inv: pd.DataFrame) -> list[str]:
    """How many scenes have each product, so a systematic gap is obvious."""
    cols = [c for c in inv.columns
            if c == "has_ref" or c.startswith(("dsm_", "dtm_"))]
    n = len(inv)
    return [f"  {c:10s} {int(inv[c].sum()):4d} / {n}" for c in cols]


def explain_scene(key: str) -> list[str]:
    """Expected path for every input of one scene, with an exists flag."""
    lines = [f"expected paths for {key}:"]
    p = REF_ROOT / key / f"{key}{REF_DSM_SUFFIX}"
    lines.append(f"  [{'OK ' if p.is_file() else 'MISS'}] ref  {p}")
    for r in RESOLUTIONS:
        for label, sub in (("DSM", DSM_SUBDIR), ("DTM", DTM_SUBDIR)):
            q = BASE / RES_DIRS[r] / sub / f"{key}_{label}.tif"
            lines.append(f"  [{'OK ' if q.is_file() else 'MISS'}] {r:g}m {label}  {q}")
    return lines


def discover_scenes() -> tuple[dict, pd.DataFrame]:
    """Build the scene -> paths mapping plus a tabular inventory."""
    ref = list_reference(REF_ROOT)
    per_res = {
        r: {
            "DSM": list_products(BASE / RES_DIRS[r], DSM_SUBDIR, "DSM"),
            "DTM": list_products(BASE / RES_DIRS[r], DTM_SUBDIR, "DTM"),
        }
        for r in RESOLUTIONS
    }

    keys = set(ref)
    for r in RESOLUTIONS:
        keys |= set(per_res[r]["DSM"]) | set(per_res[r]["DTM"])

    scenes, rows = {}, []
    for k in sorted(keys):
        if SCENE_REGEX and not re.search(SCENE_REGEX, k):
            continue
        if k in EXCLUDE_SCENES:
            continue
        m = SCENE_PATTERN.match(k)
        meta = m.groupdict() if m else {"region": None, "site": k, "date": None,
                                        "gsd": None, "flight": None}
        entry = {"key": k, "meta": meta, "ref": ref.get(k), "res": {}}
        row = {"scene": k, **meta, "has_ref": ref.get(k) is not None}
        complete = ref.get(k) is not None
        for r in RESOLUTIONS:
            dsm, dtm = per_res[r]["DSM"].get(k), per_res[r]["DTM"].get(k)
            entry["res"][r] = {"DSM": dsm, "DTM": dtm}
            row[f"dsm_{r:g}m"] = dsm is not None
            row[f"dtm_{r:g}m"] = dtm is not None
            complete &= dsm is not None and dtm is not None
        row["complete"] = complete
        rows.append(row)
        if complete:
            scenes[k] = entry

    if not rows:
        msg = ["No scenes found. Configured paths:", *check_roots()]
        if SCENE_REGEX:
            msg.append(f"  SCENE_REGEX is set to {SCENE_REGEX!r} and may exclude "
                       f"everything; set it to None to disable filtering.")
        msg.append("Check BASE, REF_ROOT, RES_DIRS and the product subfolder names.")
        raise SystemExit("\n".join(msg))

    inv = pd.DataFrame(rows).sort_values("scene").reset_index(drop=True)
    return scenes, inv


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------


@dataclass
class Grid:
    transform: rasterio.Affine
    shape: tuple
    crs: object

    @property
    def res(self) -> float:
        return abs(self.transform.a)

    @property
    def n_cells(self) -> int:
        return self.shape[0] * self.shape[1]


def same_crs(a, b) -> bool:
    """
    Equivalence rather than object identity: the same projection defined via
    EPSG in one file and WKT in another compares unequal as objects.
    """
    if a is None or b is None:
        return a is b
    if a == b:
        return True
    try:
        ea, eb = a.to_epsg(), b.to_epsg()
        if ea is not None and eb is not None:
            return ea == eb
    except Exception:
        pass
    return False


def _assert_axis_aligned(t: rasterio.Affine, path) -> None:
    if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
        raise ValueError(f"{path}: rotated transform not supported")


def read_grid(path) -> Grid:
    with rasterio.open(path) as src:
        _assert_axis_aligned(src.transform, path)
        return Grid(src.transform, (src.height, src.width), src.crs)


def _gate_elevation(arr: np.ndarray) -> np.ndarray:
    """Mask physically impossible elevations, i.e. undeclared nodata sentinels."""
    if ELEV_VALID_RANGE is not None:
        lo, hi = ELEV_VALID_RANGE
        arr[(arr < lo) | (arr > hi)] = np.nan
    return arr


def read_array(path, band: int = 1) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float32)
        if src.nodata is not None:
            arr[arr == np.float32(src.nodata)] = np.nan
    return _gate_elevation(arr)


def resample_array_to_grid(arr, src_grid: Grid, dst_grid: Grid,
                           resampling: Resampling) -> np.ndarray:
    dst = np.full(dst_grid.shape, np.nan, dtype=np.float32)
    reproject(
        source=arr.astype(np.float32), destination=dst,
        src_transform=src_grid.transform, src_crs=src_grid.crs, src_nodata=np.nan,
        dst_transform=dst_grid.transform, dst_crs=dst_grid.crs, dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


def align_raster_to_grid(path, grid: Grid, resampling: Resampling) -> np.ndarray:
    """
    Read a raster and place it on the target grid.

    Products at the same nominal resolution do not necessarily share an extent:
    a DTM may be padded to a square while the DSM is clipped to the data
    footprint. Where the two grids are already aligned (same cell size, origins
    an integer number of cells apart) this is an exact copy-and-pad; where they
    are not, it interpolates.
    """
    with rasterio.open(path) as src:
        _assert_axis_aligned(src.transform, path)
        src_grid = Grid(src.transform, (src.height, src.width), src.crs)
        arr = _gate_elevation(src.read(1).astype(np.float32))
        if src.nodata is not None:
            arr[arr == np.float32(src.nodata)] = np.nan
    if (src_grid.shape == grid.shape
            and np.allclose(tuple(src_grid.transform), tuple(grid.transform))):
        return arr
    return resample_array_to_grid(arr, src_grid, grid, resampling)


def grids_match(path, grid: Grid) -> bool:
    """True when a raster already sits exactly on the target grid."""
    with rasterio.open(path) as src:
        return (src.height, src.width) == grid.shape and np.allclose(
            tuple(src.transform), tuple(grid.transform))


def write_raster(path, arr: np.ndarray, grid: Grid) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": grid.shape[0], "width": grid.shape[1],
        "count": 1, "dtype": "float32", "crs": grid.crs,
        "transform": grid.transform, "nodata": np.nan,
        "compress": "deflate", "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


# ---------------------------------------------------------------------------
# Block aggregation of the fine reference onto a coarse grid
# ---------------------------------------------------------------------------


def _sortable_u32(v: np.ndarray) -> np.ndarray:
    """
    Monotonic float32 -> uint32 map: integer order matches float order, for
    negative values too. Lets a cell index and a value share one sort key.
    """
    u = v.astype(np.float32).view(np.uint32)
    neg = (u >> np.uint32(31)).astype(bool)
    return np.where(neg, ~u, u | np.uint32(0x80000000)).astype(np.uint64)


def aggregate_to_grid(src_path, grid: Grid, stat: str = "p95",
                      min_cover_frac: float = 0.5,
                      band_rows: int = BAND_TARGET_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate a fine raster onto a coarser grid by block statistic.

    Processing runs in bands of target rows: for each band only the source rows
    falling inside it are read, so peak memory is bounded by the band size
    rather than by the size of the mosaic. This matters for a full-flight
    0.15 m DSM, which can run to 10^9 pixels.

    Each fine pixel is assigned to the target cell containing its centre, so
    non-integer resolution ratios (1 / 0.15 = 6.67) are handled exactly.

    Returns (aggregated, cover_fraction).
    stat: "max" | "mean" | "sd" | "median" | "pXX" (e.g. "p95")
    """
    dt = grid.transform
    dst_h, dst_w = grid.shape
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    cover = np.zeros(grid.shape, dtype=np.float32)

    with rasterio.open(src_path) as src:
        _assert_axis_aligned(src.transform, src_path)
        st, nodata = src.transform, src.nodata
        expected = (grid.res / abs(st.a)) * (grid.res / abs(st.e))
        q = None if stat in ("max", "mean", "sd") else (
            0.5 if stat == "median" else float(stat[1:]) / 100.0)

        # Source columns covering the target grid: computed once, reused per band.
        c0 = int(np.floor((dt.c - st.c) / st.a))
        c1 = int(np.ceil((dt.c + dst_w * dt.a - st.c) / st.a))
        c0, c1 = max(c0, 0), min(c1, src.width)
        if c1 <= c0:
            return out, cover
        xs = st.c + (np.arange(c0, c1) + 0.5) * st.a
        cols = np.floor((xs - dt.c) / dt.a).astype(np.int64)
        col_ok = (cols >= 0) & (cols < dst_w)
        if not col_ok.any():
            return out, cover
        cols_v = cols[col_ok]

        for r0 in range(0, dst_h, band_rows):
            r1 = min(r0 + band_rows, dst_h)
            sr0 = int(np.floor((dt.f + r0 * dt.e - st.f) / st.e))
            sr1 = int(np.ceil((dt.f + r1 * dt.e - st.f) / st.e))
            sr0, sr1 = max(sr0, 0), min(sr1, src.height)
            if sr1 <= sr0:
                continue

            arr = src.read(1, window=Window(c0, sr0, c1 - c0, sr1 - sr0)).astype(np.float32)
            if nodata is not None:
                arr[arr == np.float32(nodata)] = np.nan
            arr = _gate_elevation(arr)
            arr = arr[:, col_ok]

            ys = st.f + (np.arange(sr0, sr1) + 0.5) * st.e
            rows = np.floor((ys - dt.f) / dt.e).astype(np.int64)
            row_ok = (rows >= r0) & (rows < r1)
            if not row_ok.any():
                continue
            arr = arr[row_ok]
            lin = (rows[row_ok] - r0)[:, None] * dst_w + cols_v[None, :]

            good = np.isfinite(arr)
            if not good.any():
                continue
            lin_v = lin[good].astype(np.int32)
            val_v = arr[good]

            n_local = (r1 - r0) * dst_w
            counts = np.bincount(lin_v, minlength=n_local)
            nz = counts > 0

            if stat in ("mean", "sd"):
                s1 = np.bincount(lin_v, weights=val_v.astype(np.float64),
                                 minlength=n_local)
                acc = np.full(n_local, np.nan, dtype=np.float64)
                if stat == "mean":
                    acc[nz] = s1[nz] / counts[nz]
                else:
                    s2 = np.bincount(lin_v,
                                     weights=(val_v.astype(np.float64) ** 2),
                                     minlength=n_local)
                    m = s1[nz] / counts[nz]
                    acc[nz] = np.sqrt(np.maximum(s2[nz] / counts[nz] - m * m, 0.0))
            else:
                # Sort once so every per-cell order statistic is an index
                # lookup. Cell index and value are packed into one uint64 key
                # (value mapped monotonically into the low 32 bits), so this is
                # a single sort rather than lexsort's two.
                key = ((lin_v.astype(np.uint64) << np.uint64(32))
                       | _sortable_u32(val_v))
                order = np.argsort(key, kind="stable")
                vs = val_v[order]
                starts = np.zeros(n_local, dtype=np.int64)
                np.cumsum(counts[:-1], out=starts[1:])
                base, c = starts[nz], counts[nz]
                acc = np.full(n_local, np.nan, dtype=np.float64)
                if stat == "max":
                    acc[nz] = vs[base + c - 1]
                else:
                    pos = (c - 1) * q                      # linear interpolation,
                    lo = np.floor(pos).astype(np.int64)    # matching np.percentile
                    hi = np.ceil(pos).astype(np.int64)
                    v_lo, v_hi = vs[base + lo], vs[base + hi]
                    acc[nz] = v_lo + (v_hi - v_lo) * (pos - lo)

            band_cover = counts / expected
            acc = np.where(band_cover >= min_cover_frac, acc, np.nan)
            out[r0:r1] = acc.reshape(r1 - r0, dst_w).astype(np.float32)
            cover[r0:r1] = band_cover.reshape(r1 - r0, dst_w).astype(np.float32)

    return out, cover

# ---------------------------------------------------------------------------
# Terrain and metrics
# ---------------------------------------------------------------------------


def erode_valid(err: np.ndarray, cells: int) -> np.ndarray:
    """
    Shrink the valid-comparison area inward by `cells`, so pixels near the
    footprint boundary are dropped. Uses 4-connectivity, applied iteratively;
    no scipy dependency.
    """
    if not cells:
        return err
    m = np.isfinite(err)
    for _ in range(int(cells)):
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = False   # image border too
        m = e
    return np.where(m, err, np.nan)


def slope_degrees(dem: np.ndarray, cell_size: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dy, dx = np.gradient(dem.astype(np.float64), cell_size, cell_size)
        return np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)


_METRIC_KEYS = ("n", "me", "mae", "rmse", "sd", "nmad", "le68", "le95", "p50",
                "blunder_frac")


def error_metrics(err: np.ndarray, clip_pct=ERROR_CLIP_PCT) -> dict:
    """Signed-error metrics. err = product - reference."""
    e = err[np.isfinite(err)]
    n_finite = e.size
    blunder_frac = 0.0
    if MAX_ABS_ERROR_M is not None and n_finite:
        keep = np.abs(e) <= MAX_ABS_ERROR_M
        blunder_frac = float(1.0 - keep.mean())
        e = e[keep]
    if e.size < 30:
        return {**{k: np.nan for k in _METRIC_KEYS}, "n": int(e.size),
                "blunder_frac": blunder_frac}
    if clip_pct is not None:
        lo, hi = np.percentile(e, clip_pct)
        e = e[(e >= lo) & (e <= hi)]
    med = np.median(e)
    abs_e = np.abs(e)
    return {
        "n": int(e.size),
        "me": float(np.mean(e)),
        "mae": float(np.mean(abs_e)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "sd": float(np.std(e, ddof=1)),
        "nmad": float(1.4826 * np.median(np.abs(e - med))),
        "le68": float(np.percentile(abs_e, 68.3)),
        "le95": float(np.percentile(abs_e, 95.0)),
        "p50": float(med),
        "blunder_frac": blunder_frac,
    }


def surface_stats(arr: np.ndarray) -> dict:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return {}
    return {
        "surf_mean": float(np.mean(a)),
        "surf_sd": float(np.std(a, ddof=1)),
        "surf_p05": float(np.percentile(a, 5)),
        "surf_p95": float(np.percentile(a, 95)),
    }


def slope_labels() -> list[str]:
    return [f"{SLOPE_BINS[i]:g}-{SLOPE_BINS[i+1]:g}"
            for i in range(len(SLOPE_BINS) - 1)]


def stratified_rows(err, strata, base: dict) -> list[dict]:
    """
    strata: list of (kind, array, bins, labels). Each is stratified
    independently, so the same pixel appears once per stratification scheme.
    """
    rows = [{**base, "stratum_kind": "all", "stratum": "all", **error_metrics(err)}]
    for kind, arr, bins, labels in strata:
        if arr is None:
            continue
        for i, lab in enumerate(labels):
            sel = (arr >= bins[i]) & (arr < bins[i + 1])
            rows.append({**base, "stratum_kind": kind, "stratum": lab,
                         "stratum_frac": float(np.isfinite(np.where(sel, err, np.nan)).mean()),
                         **error_metrics(np.where(sel, err, np.nan))})
    return rows


def subcell_sd(fine_path, grid: Grid) -> np.ndarray:
    """SD of fine-grid values within each coarse cell = variability discarded."""
    sd, _ = aggregate_to_grid(fine_path, grid, stat="sd",
                              min_cover_frac=MIN_COVER_FRAC)
    return sd


def _downsample(a, max_px):
    """Decimate for display only; keeps PNG size and render time sane."""
    step = max(1, int(np.ceil(max(a.shape) / max_px)))
    return a[::step, ::step], step


def save_diff_map(key: str, errs: dict, grids: dict, stat: str,
                  out_dir: Path) -> Path | None:
    """
    Spatial difference map per scene: where the two surfaces disagree, not just
    by how much. Blue = lidar below reference, red = lidar above. One panel per
    resolution on a shared colour scale, so panels are directly comparable.
    Structure here is diagnostic: blotches over lakes are SfM matching failure,
    ridges and gullies are true resolution loss, a uniform tint is residual
    datum offset, and a hard straight edge is a footprint or tiling artefact.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    res = sorted(errs)
    finite = np.concatenate([errs[r][np.isfinite(errs[r])].ravel() for r in res]) \
        if any(np.isfinite(errs[r]).any() for r in res) else np.array([])
    if finite.size == 0:
        return None
    vmax = (float(DIFF_MAP_CLIP_M) if DIFF_MAP_CLIP_M
            else float(np.percentile(np.abs(finite), 98)) or 1.0)

    fig, axes = plt.subplots(1, len(res), figsize=(4.1 * len(res), 4.4),
                             squeeze=False)
    for j, r in enumerate(res):
        ax = axes[0][j]
        a, step = _downsample(errs[r], DIFF_MAP_MAX_PX)
        g = grids[r]
        w = g.shape[1] * g.res / 1000.0
        h = g.shape[0] * g.res / 1000.0
        im = ax.imshow(np.ma.masked_invalid(a), cmap="RdBu_r", vmin=-vmax,
                       vmax=vmax, extent=(0, w, 0, h), interpolation="nearest")
        e = errs[r][np.isfinite(errs[r])]
        if e.size:
            med = float(np.median(e))
            nmad = 1.4826 * np.median(np.abs(e - med))
            # median, not mean: a handful of blunders drags the mean far outside
            # the colour range and makes the panel titles unreadable
            ax.set_title("%g m  |  NMAD %.2f m  |  median %+.2f m"
                         % (r, nmad, med), fontsize=9)
        ax.set_xlabel("km")
        if j == 0:
            ax.set_ylabel("km")
        ax.set_facecolor("0.9")            # NaN shows as grey = no comparison
    cb = fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02)
    cb.set_label("lidar DSM  −  SfM reference   (m)"
                 + ("  [clipped at ±%g m]" % vmax if DIFF_MAP_CLIP_M else ""),
                 fontsize=8)
    fig.suptitle("%s\ndifference map, offset removed, agg stat %s   "
                 "(red = lidar higher, blue = lidar lower, grey = no data)"
                 % (key, stat), fontsize=10)
    path = out_dir / "diff_maps" / ("%s_%s.png" % (key, stat))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DIFF_MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------


def process_scene(entry: dict, stat: str) -> list[dict]:
    key, meta, ref_path = entry["key"], entry["meta"], entry["ref"]
    paths = entry["res"]

    grids = {r: read_grid(paths[r]["DSM"]) for r in RESOLUTIONS}
    ref_grid_native = read_grid(ref_path)

    if ref_grid_native.res > min(RESOLUTIONS) / 2:
        raise ValueError(f"reference cell {ref_grid_native.res} m is not clearly "
                         f"finer than the finest target grid "
                         f"({min(RESOLUTIONS)} m)")
    for r in RESOLUTIONS:
        if abs(grids[r].res - r) > CELL_SIZE_TOL:
            raise ValueError(f"{RES_DIRS[r]} cell size is {grids[r].res} m, "
                             f"expected {r} m")
        if not same_crs(grids[r].crs, ref_grid_native.crs):
            raise ValueError(f"CRS mismatch: {r} m grid is {grids[r].crs.to_string()}, "
                             f"reference is {ref_grid_native.crs.to_string()}")

    tmpdir = tempfile.mkdtemp(prefix="resval_")
    try:
        ref_agg = {}
        for r in RESOLUTIONS:
            ref_agg[r], _ = aggregate_to_grid(ref_path, grids[r], stat=stat,
                                              min_cover_frac=MIN_COVER_FRAC)

        base_grid = grids[REF_RES]
        dtm_aligned = grids_match(paths[REF_RES]["DTM"], base_grid)
        dtm_ref = align_raster_to_grid(paths[REF_RES]["DTM"], base_grid,
                                       Resampling.bilinear)
        slope_ref = slope_degrees(dtm_ref, base_grid.res)
        slope = {r: (slope_ref if r == REF_RES else
                     resample_array_to_grid(slope_ref, base_grid, grids[r],
                                            Resampling.average))
                 for r in RESOLUTIONS}

        # Vegetation height from the finest lidar CHM, held fixed across
        # resolutions so the strata are the same pixels at every cell size.
        dsm_base = read_array(paths[REF_RES]["DSM"])
        veg_ref = dsm_base - dtm_ref
        veg = {r: (veg_ref if r == REF_RES else
                   resample_array_to_grid(veg_ref, base_grid, grids[r],
                                          Resampling.average))
               for r in RESOLUTIONS}

        # Vertical offset, estimated once at the finest resolution.
        offset_scene, n_stable = 0.0, 0
        if OFFSET_MODE != "none":
            d = dsm_base - ref_agg[REF_RES]
            stable = (np.isfinite(d) & (slope_ref < OFFSET_MAX_SLOPE_DEG)
                      & (np.abs(veg_ref) < OFFSET_MAX_CHM_M))
            n_stable = int(stable.sum())
            if n_stable >= 100:
                offset_scene = float(np.median(d[stable]))

        rows: list[dict] = []
        diff_maps: dict = {}
        for r in RESOLUTIONS:
            grid, ref, slp, vg = grids[r], ref_agg[r], slope[r], veg[r]
            dsm = read_array(paths[r]["DSM"])
            dtm_matched = align_raster_to_grid(paths[r]["DTM"], grid,
                                               Resampling.bilinear)
            if dsm.shape != grid.shape:
                raise ValueError(f"{r} m DSM shape {dsm.shape} does not match its "
                                 f"own grid {grid.shape}")

            dtm_fixed = (dtm_ref if r == REF_RES else
                         resample_array_to_grid(dtm_ref, base_grid, grid,
                                                Resampling.average))

            offset = offset_scene
            if OFFSET_MODE == "per_resolution":
                d0 = dsm - ref
                sel = np.isfinite(d0) & (slp < OFFSET_MAX_SLOPE_DEG)
                offset = float(np.median(d0[sel])) if sel.sum() >= 100 else 0.0

            products = {
                "DSM": (dsm - offset) - ref,
                "CHM_fixed_dtm": ((dsm - offset) - dtm_fixed) - (ref - dtm_fixed),
                "CHM_matched_dtm": ((dsm - offset) - dtm_matched) - (ref - dtm_fixed),
                "DTM_vs_finest": dtm_matched - dtm_fixed,
            }

            if WRITE_DIFF_MAPS and stat == AGG_STAT_PRIMARY:
                diff_maps[r] = products["DSM"].copy()

            if ERODE_EDGE_CELLS:
                products = {k2: erode_valid(v2, ERODE_EDGE_CELLS)
                            for k2, v2 in products.items()}

            strata = [("slope_deg", slp, SLOPE_BINS, slope_labels()),
                      ("veg_height", vg, VEG_BINS, VEG_LABELS)]

            for product, err in products.items():
                base = {"scene": key, **meta, "resolution_m": r, "product": product,
                        "agg_stat": stat, "offset_m": offset,
                        "n_stable_offset": n_stable, "dtm_on_dsm_grid": dtm_aligned,
                        "valid_frac": float(np.isfinite(err).mean())}
                if product == "DSM":
                    base.update(surface_stats(dsm))
                rows.extend(stratified_rows(err, strata, base))

                if WRITE_DIFF_RASTERS:
                    write_raster(OUT_DIR / "diff" / f"{key}_{product}_{r:g}m_{stat}.tif",
                                 err, grid)

            if r != REF_RES:
                sd = subcell_sd(paths[REF_RES]["DSM"], grid)
                rows.append({"scene": key, **meta, "resolution_m": r,
                             "product": "DSM_subcell_sd", "agg_stat": stat,
                             "stratum_kind": "all", "stratum": "all",
                             "n": int(np.isfinite(sd).sum()),
                             "me": float(np.nanmean(sd)),
                             "p50": float(np.nanmedian(sd)),
                             "le95": float(np.nanpercentile(sd, 95))})

        if diff_maps:
            save_diff_map(key, diff_maps, grids, stat, OUT_DIR / "plots")

        # Aggregation floor: reference -> REF_RES -> r, versus reference -> r.
        floor_src = os.path.join(tmpdir, "ref_fine.tif")
        write_raster(floor_src, ref_agg[REF_RES], base_grid)
        for r in RESOLUTIONS:
            if r == REF_RES:
                continue
            two_step, _ = aggregate_to_grid(floor_src, grids[r], stat=stat,
                                            min_cover_frac=MIN_COVER_FRAC)
            rows.append({"scene": key, **meta, "resolution_m": r,
                         "product": "AGGREGATION_FLOOR", "agg_stat": stat,
                         "stratum_kind": "all", "stratum": "all",
                         **error_metrics(two_step - ref_agg[r])})
        return rows
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def scene_csv_path(key: str) -> Path:
    return OUT_DIR / "scenes" / f"{key}.csv"


def run_scene(args) -> tuple[str, bool, str, str]:
    """Returns (key, ok, short_message, full_traceback)."""
    key, entry, stats = args
    t0 = time.perf_counter()
    try:
        rows = []
        for stat in stats:
            rows.extend(process_scene(entry, stat))
        out = scene_csv_path(key)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        return key, True, f"ok ({time.perf_counter() - t0:.0f}s)", ""
    except Exception as exc:
        short = f"{type(exc).__name__}: {exc}"
        return key, False, short.replace("\n", " ")[:200], traceback.format_exc()



# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

PLOT_PRODUCTS = ["DSM", "CHM_matched_dtm", "DTM_vs_finest"]
PLOT_LABELS = {
    "DSM": "DSM\nlidar $-$ SfM reference",
    "CHM_matched_dtm": "CHM\n(lidar DSM$-$DTM) $-$ (SfM$-$DTM)",
    "DTM_vs_finest": "DTM\ncoarse $-$ finest DTM",
}
CAPTION = ("Reference: MACS SfM DSM at native GSD, block-aggregated to each "
           "target grid (%s). Sign: positive = product above reference. A "
           "per-scene vertical offset is removed first.")


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop scenes whose footprints barely overlap the reference."""
    if MIN_VALID_FRAC is None or "valid_frac" not in df.columns:
        return df
    ok = (df[df["product"] == "DSM"]
          .groupby("scene")["valid_frac"].max() >= MIN_VALID_FRAC)
    return df[df["scene"].isin(ok[ok].index)]


def _scene_level(df: pd.DataFrame, stat: str | None = None) -> pd.DataFrame:
    stat = stat or AGG_STAT_PRIMARY
    return usable(df[(df.stratum_kind == "all") & (df.agg_stat == stat)])


def _jitter(x, n, width=0.06, seed=0):
    return np.asarray(x, float) * (1 + np.random.default_rng(seed).uniform(
        -width, width, size=n))


def _res_axis(ax, resolutions):
    ax.set_xscale("log")
    ax.set_xticks(sorted(resolutions))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    # log minor ticks label themselves as 3x10^0 etc. and clutter a 3-point axis
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(alpha=0.3)


def _spread_panel(ax, sub, metric, colour="C0", label=None):
    """Per-scene points plus the median line."""
    res = sorted(sub["resolution_m"].unique())
    for i, r in enumerate(res):
        v = sub.loc[sub.resolution_m == r, metric].dropna()
        if v.empty:
            continue
        ax.scatter(_jitter([r] * len(v), len(v), seed=i), v, s=7, alpha=0.35,
                   color=colour, linewidths=0, zorder=2)
    med = sub.groupby("resolution_m")[metric].median().reindex(res)
    ax.plot(res, med.values, "-o", color=colour, lw=2, ms=5, zorder=3, label=label)
    return med


def plot_error_vs_resolution(df, out_dir):
    """Core figure: each error metric against cell size, one column per product."""
    sel = _scene_level(df)
    metrics = [("rmse", "RMSE (m)"), ("nmad", "NMAD (m)"), ("le95", "LE95 (m)")]
    prods = [p for p in PLOT_PRODUCTS if p in set(sel["product"])]
    if not prods:
        return None
    floor = sel[sel["product"] == "AGGREGATION_FLOOR"]
    res_all = sorted(sel["resolution_m"].unique())

    fig, axes = plt.subplots(len(metrics), len(prods),
                             figsize=(3.6 * len(prods), 2.9 * len(metrics)),
                             sharex=True, squeeze=False)
    for j, prod in enumerate(prods):
        sub_p = sel[sel["product"] == prod]
        for i, (metric, ylab) in enumerate(metrics):
            ax = axes[i][j]
            _spread_panel(ax, sub_p, metric, colour="C%d" % j)
            if not floor.empty:
                f = floor.groupby("resolution_m")[metric].median().dropna()
                if not f.empty:
                    ax.plot(f.index, f.values, "--o", color="0.4", lw=1.4,
                            ms=4, label="aggregation floor"
                            if (i == 0 and j == 0) else None)
            _res_axis(ax, res_all)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(PLOT_LABELS.get(prod, prod), fontsize=10)
            if i == len(metrics) - 1:
                ax.set_xlabel("cell size (m)")
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Error vs resolution, %d scenes; points are scenes, "
                 "line is the median" % sel["scene"].nunique(), fontsize=11)
    fig.text(0.5, 0.925, CAPTION % AGG_STAT_PRIMARY, ha="center", fontsize=7.5,
             color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    return _save(fig, out_dir / "01_error_vs_resolution.png")


def plot_slope_strata(df, out_dir):
    """The resolution effect concentrates on slopes; this is where to see it."""
    sub = usable(df[(df.stratum_kind == "slope_deg")
                    & (df.agg_stat == AGG_STAT_PRIMARY)])
    if sub.empty:
        return None
    prods = [p for p in PLOT_PRODUCTS if p in set(sub["product"])]
    if not prods:
        return None
    strata = _ordered_strata(sub, "slope_deg")
    cmap = plt.get_cmap("viridis")
    res_all = sorted(sub["resolution_m"].unique())

    fig, axes = plt.subplots(1, len(prods), figsize=(3.7 * len(prods), 3.3),
                             sharey=True, squeeze=False)
    for j, prod in enumerate(prods):
        ax = axes[0][j]
        for k, st in enumerate(strata):
            m = sub[(sub["product"] == prod) & (sub.stratum == st)]
            med = m.groupby("resolution_m")["nmad"].median().dropna()
            if med.empty:
                continue
            ax.plot(med.index, med.values, "-o", ms=4,
                    color=cmap(k / max(len(strata) - 1, 1)), label=st + " deg")
        _res_axis(ax, res_all)
        ax.set_xlabel("cell size (m)")
        ax.set_title(PLOT_LABELS.get(prod, prod), fontsize=10)
        if j == 0:
            ax.set_ylabel("NMAD (m)")
    axes[0][-1].legend(title="slope", fontsize=7, title_fontsize=7, frameon=False)
    fig.suptitle("Error by terrain slope class (median across scenes)",
                 fontsize=11)
    fig.text(0.5, 0.9, "Slope from the finest DTM. NOTE: the flattest class "
             "also contains water and wet tundra, where SfM matching fails - "
             "a high 0-2 deg curve is a surface-type effect, not terrain.",
             ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, out_dir / "02_slope_strata.png")


def plot_scene_ranking(df, out_dir):
    """Which scenes drive the pooled numbers, and whether any are outliers."""
    sel = _scene_level(df)
    sel = sel[sel["product"] == "DSM"]
    if sel.empty:
        return None
    r_max = sel["resolution_m"].max()
    s = (sel[sel.resolution_m == r_max]
         .set_index("scene")["rmse"].dropna().sort_values())
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.17 * len(s))))
    ax.barh(np.arange(len(s)), s.values, color="C0", height=0.75)
    ax.axvline(s.median(), color="0.3", ls="--", lw=1.2,
               label="median %.3f m" % s.median())
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(s.index, fontsize=6)
    ax.set_xlabel("DSM RMSE at %g m (m)" % r_max)
    ax.set_title("Per-scene DSM RMSE at %g m" % r_max, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / "03_scene_ranking.png")


def plot_offsets(df, out_dir):
    """
    Sanity check on the vertical offset: it should be small, and estimated from
    a decent number of stable pixels. Scenes sitting at exactly 0 fell back
    because too little flat bare ground was found, and their bias is not
    comparable with the rest.
    """
    if "offset_m" not in df.columns:
        return None
    s = (df.dropna(subset=["offset_m"])
           .groupby("scene")[["offset_m", "n_stable_offset"]].first())
    if s.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    axes[0].hist(s["offset_m"], bins=min(25, max(5, len(s))), color="C2",
                 edgecolor="white")
    axes[0].set_xlabel("vertical offset (m)")
    axes[0].set_ylabel("scenes")
    axes[0].set_title("Offset removed per scene", fontsize=10)
    axes[0].grid(alpha=0.3)

    fallback = s["n_stable_offset"] < 100
    axes[1].scatter(s.loc[~fallback, "n_stable_offset"],
                    s.loc[~fallback, "offset_m"], s=14, color="C2")
    if fallback.any():
        axes[1].scatter(s.loc[fallback, "n_stable_offset"],
                        s.loc[fallback, "offset_m"], s=30, color="C3",
                        marker="x", label="fallback to 0")
        axes[1].legend(frameon=False, fontsize=8)
    n = s["n_stable_offset"].replace(0, np.nan).dropna()
    if not n.empty and n.max() / max(n.min(), 1) > 20:
        axes[1].set_xscale("log")
    axes[1].set_xlabel("stable pixels used")
    axes[1].set_ylabel("offset (m)")
    axes[1].set_title("Offset vs sample size", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir / "04_offsets.png")


def plot_subcell_sd(df, out_dir):
    """Variability discarded by each grid: what coarsening actually throws away."""
    sub = df[(df["product"] == "DSM_subcell_sd")
             & (df.agg_stat == AGG_STAT_PRIMARY)]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    _spread_panel(ax, sub, "me", colour="C4", label="mean sub-cell SD")
    med = sub.groupby("resolution_m")["le95"].median().dropna()
    if not med.empty:
        ax.plot(med.index, med.values, "--s", color="C4", ms=4, lw=1.2, alpha=0.7,
                label="95th percentile")
    _res_axis(ax, sorted(sub["resolution_m"].unique()))
    ax.set_xlabel("cell size (m)")
    ax.set_ylabel("within-cell SD of finest DSM (m)")
    ax.set_title("Variability discarded by coarsening", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / "05_subcell_sd.png")


def plot_stat_sensitivity(df, out_dir):
    """
    How much of the measured effect is the aggregation statistic rather than
    the resolution. If these diverge, the statistic matters more than the grid.
    """
    stats = sorted(df["agg_stat"].dropna().unique())
    if len(stats) < 2:
        return None
    sel = df[(df.stratum_kind == "all") & (df["product"] == "DSM")]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for k, st in enumerate(stats):
        med = sel[sel.agg_stat == st].groupby("resolution_m")["rmse"].median()
        ax.plot(med.index, med.values, "-o", ms=5, label=st, color="C%d" % k)
    _res_axis(ax, sorted(sel["resolution_m"].unique()))
    ax.set_xlabel("cell size (m)")
    ax.set_ylabel("DSM RMSE (m)")
    ax.set_title("Sensitivity to the aggregation statistic", fontsize=10)
    ax.legend(frameon=False, fontsize=8, title="statistic", title_fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / "06_stat_sensitivity.png")


def _quality_matrices(sel, metric):
    """Shared scene ordering and per-product matrices for the overview figures."""
    prods = [p for p in PLOT_PRODUCTS if p in set(sel["product"])]
    res = sorted(sel["resolution_m"].unique())

    def mat(sub, col):
        return (sub.pivot_table(index="scene", columns="resolution_m",
                                values=col, aggfunc="median")
                   .reindex(columns=res))

    dsm = mat(sel[sel["product"] == "DSM"], metric)
    if dsm.empty:
        return prods, res, [], []
    order = list(dsm[res[-1]].sort_values(ascending=False).index)
    order += [s for s in dsm.index if s not in set(order)]
    panels = [(p, mat(sel[sel["product"] == p], metric)) for p in prods]
    return prods, res, order, panels


def _matrix_figure(panels, order, res, title, out_path, metric,
                   relative=False, coverage=None):
    n = len(order)
    ncol = len(panels) + (1 if coverage is not None else 0)
    fig, axes = plt.subplots(1, ncol,
                             figsize=(2.5 * ncol + 1.2, max(3.2, 0.16 * n + 1.6)),
                             sharey=True, squeeze=False)
    items = list(panels) + ([("valid data fraction", coverage)]
                            if coverage is not None else [])

    for j, (name, m) in enumerate(items):
        ax = axes[0][j]
        m = m.reindex(order)
        is_cov = name.startswith("valid")
        arr = m.to_numpy(dtype=float)

        if relative and not is_cov:
            with np.errstate(invalid="ignore", divide="ignore"):
                med = np.nanmedian(arr, axis=0)            # per-resolution median
                arr = np.log2(arr / med)
        masked = np.ma.masked_invalid(arr)

        if is_cov:
            cmap = plt.get_cmap("viridis").copy(); vmin, vmax = 0.0, 1.0
        elif relative:
            cmap = plt.get_cmap("RdBu_r").copy()
            lim = float(np.nanpercentile(np.abs(masked.compressed()), 98)) \
                if masked.count() else 1.0
            vmin, vmax = -max(lim, 0.1), max(lim, 0.1)
        else:
            cmap = plt.get_cmap("magma_r").copy()
            f = masked.compressed()
            vmin, vmax = 0.0, (float(np.percentile(f, 98)) if f.size else 1.0)
        cmap.set_bad("0.85")

        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(len(res)))
        ax.set_xticklabels(["%g" % r for r in res])
        ax.set_xlabel("cell size (m)")
        ax.set_title(name if is_cov else PLOT_LABELS.get(name, name), fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        if is_cov:
            pass
        elif relative:
            cb.set_label("log2(scene / resolution median)", fontsize=6)
        else:
            cb.set_label(metric.upper() + " (m)", fontsize=7)

        if n <= 25:                                  # annotate only if legible
            for a in range(masked.shape[0]):
                for b in range(masked.shape[1]):
                    if masked.mask[a, b]:
                        continue
                    v = masked[a, b]
                    rel = abs(v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                    dark = rel > 0.75 or (relative and abs(v) > 0.7 * vmax)
                    ax.text(b, a, ("%.2f" if is_cov or relative else "%.3f") % v,
                            ha="center", va="center", fontsize=6,
                            color="white" if dark else "black")

    axes[0][0].set_yticks(range(n))
    axes[0][0].set_yticklabels(order,
                               fontsize=max(4, min(7, 420 / max(n, 1))))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_path)


def plot_quality_matrix(df, out_dir, metric="nmad"):
    """
    Every raster at once: scenes down the rows, resolutions across the columns,
    one panel per product plus data coverage. Absolute error, so the dominant
    pattern is the resolution effect (columns getting darker). Scenes are
    ordered worst-first at the coarsest grid and grey cells are missing
    combinations.
    """
    sel = _scene_level(df)
    if sel.empty:
        return None
    prods, res, order, panels = _quality_matrices(sel, metric)
    if not panels:
        return None
    cov = (sel[sel["product"] == "DSM"]
           .pivot_table(index="scene", columns="resolution_m",
                        values="valid_frac", aggfunc="median")
           .reindex(columns=res))
    return _matrix_figure(
        panels, order, res,
        "Quality of every raster across resolutions (%d scenes, agg stat: %s)"
        % (len(order), AGG_STAT_PRIMARY),
        out_dir / "07_quality_matrix.png", metric,
        relative=False, coverage=cov)


def plot_quality_matrix_relative(df, out_dir, metric="nmad"):
    """
    The same matrix normalised to each resolution's median, so the resolution
    effect is divided out and what remains is the scene effect. A scene that is
    consistently red is a problem raster regardless of grid; a cell that is red
    at one resolution only points at that specific product.
    """
    sel = _scene_level(df)
    if sel.empty:
        return None
    prods, res, order, panels = _quality_matrices(sel, metric)
    if not panels or len(order) < 3:
        return None
    return _matrix_figure(
        panels, order, res,
        "Scene quality relative to the median at each resolution "
        "(%d scenes; red = worse than typical)" % len(order),
        out_dir / "08_quality_matrix_relative.png", metric, relative=True)


def plot_decision(df, out_dir):
    """
    The decision figure. Everything else describes the data; this one answers
    "which grid can I use". Plotted as the error ADDED by coarsening relative
    to the 1 m grid, because the constant lidar-vs-SfM disagreement at 1 m is
    not a cost of coarsening and should not be charged to it. A resolution is
    acceptable for a slope class where its bar stays under the tolerance line.
    """
    sub = usable(df[(df.stratum_kind == "slope_deg")
                    & (df.agg_stat == AGG_STAT_PRIMARY)
                    & (df["product"] == "DSM")])
    if sub.empty:
        return None
    res = sorted(sub["resolution_m"].unique())
    base_r = min(res)
    strata = _ordered_strata(sub, "slope_deg")

    med = (sub.groupby(["stratum", "resolution_m"])["nmad"].median()
              .unstack("resolution_m").reindex(strata))
    added = med.sub(med[base_r], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    width = 0.8 / max(len(res) - 1, 1)
    x = np.arange(len(strata))
    for k, r in enumerate([r for r in res if r != base_r]):
        ax.bar(x + k * width - 0.4 + width / 2, added[r].values, width * 0.92,
               label="%g m" % r, color=plt.get_cmap("plasma")(0.25 + 0.45 * k))
    ax.axhline(TOLERANCE_NMAD_M, color="crimson", ls="--", lw=1.6,
               label="tolerance %.2f m" % TOLERANCE_NMAD_M)
    ax.set_xticks(x); ax.set_xticklabels([s + "\u00b0" for s in strata])
    ax.set_xlabel("terrain slope class")
    ax.set_ylabel("NMAD added vs %g m grid (m)" % base_r)
    ax.set_title("Cost of coarsening, by terrain", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    tot = (usable(_scene_level(df))
           .query("product == 'DSM'")
           .groupby("resolution_m")["nmad"].median().reindex(res))
    ax.plot(res, tot.values, "-o", color="C0", lw=2, label="total disagreement")
    ax.plot(res, (tot - tot[base_r]).values, "-s", color="C3", lw=2,
            label="attributable to coarsening")
    ax.axhline(tot[base_r], color="0.5", ls=":", lw=1.4,
               label="lidar vs SfM floor (%.2f m)" % tot[base_r])
    ax.axhline(TOLERANCE_NMAD_M, color="crimson", ls="--", lw=1.4,
               label="tolerance")
    _res_axis(ax, res)
    ax.set_xlabel("cell size (m)")
    ax.set_ylabel("DSM NMAD (m)")
    ax.set_title("Total vs coarsening-attributable error", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)

    fig.suptitle("Decision view: how much extra vertical error each grid costs "
                 "(%d scenes, median)" % _scene_level(df)["scene"].nunique(),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_dir / "09_decision.png")


def plot_sample_selection(df, out_dir):
    """
    Who is actually in the sample, and is the surviving subset representative.
    The coverage filter is not random: overlap correlates with terrain, so the
    retained scenes can be systematically harder than the discarded ones. If
    the two NMAD distributions on the right differ, say so in the paper.
    """
    if "valid_frac" not in df.columns:
        return None
    sel = df[(df.stratum_kind == "all") & (df.agg_stat == AGG_STAT_PRIMARY)
             & (df["product"] == "DSM")]
    if sel.empty:
        return None
    r_max = sel["resolution_m"].max()
    at = sel[sel.resolution_m == r_max].set_index("scene")
    vf, nm = at["valid_frac"], at["nmad"]
    keep = vf >= (MIN_VALID_FRAC or 0)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))

    axes[0].hist(vf.dropna(), bins=20, color="C0", edgecolor="white")
    if MIN_VALID_FRAC:
        axes[0].axvline(MIN_VALID_FRAC, color="crimson", ls="--", lw=1.6,
                        label="MIN_VALID_FRAC")
        axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_xlabel("fraction of cells with a valid comparison")
    axes[0].set_ylabel("scenes")
    axes[0].set_title("Footprint overlap (%d scenes)" % len(vf), fontsize=10)

    axes[1].scatter(vf[keep], nm[keep], s=18, color="C0", label="kept")
    axes[1].scatter(vf[~keep], nm[~keep], s=18, color="C3", marker="x",
                    label="dropped")
    axes[1].set_xlabel("valid fraction")
    axes[1].set_ylabel("DSM NMAD at %g m (m)" % r_max)
    axes[1].set_title("Does overlap predict error?", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)

    data = [nm[keep].dropna(), nm[~keep].dropna()]
    labels = ["kept\n(n=%d)" % len(data[0]), "dropped\n(n=%d)" % len(data[1])]
    data = [d for d in data if len(d)]
    if data:
        axes[2].boxplot(data, tick_labels=labels[:len(data)], showfliers=False)
        for i, d in enumerate(data, 1):
            axes[2].scatter(np.full(len(d), i) + np.random.default_rng(0)
                            .uniform(-.07, .07, len(d)), d, s=12, alpha=0.5,
                            color="C0")
    axes[2].set_ylabel("DSM NMAD at %g m (m)" % r_max)
    axes[2].set_title("Is the kept subset representative?", fontsize=10)
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle("Sample selection: which scenes reach the summary, and whether "
                 "that biases the result", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out_dir / "10_sample_selection.png")


def plot_blunders(df, out_dir):
    """
    RMSE/NMAD ratio is a pure outlier meter: ~1.05 for clean Gaussian error,
    higher means the tail is doing the work. Use it to decide whether RMSE can
    be quoted at all, and to tune MAX_ABS_ERROR_M.
    """
    sel = _scene_level(df)
    sel = sel[sel["product"] == "DSM"]
    if sel.empty or "blunder_frac" not in sel.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    res = sorted(sel["resolution_m"].unique())

    ratio = (sel.assign(ratio=sel["rmse"] / sel["nmad"])
                .groupby("resolution_m")["ratio"])
    axes[0].plot(res, ratio.median().reindex(res).values, "-o", color="C1", lw=2)
    axes[0].axhline(1.05, color="0.4", ls="--", lw=1.3,
                    label="clean Gaussian (~1.05)")
    axes[0].set_xlabel("cell size (m)"); _res_axis(axes[0], res)
    axes[0].set_ylabel("RMSE / NMAD")
    axes[0].set_title("Outlier influence on RMSE", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    b = sel.groupby("resolution_m")["blunder_frac"].median().reindex(res) * 100
    axes[1].bar([str(r) for r in res], b.values, color="C3", width=0.6)
    axes[1].set_xlabel("cell size (m)")
    axes[1].set_ylabel("cells excluded as blunders (%)")
    axes[1].set_title("Excluded by |error| > %g m" % (MAX_ABS_ERROR_M or 0),
                      fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Blunder diagnostics: can RMSE be quoted, or only NMAD?",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, out_dir / "11_blunders.png")


def _ordered_strata(sub, kind):
    """Config order, not alphabetical, so bins read low-to-high."""
    order = VEG_LABELS if kind == "veg_height" else slope_labels()
    present = set(sub["stratum"].unique())
    return [x for x in order if x in present]


def plot_veg_strata(df, out_dir):
    """
    The canopy test. Difference maps show the error living in shrub networks,
    not on bare terrain, so slope alone confounds two mechanisms. If the bare
    curve is flat while the shrub curves climb, what looks like a resolution
    effect is largely a canopy-representation effect: a max-type lidar
    aggregate rises with more samples per cell, the SfM envelope does not.
    """
    sub = usable(df[(df.stratum_kind == "veg_height")
                    & (df.agg_stat == AGG_STAT_PRIMARY)])
    if sub.empty:
        return None
    prods = [p for p in PLOT_PRODUCTS if p in set(sub["product"])]
    if not prods:
        return None
    strata = _ordered_strata(sub, "veg_height")
    cmap = plt.get_cmap("YlGn")
    res_all = sorted(sub["resolution_m"].unique())

    fig, axes = plt.subplots(1, len(prods) + 1,
                             figsize=(3.7 * (len(prods) + 1), 3.4),
                             squeeze=False)
    for j, prod in enumerate(prods):
        ax = axes[0][j]
        for k, st in enumerate(strata):
            m = sub[(sub["product"] == prod) & (sub.stratum == st)]
            med = m.groupby("resolution_m")["nmad"].median().dropna()
            if med.empty:
                continue
            ax.plot(med.index, med.values, "-o", ms=4, label=st,
                    color=cmap(0.30 + 0.65 * k / max(len(strata) - 1, 1)))
        _res_axis(ax, res_all)
        ax.set_xlabel("cell size (m)")
        ax.set_title(PLOT_LABELS.get(prod, prod), fontsize=9)
        if j == 0:
            ax.set_ylabel("NMAD (m)")
    axes[0][0].legend(title="canopy height", fontsize=7, title_fontsize=7,
                      frameon=False)

    # how much of the scene each stratum covers - a class with 1% of pixels
    # driving the story would be a red flag
    ax = axes[0][-1]
    frac = (sub[sub["product"] == "DSM"]
            .groupby("stratum")["stratum_frac"].median().reindex(strata) * 100)
    ax.barh(np.arange(len(strata)), frac.values,
            color=[cmap(0.30 + 0.65 * k / max(len(strata) - 1, 1))
                   for k in range(len(strata))])
    ax.set_yticks(np.arange(len(strata))); ax.set_yticklabels(strata, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("% of valid cells")
    ax.set_title("Stratum size", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Error by canopy height: is the resolution effect terrain "
                 "or vegetation?", fontsize=11)
    fig.text(0.5, 0.9, "Canopy height from the finest lidar CHM (DSM\u2212DTM), "
             "held fixed across resolutions. Bare-ground curve flat + shrub "
             "curves rising = canopy effect, not terrain detail loss.",
             ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, out_dir / "12_veg_strata.png")


def plot_terrain_vs_canopy(df, out_dir):
    """
    Attribution in one panel: added NMAD relative to the 1 m grid, split by
    what the pixel sits on. Bars are the coarsening cost on bare ground (the
    terrain effect) against tall vegetation (terrain + canopy). The gap between
    them is what a max-type aggregate adds over canopy, and it is the part that
    would disappear under a median aggregate.
    """
    sub = usable(df[(df.stratum_kind == "veg_height")
                    & (df.agg_stat == AGG_STAT_PRIMARY)
                    & (df["product"] == "DSM")])
    if sub.empty:
        return None
    res = sorted(sub["resolution_m"].unique())
    base_r = min(res)
    strata = _ordered_strata(sub, "veg_height")
    med = (sub.groupby(["stratum", "resolution_m"])["nmad"].median()
              .unstack("resolution_m").reindex(strata))
    if base_r not in med.columns:
        return None
    added = med.sub(med[base_r], axis=0)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    coarse = [r for r in res if r != base_r]
    width = 0.8 / max(len(coarse), 1)
    x = np.arange(len(strata))
    for k, r in enumerate(coarse):
        ax.bar(x + k * width - 0.4 + width / 2, added[r].values, width * 0.92,
               label="%g m" % r, color=plt.get_cmap("plasma")(0.25 + 0.45 * k))
    ax.axhline(TOLERANCE_NMAD_M, color="crimson", ls="--", lw=1.6,
               label="tolerance %.2f m" % TOLERANCE_NMAD_M)
    ax.set_xticks(x); ax.set_xticklabels(strata, fontsize=8)
    ax.set_xlabel("surface type (canopy height)")
    ax.set_ylabel("NMAD added vs %g m grid (m)" % base_r)
    ax.set_title("Cost of coarsening: terrain vs canopy", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir / "13_terrain_vs_canopy.png")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    global plt, mticker
    try:
        import matplotlib
        matplotlib.use("Agg")              # headless: no display on the nodes
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not installed; skipping plots (pixi add matplotlib)")
        return

    plt.rcParams.update({"font.size": 9, "figure.facecolor": "white"})
    plots = out_dir / "plots"
    made = []
    for fn in (plot_error_vs_resolution, plot_slope_strata, plot_scene_ranking,
               plot_offsets, plot_subcell_sd, plot_stat_sensitivity,
               plot_quality_matrix, plot_quality_matrix_relative,
               plot_decision, plot_sample_selection, plot_blunders,
               plot_veg_strata, plot_terrain_vs_canopy):
        try:
            p = fn(df, plots)
            if p is not None:
                made.append(p)
        except Exception as exc:
            print("  plot %s failed: %s: %s" % (fn.__name__, type(exc).__name__, exc))
    print("\nwrote %d figures -> %s" % (len(made), plots))
    for p in made:
        print("  " + p.name)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if PLOT_ONLY:
        csv = OUT_DIR / "resolution_metrics_long.csv"
        if not csv.is_file():
            raise SystemExit(f"PLOT_ONLY is set but {csv} does not exist yet")
        make_plots(pd.read_csv(csv), OUT_DIR)
        return

    scenes, inv = discover_scenes()

    inv_path = OUT_DIR / "scene_inventory.csv"
    inv.to_csv(inv_path, index=False)
    n_total, n_ok = len(inv), int(inv["complete"].sum())
    print(f"{n_total} scene keys found, {n_ok} complete across all resolutions")
    print("\navailability by product:")
    print("\n".join(availability(inv)))
    if n_ok == 0:
        print("\nNothing is complete. Configured paths:")
        print("\n".join(check_roots()))
        print()
        print("\n".join(explain_scene(inv["scene"].iloc[0])))
        print(f"\ninventory -> {inv_path}")
        return
    if n_total != n_ok:
        print("\nincomplete scenes (first 20; full list in scene_inventory.csv):")
        print(inv.loc[~inv["complete"], "scene"].head(20).to_string(index=False))
    print(f"inventory -> {inv_path}\n")

    if INVENTORY_ONLY:
        return

    stats = [AGG_STAT_PRIMARY] + ([AGG_STAT_SENSITIVITY] if AGG_STAT_SENSITIVITY else [])
    todo = [(k, e, stats) for k, e in scenes.items()
            if not (RESUME and scene_csv_path(k).exists())]
    print(f"processing {len(todo)} scenes ({len(scenes) - len(todo)} already done)")

    failures = []
    first_tb_shown = False

    def _record(i: int, result) -> None:
        nonlocal first_tb_shown
        key, ok, short, tb = result
        print(f"  [{i}/{len(todo)}] {key}: {short}")
        if not ok:
            failures.append((key, short, tb))
            if not first_tb_shown:
                print("\n--- full traceback for the first failure "
                      "-------------------------------")
                print(tb.rstrip())
                print("-" * 70 + "\n")
                first_tb_shown = True

    if MAX_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(run_scene, a): a[0] for a in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                _record(i, fut.result())
    else:
        for i, a in enumerate(todo, 1):
            _record(i, run_scene(a))

    # Written before anything can return early, so a total failure is still logged.
    if failures:
        log = OUT_DIR / "failures.log"
        with open(log, "w") as f:
            for key, short, tb in failures:
                f.write(f"=== {key} ===\n{short}\n{tb}\n")
        print(f"\n{len(failures)} of {len(todo)} scenes failed -> {log}")
        counts = pd.Series([s for _, s, _ in failures]).value_counts()
        print("distinct errors:")
        for msg, n in counts.items():
            print(f"  {n:4d}x  {msg}")

    parts = sorted((OUT_DIR / "scenes").glob("*.csv"))
    if not parts:
        print("no results to combine")
        return
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    lead = [c for c in ["scene", "site", "date", "flight", "resolution_m", "product",
                        "agg_stat", "stratum_kind", "stratum"] if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    out_csv = OUT_DIR / "resolution_metrics_long.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df)} rows from {len(parts)} scenes)")

    all_scenes = df["scene"].nunique()
    keep = usable(df)
    n_keep = keep["scene"].nunique()
    if n_keep < all_scenes:
        print(f"\n{all_scenes - n_keep} of {all_scenes} scenes dropped from the "
              f"summary: valid_frac < {MIN_VALID_FRAC} (kept in the CSV)")

    sel = keep[(keep.stratum_kind == "all") & (keep.agg_stat == AGG_STAT_PRIMARY)
               & keep["product"].isin(["DSM", "CHM_matched_dtm", "DTM_vs_finest",
                                       "AGGREGATION_FLOOR"])]
    summary = (sel.groupby(["product", "resolution_m"])
                  .agg(scenes=("scene", "nunique"), nmad=("nmad", "median"),
                       le95=("le95", "median"), me=("me", "median"),
                       rmse=("rmse", "median"),
                       blunder=("blunder_frac", "median"))
                  .reset_index())
    print("\nAcross-scene medians (primary statistic). NMAD is the headline: "
          "RMSE stays outlier-sensitive.")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if MAKE_PLOTS:
        make_plots(df, OUT_DIR)


if __name__ == "__main__":
    main()