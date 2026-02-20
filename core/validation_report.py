"""
validation_report.py
─────────────────────
Generates a self-contained HTML validation report in the same style as report.py.

Replaces the WeasyPrint-based generate_validation_report() in core/utils.py.

Usage
-----
    from validation_report import generate_validation_report

    generate_validation_report(
        gdf,
        reference_col="val_value",
        prediction_col="dem_value",
        output_path="/path/to/report.html",
        config=config,          # optional — adds run metadata to header
    )

The GeoDataFrame must have:
  - a geometry column in any CRS (will be reprojected to EPSG:4326 for maps)
  - reference_col, prediction_col, and 'raster_name' columns
"""

import os
import base64
import datetime
from io import BytesIO
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Shared style constants (mirrors report.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

_FONTS = (
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500'
    '&display=swap" rel="stylesheet">'
)

_CSS = """<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:    #ffffff; --bg-row: #fafafa; --line: #ebebeb; --line-md: #d4d4d4;
  --text:  #0a0a0a; --mid:    #525252; --soft: #a3a3a3; --xs:      #d4d4d4;
  --blue:  #2563eb; --green:  #16a34a; --red:  #dc2626; --amber: #d97706;
  --mono: 'JetBrains Mono', monospace;
  --sans: 'Inter', sans-serif;
}
html { scroll-behavior: smooth; }
body {
  font-family: var(--sans); font-size: 13px; line-height: 1.5;
  background: var(--bg); color: var(--text);
  -webkit-font-smoothing: antialiased;
  font-feature-settings: "cv01","cv02","cv03","cv04","ss01";
}
.top {
  height: 44px; border-bottom: 1px solid var(--line);
  display: flex; align-items: center; padding: 0 24px; gap: 20px;
  position: sticky; top: 0; background: var(--bg); z-index: 10;
}
.top-brand { font-size: 13px; font-weight: 600; color: var(--text); letter-spacing: -0.2px; }
.top-sep   { color: var(--line-md); font-weight: 300; }
.top-run   { font-size: 13px; color: var(--mid); }
.top-right { margin-left: auto; display: flex; align-items: center; gap: 16px; }
.top-meta  { font-size: 12px; color: var(--soft); }
.top-meta strong { color: var(--mid); font-weight: 500; }
.wrap { display: grid; grid-template-columns: 200px 1fr; max-width: 1080px; margin: 0 auto; }
.nav { border-right: 1px solid var(--line); padding: 28px 0; position: sticky; top: 44px; height: calc(100vh - 44px); overflow-y: auto; }
.nav-link { display: flex; align-items: center; gap: 10px; padding: 6px 16px; font-size: 12.5px; color: var(--soft); text-decoration: none; transition: color 0.1s; position: relative; }
.nav-link:hover { color: var(--text); }
.nav-link.active { color: var(--text); font-weight: 500; }
.nav-link.active::before { content: ''; position: absolute; left: 0; width: 2px; height: 20px; background: var(--text); border-radius: 0 2px 2px 0; }
.n-idx { font-family: var(--mono); font-size: 10px; color: var(--xs); min-width: 16px; }
.nav-link.active .n-idx { color: var(--soft); }
.content { padding: 40px 40px 80px; min-width: 0; }
.page-top { margin-bottom: 40px; }
.pt-label { font-size: 11px; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: var(--soft); margin-bottom: 8px; }
.pt-title { font-size: 22px; font-weight: 600; letter-spacing: -0.4px; color: var(--text); margin-bottom: 6px; line-height: 1.2; }
.pt-desc  { font-size: 13px; color: var(--mid); line-height: 1.6; max-width: 480px; margin-bottom: 20px; }
.tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
.t       { font-size: 11.5px; font-weight: 500; padding: 2px 8px; border-radius: 3px; border: 1px solid var(--line-md); color: var(--mid); background: var(--bg-row); }
.t-green { color: var(--green); border-color: #bbf7d0; background: #f0fdf4; }
.t-red   { color: var(--red);   border-color: #fecaca; background: #fef2f2; }
.t-blue  { color: var(--blue);  border-color: #bfdbfe; background: #eff6ff; }
.notice  { font-size: 12.5px; color: var(--mid); background: var(--bg-row); border-left: 2px solid var(--line-md); padding: 10px 14px; margin-bottom: 36px; line-height: 1.6; }
.section { margin-bottom: 40px; }
.sec-head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 0; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
.si { font-family: var(--mono); font-size: 10.5px; color: var(--xs); }
.st { font-size: 13px; font-weight: 600; color: var(--text); letter-spacing: -0.1px; }
.sc { margin-left: auto; font-size: 11px; color: var(--soft); font-family: var(--mono); }
.tbl { width: 100%; border-collapse: collapse; }
.tbl-group td { font-size: 10.5px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: var(--soft); padding: 12px 0 4px; }
.param-row td { padding: 9px 0; border-bottom: 1px solid var(--line); vertical-align: middle; transition: background 0.1s, padding 0.1s; }
.param-row:last-child td { border-bottom: none; }
.param-row:hover td { background: var(--bg-row); }
.param-row:hover td:first-child { padding-left: 6px; }
.param-row:hover td:last-child  { padding-right: 6px; }
.k  { font-family: var(--mono); font-size: 12px; color: var(--mid); width: 55%; }
.kd { font-size: 11px; color: var(--soft); font-family: var(--sans); margin-top: 1px; }
.v-cell { text-align: right; vertical-align: middle; }
.v      { font-family: var(--mono); font-size: 12.5px; font-weight: 500; color: var(--text); }
.v-unit { font-size: 10.5px; font-weight: 400; color: var(--soft); margin-left: 2px; }
.v-good { color: var(--green) !important; }
.v-warn { color: var(--amber) !important; }
.v-bad  { color: var(--red)   !important; }
.plot-wrap  { margin-top: 12px; }
.plot-wrap img { width: 100%; height: auto; display: block; border: 1px solid var(--line); border-radius: 4px; }
.plot-label { font-size: 10.5px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--soft); margin-bottom: 5px; }
.plot-grid  { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; margin-bottom: 32px; }
.card       { background: var(--bg); padding: 14px 16px; }
.card-label { font-size: 10.5px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--soft); margin-bottom: 4px; }
.card-val   { font-family: var(--mono); font-size: 18px; font-weight: 600; color: var(--text); letter-spacing: -0.5px; line-height: 1.1; }
.card-unit  { font-size: 10.5px; color: var(--soft); font-weight: 400; margin-left: 2px; }
.card-val.good { color: var(--green); }
.card-val.warn { color: var(--amber); }
.card-val.bad  { color: var(--red);   }
.raster-block  { margin-bottom: 24px; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.raster-header { display: flex; align-items: center; gap: 10px; padding: 10px 16px; background: var(--bg-row); border-bottom: 1px solid var(--line); cursor: pointer; user-select: none; }
.raster-header:hover { background: #f3f4f6; }
.raster-title { font-size: 12.5px; font-weight: 600; color: var(--text); flex: 1; font-family: var(--mono); }
.raster-rmse  { font-family: var(--mono); font-size: 11.5px; color: var(--soft); }
.chevron      { color: var(--xs); transition: transform 0.2s; font-size: 10px; }
.raster-block.open .chevron { transform: rotate(90deg); }
.raster-body  { display: none; padding: 16px; }
.raster-block.open .raster-body { display: block; }
.raster-cols  { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
.footer { padding-top: 20px; border-top: 1px solid var(--line); display: flex; justify-content: space-between; align-items: center; margin-top: 40px; }
.fl { font-size: 12px; color: var(--soft); }
.fr { font-family: var(--mono); font-size: 11px; color: var(--xs); text-align: right; line-height: 1.8; }
@media (max-width: 640px) {
  .wrap { grid-template-columns: 1fr; } .nav { display: none; }
  .content { padding: 24px 16px 60px; }
  .raster-cols { grid-template-columns: 1fr; }
  .plot-grid   { grid-template-columns: 1fr; }
}
</style>"""

_JS = """<script>
  const secs  = document.querySelectorAll('.section[id]');
  const links = document.querySelectorAll('.nav-link');
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        const a = document.querySelector('.nav-link[href="#' + e.target.id + '"]');
        if (a) a.classList.add('active');
      }
    });
  }, { rootMargin: '-10% 0px -80% 0px' });
  secs.forEach(s => io.observe(s));

  document.querySelectorAll('.raster-header').forEach(h => {
    h.addEventListener('click', () => h.closest('.raster-block').classList.toggle('open'));
  });
</script>"""


# ─────────────────────────────────────────────────────────────────────────────
# Plot style
# ─────────────────────────────────────────────────────────────────────────────

def _apply_plot_style():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
        "axes.edgecolor":     "#d4d4d4",
        "axes.linewidth":     0.8,
        "axes.grid":          True,
        "grid.color":         "#ebebeb",
        "grid.linewidth":     0.6,
        "xtick.color":        "#a3a3a3",
        "ytick.color":        "#a3a3a3",
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "axes.labelcolor":    "#525252",
        "axes.labelsize":     10,
        "axes.titlesize":     11,
        "axes.titleweight":   "600",
        "axes.titlecolor":    "#0a0a0a",
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
    })


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(ref, pred) -> dict:
    res     = pred - ref
    abs_res = np.abs(res)
    slope, intercept, r, *_ = stats.linregress(ref, pred)
    return {
        "RMSE":           float(np.sqrt(np.mean(res ** 2))),
        "MAE":            float(np.mean(abs_res)),
        "NMAD":           float(1.4826 * stats.median_abs_deviation(res, scale=1.0)),
        "Mean Residual":  float(np.mean(res)),
        "Std Error":      float(np.std(res)),
        "Median Error":   float(np.median(res)),
        "LE90":           float(np.percentile(abs_res, 90)),
        "LE95":           float(np.percentile(abs_res, 95)),
        "Max Over":       float(np.max(res)),
        "Max Under":      float(np.min(res)),
        "R²":             float(r ** 2),
        "n":              int(len(ref)),
    }


_METRIC_DESC = {
    "RMSE":          "Root Mean Square Error",
    "MAE":           "Mean Absolute Error",
    "NMAD":          "Normalized Median Absolute Deviation",
    "Mean Residual": "Mean of (pred − ref); bias indicator",
    "Std Error":     "Standard deviation of residuals",
    "Median Error":  "Median of (pred − ref)",
    "LE90":          "Linear Error at 90th percentile",
    "LE95":          "Linear Error at 95th percentile",
    "Max Over":      "Largest positive residual",
    "Max Under":     "Largest negative residual",
    "R²":            "Coefficient of determination",
    "n":             "Number of validation points",
}


# ─────────────────────────────────────────────────────────────────────────────
# Statistical plots
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_plot(ref, pred, title="") -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.scatter(ref, pred, s=14, alpha=0.55, color="#2563eb",
               edgecolors="#1d4ed8", linewidths=0.3, zorder=3)
    lo, hi  = min(ref.min(), pred.min()), max(ref.max(), pred.max())
    pad     = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="#dc2626", linewidth=1.0, linestyle="--", label="1 : 1", zorder=2)
    slope, intercept, r, *_ = stats.linregress(ref, pred)
    x_fit = np.linspace(lo - pad, hi + pad, 200)
    ax.plot(x_fit, slope * x_fit + intercept,
            color="#16a34a", linewidth=1.0, label=f"fit  r²={r**2:.3f}", zorder=2)
    ax.set_xlabel("Reference")
    ax.set_ylabel("Modelled")
    if title:
        ax.set_title(title, pad=8)
    ax.legend(fontsize=8, frameon=False)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    return _fig_to_b64(fig)


def _residual_plot(ref, pred) -> str:
    _apply_plot_style()
    res = pred - ref
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.scatter(ref, res, s=14, alpha=0.55, color="#525252",
               edgecolors="#404040", linewidths=0.3, zorder=3)
    ax.axhline(0, color="#dc2626", linewidth=1.0, linestyle="--", zorder=2)
    ax.axhline(np.mean(res), color="#2563eb", linewidth=0.8, linestyle=":",
               label=f"mean {np.mean(res):+.3f}", zorder=2)
    ax.set_xlabel("Reference")
    ax.set_ylabel("Residual  (pred − ref)")
    ax.set_title("Residuals", pad=8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial map plots
# ─────────────────────────────────────────────────────────────────────────────

_RASTER_COLORS = [
    "#2563eb", "#16a34a", "#d97706", "#9333ea",
    "#0891b2", "#e11d48", "#65a30d", "#7c3aed",
]


def _map_points_by_raster(x, y, raster_names) -> str:
    """Points coloured by source raster — shows spatial coverage."""
    _apply_plot_style()
    unique    = list(dict.fromkeys(raster_names))
    color_map = {r: _RASTER_COLORS[i % len(_RASTER_COLORS)] for i, r in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, color="#e5e7eb", linewidth=0.5, zorder=0)

    for rname in unique:
        mask = raster_names == rname
        ax.scatter(x[mask], y[mask], s=12, alpha=0.75,
                   color=color_map[rname], edgecolors="none",
                   label=rname, zorder=3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Validation Point Locations", pad=8)
    handles, labels = ax.get_legend_handles_labels()
    labels = [l if len(l) <= 30 else l[:27] + "…" for l in labels]
    ax.legend(handles, labels, fontsize=7.5, frameon=True,
              framealpha=0.95, edgecolor="#e5e7eb", loc="best", markerscale=1.4)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    return _fig_to_b64(fig)


def _map_signed_error(x, y, residuals) -> str:
    """
    Points coloured by signed residual (pred − ref).
    Diverging RdBu_r colormap, symmetric around zero, clipped at ±3×NMAD.
    Blue = model underestimates, red = model overestimates.
    """
    _apply_plot_style()
    nmad = 1.4826 * stats.median_abs_deviation(residuals, scale=1.0)
    vmax = max(3 * nmad, 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, color="#e5e7eb", linewidth=0.5, zorder=0)

    sc = ax.scatter(x, y, c=residuals, cmap="RdBu_r", norm=norm,
                    s=16, alpha=0.85, edgecolors="none", zorder=3)

    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="4%", pad=0.08)
    cb      = fig.colorbar(sc, cax=cax)
    cb.set_label("pred − ref  (m)", fontsize=9, color="#525252")
    cb.ax.tick_params(labelsize=8, colors="#a3a3a3")
    cb.outline.set_edgecolor("#d4d4d4")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Signed Error  (pred − ref)", pad=8)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.annotate(f"Clipped at ±3×NMAD ({vmax:.3f} m)",
                xy=(0.01, 0.01), xycoords="axes fraction",
                fontsize=7.5, color="#a3a3a3", ha="left", va="bottom")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _map_abs_error(x, y, residuals) -> str:
    """
    Points coloured by absolute error |pred − ref|.
    Sequential YlOrRd colormap, clipped at 95th percentile.
    Immediately highlights where the largest errors cluster spatially.
    """
    _apply_plot_style()
    abs_res = np.abs(residuals)
    vmax    = np.percentile(abs_res, 95)
    norm    = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, color="#e5e7eb", linewidth=0.5, zorder=0)

    sc = ax.scatter(x, y, c=abs_res, cmap="YlOrRd", norm=norm,
                    s=16, alpha=0.85, edgecolors="none", zorder=3)

    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="4%", pad=0.08)
    cb      = fig.colorbar(sc, cax=cax)
    cb.set_label("|pred − ref|  (m)", fontsize=9, color="#525252")
    cb.ax.tick_params(labelsize=8, colors="#a3a3a3")
    cb.outline.set_edgecolor("#d4d4d4")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Absolute Error  |pred − ref|", pad=8)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.annotate(f"Clipped at 95th percentile ({vmax:.3f} m)",
                xy=(0.01, 0.01), xycoords="axes fraction",
                fontsize=7.5, color="#a3a3a3", ha="left", va="bottom")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quality_class(metric_name: str, value: float) -> str:
    thresholds = {
        "RMSE": (0.1, 0.3), "MAE":  (0.08, 0.25),
        "NMAD": (0.08, 0.25), "LE90": (0.20, 0.50), "LE95": (0.25, 0.60),
    }
    if metric_name not in thresholds:
        return ""
    lo, hi = thresholds[metric_name]
    abs_v  = abs(value)
    if abs_v <= lo: return "v-good"
    if abs_v <= hi: return "v-warn"
    return "v-bad"


def _fmt(v) -> str:
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:.4f}" if abs(v) < 1000 else f"{v:,.1f}"


def _metric_rows(metrics: dict) -> str:
    rows = ""
    for key, val in metrics.items():
        desc      = _METRIC_DESC.get(key, "")
        desc_html = f'<div class="kd">{desc}</div>' if desc else ""
        qcls      = _quality_class(key, val)
        rows += f"""
        <tr class="param-row">
          <td class="k">{key}{desc_html}</td>
          <td class="v-cell"><span class="v {qcls}">{_fmt(val)}</span></td>
        </tr>"""
    return rows


def _summary_cards(metrics: dict) -> str:
    cards = ""
    for k in ["RMSE", "MAE", "R²", "n"]:
        v = metrics.get(k)
        if v is None:
            continue
        qcls = ("good" if _quality_class(k, v) == "v-good" else
                "warn" if _quality_class(k, v) == "v-warn" else
                "bad"  if _quality_class(k, v) == "v-bad"  else "")
        unit = "" if k in ("R²", "n") else " m"
        cards += f"""
    <div class="card">
      <div class="card-label">{k}</div>
      <div class="card-val {qcls}">{_fmt(v)}<span class="card-unit">{unit}</span></div>
    </div>"""
    return f'<div class="cards">{cards}\n</div>'


def _img(b64: str, alt: str = "", label: str = "") -> str:
    label_html = f'<div class="plot-label">{label}</div>' if label else ""
    return f'<div class="plot-wrap">{label_html}<img src="data:image/png;base64,{b64}" alt="{alt}"></div>'


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate extraction (works with/without geopandas)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_xy(gdf):
    """
    Return (x, y) numpy arrays in WGS-84.
    Tries geopandas geometry first, then falls back to named columns.
    Returns (None, None) when no spatial info is available.
    """
    try:
        geom = gdf.geometry
        try:
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
                geom = gdf.geometry
        except Exception:
            pass
        x = np.array(geom.x)
        y = np.array(geom.y)
        if len(x) > 0:
            return x, y
    except Exception:
        pass

    for xcol, ycol in [("x", "y"), ("lon", "lat"),
                        ("longitude", "latitude"), ("X", "Y")]:
        if xcol in gdf.columns and ycol in gdf.columns:
            return np.array(gdf[xcol]), np.array(gdf[ycol])

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_validation_report(
    gdf,
    reference_col: str,
    prediction_col: str,
    output_path: str,
    config=None,
) -> str:
    """
    Generate a self-contained HTML validation report with spatial maps.

    Parameters
    ----------
    gdf            : GeoDataFrame with reference, prediction, raster_name
                     columns and (ideally) a geometry column.
    reference_col  : Column name for reference elevation values.
    prediction_col : Column name for modelled elevation values.
    output_path    : Destination path — extension is forced to .html.
    config         : Optional Configuration object for run metadata.

    Returns
    -------
    str : Absolute path of the written HTML file.
    """
    base, _ = os.path.splitext(output_path)
    output_path = base + ".html"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    now     = datetime.datetime.now()
    iso     = now.isoformat(timespec="seconds")
    now_str = now.strftime("%Y-%m-%d %H:%M")

    run_name   = getattr(config, "run_name",          "Validation Report") if config else "Validation Report"
    val_target = getattr(config, "validation_target", "")                   if config else ""

    # ── valid rows ────────────────────────────────────────────────────────
    df = gdf[[reference_col, prediction_col, "raster_name"]].dropna()
    if df.empty:
        raise ValueError("No valid (non-NaN) validation rows in GeoDataFrame.")

    ref       = df[reference_col].values
    pred      = df[prediction_col].values
    residuals = pred - ref
    rnames    = df["raster_name"].values

    # ── spatial coordinates ───────────────────────────────────────────────
    x_all, y_all = _extract_xy(gdf)
    has_spatial  = x_all is not None

    if has_spatial:
        valid_idx = df.index
        try:
            x_v = x_all[valid_idx]
            y_v = y_all[valid_idx]
        except (IndexError, TypeError):
            x_v = x_all[:len(df)]
            y_v = y_all[:len(df)]
        has_spatial = len(x_v) == len(residuals)

    # ── metrics ───────────────────────────────────────────────────────────
    global_metrics = _compute_metrics(ref, pred)
    raster_names   = list(dict.fromkeys(rnames))

    per_raster = {}
    for rname in raster_names:
        mask = rnames == rname
        entry = {
            "metrics":      _compute_metrics(ref[mask], pred[mask]),
            "scatter_b64":  _scatter_plot(ref[mask], pred[mask], rname),
            "residual_b64": _residual_plot(ref[mask], pred[mask]),
        }
        if has_spatial:
            entry["map_error_b64"] = _map_signed_error(
                x_v[mask], y_v[mask], residuals[mask]
            )
        per_raster[rname] = entry

    # ── global plots ──────────────────────────────────────────────────────
    global_scatter_b64  = _scatter_plot(ref, pred, "Global")
    global_residual_b64 = _residual_plot(ref, pred)



    # ── nav ───────────────────────────────────────────────────────────────
    nav_items = [("s01","Summary"),("s02","Global Metrics"),
                 ("s03","Per-Raster Results")]

    nav_html = "\n".join(
        f'<a class="nav-link{"  active" if i==0 else ""}" href="#{sid}">'
        f'<span class="n-idx">{i+1:02d}</span>{label}</a>'
        for i, (sid, label) in enumerate(nav_items)
    )

    # ── header tags ───────────────────────────────────────────────────────
    rmse     = global_metrics["RMSE"]
    r2       = global_metrics["R²"]
    rmse_cls = "t-green" if rmse < 0.1 else ("t" if rmse < 0.3 else "t-red")
    r2_cls   = "t-green" if r2   > 0.9 else ("t" if r2   > 0.7 else "t-red")
    tags_html = (
        (f'<span class="t t-blue">{val_target}</span>' if val_target else "")
        + f'<span class="t {rmse_cls}">RMSE {rmse:.3f} m</span>'
        + f'<span class="t {r2_cls}">R² {r2:.3f}</span>'
        + f'<span class="t">{global_metrics["n"]:,} points</span>'
        + f'<span class="t">{len(raster_names)} raster{"s" if len(raster_names)!=1 else ""}</span>'
    )

    # ── s01 summary ───────────────────────────────────────────────────────
    s01 = f"""
  <div class="section" id="s01">
    <div class="sec-head">
      <span class="si">01</span><span class="st">Summary</span>
      <span class="sc">{global_metrics["n"]:,} pts · {len(raster_names)} rasters</span>
    </div>
    {_summary_cards(global_metrics)}
    <div class="notice">
      Colour coding: <strong style="color:#16a34a">green</strong> = good,
      <strong style="color:#d97706">amber</strong> = acceptable,
      <strong style="color:#dc2626">red</strong> = review required.
      Thresholds: RMSE ≤ 0.10 m (good) / ≤ 0.30 m (acceptable).
    </div>
  </div>"""

    s02 = ""
    gi, ri = "02", "03"

    # ── global metrics ────────────────────────────────────────────────────
    s_global = f"""
  <div class="section" id="s{gi}">
    <div class="sec-head">
      <span class="si">{gi}</span><span class="st">Global Metrics</span>
      <span class="sc">{global_metrics["n"]:,} points</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start;margin-top:12px;">
      <table class="tbl">{_metric_rows(global_metrics)}</table>
      <div>
        {_img(global_scatter_b64,  "Global scatter",   "Scatter Plot")}
        {_img(global_residual_b64, "Global residuals", "Residuals")}
      </div>
    </div>
  </div>"""

    # ── per-raster ────────────────────────────────────────────────────────
    raster_blocks = ""
    for i, (rname, data) in enumerate(per_raster.items()):
        rmse_r     = data["metrics"]["RMSE"]
        rmse_cls_r = "v-good" if rmse_r < 0.1 else ("v-bad" if rmse_r > 0.3 else "v-warn")

        # Per-raster spatial error map sits below the stats+scatter columns
        raster_map_html = ""
        if "map_error_b64" in data:
            raster_map_html = f"""
        <div style="margin-top:16px; border-top:1px solid var(--line); padding-top:16px;">
          {_img(data["map_error_b64"], f"Error map {rname}", "Spatial Error  (pred − ref)")}
        </div>"""

        raster_blocks += f"""
    <div class="raster-block{"  open" if i==0 else ""}">
      <div class="raster-header">
        <span class="raster-title">{rname}</span>
        <span class="raster-rmse">RMSE <span class="{rmse_cls_r}">{rmse_r:.4f} m</span></span>
        <span class="chevron">▶</span>
      </div>
      <div class="raster-body">
        <div class="raster-cols">
          <table class="tbl">{_metric_rows(data["metrics"])}</table>
          <div>
            {_img(data["scatter_b64"],  f"Scatter {rname}",   "Scatter Plot")}
            {_img(data["residual_b64"], f"Residuals {rname}", "Residuals")}
          </div>
        </div>
        {raster_map_html}
      </div>
    </div>"""

    s_raster = f"""
  <div class="section" id="s{ri}">
    <div class="sec-head">
      <span class="si">{ri}</span><span class="st">Per-Raster Results</span>
      <span class="sc">{len(raster_names)} rasters</span>
    </div>
    <div style="margin-top:12px;">{raster_blocks}
    </div>
  </div>"""

    # ── assemble ──────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Validation Report · {run_name}</title>
{_FONTS}
{_CSS}
</head>
<body>

<div class="top">
  <span class="top-brand">LidarProcessing</span>
  <span class="top-sep">/</span>
  <span class="top-run">{run_name}</span>
  <div class="top-right">
    <span class="top-meta"><strong>{now_str}</strong></span>
    <span class="top-meta">Validation</span>
  </div>
</div>

<div class="wrap">
  <nav class="nav">{nav_html}
  </nav>
  <div class="content">
    <div class="page-top">
      <div class="pt-label">Validation Report</div>
      <div class="pt-title">{run_name}</div>
      <p class="pt-desc">
        Accuracy assessment of modelled vs. reference elevation data.
        Metrics computed on {global_metrics["n"]:,} points
        across {len(raster_names)} raster file{"s" if len(raster_names)!=1 else ""}.
      </p>
      <div class="tags">{tags_html}</div>
    </div>

    {s01}
    {s02}
    {s_global}
    {s_raster}

    <div class="footer">
      <span class="fl">{run_name} · Validation Report</span>
      <span class="fr">Generated {iso}</span>
    </div>
  </div>
</div>

{_JS}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"[validation_report] Written → {output_path}")
    return output_path