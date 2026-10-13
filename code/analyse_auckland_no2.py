"""
Sentinel-5P TROPOMI L3 (fortnight) tropospheric NO2 — Auckland (New Zealand) regional analysis

Input  : s5p-l3grd-no2-tropospheric-001-fortnight-*.nc in this folder (global 0.022° grid)
Output : figures (PNG), statistics (CSV) and NetCDF subsets clipped to Auckland (cache) in output/

Run    : python analyse_auckland_no2.py [--data-dir DIR] [--pattern GLOB] [--out-dir DIR]
         e.g. python analyse_auckland_no2.py --data-dir "D:/Tropomi NO2" --pattern "*season*.nc"
"""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------- Settings
parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent,
                    help="folder holding the global L3 NetCDF files (default: this script's folder)")
parser.add_argument("--pattern", default="s5p-l3grd-no2-tropospheric-*fortnight*.nc",
                    help="glob for the input files (default: fortnight files)")
parser.add_argument("--out-dir", type=Path, default=None,
                    help="output folder (default: <data-dir>/output)")
args = parser.parse_args()
DATA_DIR = args.data_dir
OUT_DIR = args.out_dir or DATA_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis domain (greater Auckland + surrounding sea/rural background)
LAT_MIN, LAT_MAX = -37.75, -36.05
LON_MIN, LON_MAX = 173.95, 175.65

CBD = (-36.8485, 174.7633)  # (lat, lon) Auckland city centre
URBAN_RADIUS_KM = 15        # urban area radius
BG_RADIUS_KM = (45, 90)     # background ring

SITES = {  # name: (lat, lon)
    "CBD / Port": (-36.8485, 174.7633),
    "Penrose (industrial)": (-36.910, 174.815),
    "Manukau": (-36.993, 174.880),
    "Airport": (-37.008, 174.785),
    "Albany (North Shore)": (-36.728, 174.700),
    "Glenbrook steel mill": (-37.205, 174.735),
    "Waiheke Is. (island)": (-36.800, 175.100),
    "Hunua Ranges (rural)": (-37.070, 175.130),
}
# Map label position offsets (lon, lat) — only for overlapping sites
LABEL_OFFSET = {"Airport": (-0.36, -0.05), "Manukau": (0.04, -0.06),
                "Penrose (industrial)": (0.05, 0.0), "CBD / Port": (-0.33, 0.03),
                "Albany (North Shore)": (-0.30, 0.04)}
SITE_HALF_WIN = 1  # site value = mean of (2*1+1)^2 = 3x3 pixels (~7 km)

UMOL_M2_TO_1E15_MOLEC_CM2 = 6.02214076e23 * 1e-6 / 1e4 / 1e15  # = 0.0602

# Chart colours (dataviz reference palette)
PERIOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed order for periods 1–5
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                     "#f2a5a4", "#e34948", "#9c1f1f"])
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 110, "savefig.dpi": 160,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- Data loading
def load_period(nc_path: Path) -> xr.Dataset:
    """Clip the Auckland domain from a global file and cache it (output/subset_*.nc)."""
    cache = OUT_DIR / f"subset_{nc_path.stem}.nc"
    if cache.exists():
        return xr.open_dataset(cache)
    with xr.open_dataset(nc_path) as ds:
        sub = ds.sel(latitude=slice(LAT_MIN, LAT_MAX),
                     longitude=slice(LON_MIN, LON_MAX)).isel(time=0).load()
    sub = sub.rename({"tropospheric_NO2_column_number_density": "no2",
                      "tropospheric_NO2_column_number_density_uncertainty": "no2_unc"})
    sub.attrs["source_file"] = nc_path.name
    sub.to_netcdf(cache)
    return sub


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# Raw files, plus periods available only as a cached subset (raw file no longer on disk)
names = ({f.name for f in DATA_DIR.glob(args.pattern)}
         | {c.name.removeprefix("subset_") for c in OUT_DIR.glob("subset_" + args.pattern)})
files = [DATA_DIR / n for n in sorted(names)]
if not files:
    raise SystemExit("No NetCDF files found.")
if len(files) > len(PERIOD_COLORS):
    raise SystemExit(f"At most {len(PERIOD_COLORS)} periods are supported, found {len(files)}.")

periods = []
for f in files:
    print(f"Reading: {f.name}")
    ds = load_period(f)
    t0 = pd.Timestamp(ds.datetime_start.values.item())
    t1 = pd.Timestamp(ds.datetime_stop.values.item())
    periods.append({"file": f.name, "ds": ds, "start": t0, "stop": t1})
# Include the year in labels when the periods span more than one year
fmt = "%m/%d" if len({t.year for p in periods for t in (p["start"], p["stop"])}) == 1 else "%Y/%m/%d"
for p in periods:
    p["label"] = f"{p['start']:{fmt}}–{p['stop']:{fmt}}"
NP = len(periods)
# Composite name for titles, from the mean period length (~14 days vs ~3 months)
COMPOSITE = "seasonal" if np.mean([(p["stop"] - p["start"]).days for p in periods]) > 40 else "fortnightly"

lat = periods[0]["ds"].latitude.values
lon = periods[0]["ds"].longitude.values
LON2, LAT2 = np.meshgrid(lon, lat)
DIST = haversine_km(CBD[0], CBD[1], LAT2, LON2)
URBAN = DIST <= URBAN_RADIUS_KM
BG = (DIST >= BG_RADIUS_KM[0]) & (DIST <= BG_RADIUS_KM[1])


def site_value(arr, la, lo):
    i = int(np.abs(lat - la).argmin())
    j = int(np.abs(lon - lo).argmin())
    w = SITE_HALF_WIN
    return float(np.nanmean(arr[i - w:i + w + 1, j - w:j + w + 1]))


# ---------------------------------------------------------------- Statistics
rows, site_rows = [], []
for p in periods:
    no2 = p["ds"].no2.values
    unc = p["ds"].no2_unc.values
    wt = p["ds"].weight.values
    urban_mean = np.nanmean(no2[URBAN])
    bg_mean = np.nanmean(no2[BG])
    rows.append({
        "period": p["label"], "start": p["start"].date(), "stop": p["stop"].date(),
        "domain_mean": np.nanmean(no2), "domain_median": np.nanmedian(no2),
        "domain_p90": np.nanpercentile(no2, 90), "domain_max": np.nanmax(no2),
        "urban_mean(<=15km)": urban_mean, "background_mean(45-90km)": bg_mean,
        "urban_enhancement": urban_mean - bg_mean,
        "urban/background_ratio": urban_mean / bg_mean,
        "CBD_value": site_value(no2, *CBD),
        "mean_rel_uncertainty_urban_%": 100 * np.nanmean(unc[URBAN]) / urban_mean,
        "mean_obs_weight": np.nanmean(wt),
        "valid_pixel_%": 100 * np.isfinite(no2).mean(),
    })
    for name, (la, lo) in SITES.items():
        site_rows.append({"period": p["label"], "site": name,
                          "no2_umol_m2": site_value(no2, la, lo),
                          "no2_unc_umol_m2": site_value(unc, la, lo)})

stats = pd.DataFrame(rows)
sites_df = pd.DataFrame(site_rows)
sites_df["no2_1e15_molec_cm2"] = sites_df["no2_umol_m2"] * UMOL_M2_TO_1E15_MOLEC_CM2
stats.to_csv(OUT_DIR / "period_stats.csv", index=False, encoding="utf-8-sig")
sites_df.to_csv(OUT_DIR / "site_values.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200, "display.max_columns", 30,
              "display.float_format", "{:.2f}".format)
print("\n=== Statistics by period (units: µmol/m², 1 µmol/m² = 0.0602×10^15 molec/cm²) ===")
print(stats.set_index("period").T)
print("\n=== NO2 by site (µmol/m², 3x3 pixel mean) ===")
print(sites_df.pivot(index="site", columns="period", values="no2_umol_m2")
      .reindex(SITES.keys()))


# ---------------------------------------------------------------- Shared figure helpers
def map_axes(fig, pos):
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    return ax


def decorate(ax, labels=False):
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor="#383835")
    for name, (la, lo) in SITES.items():
        ax.plot(lo, la, "o", ms=4, mfc="white", mec=INK, mew=0.8,
                transform=ccrs.PlateCarree(), zorder=5)
        if labels:
            dx, dy = LABEL_OFFSET.get(name, (0.03, 0.02))
            ax.text(lo + dx, la + dy, name, fontsize=6.5, color=INK,
                    transform=ccrs.PlateCarree(), zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=GRID, alpha=0.8)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": MUTED}


# ---------------------------------------------------------------- 1) Maps per period
vmax = np.nanpercentile(np.concatenate([p["ds"].no2.values.ravel() for p in periods]), 99.5)
fig = plt.figure(figsize=(5 * NP, 5.6))
for k, p in enumerate(periods):
    ax = map_axes(fig, (1, len(periods), k + 1))
    im = ax.pcolormesh(lon, lat, p["ds"].no2.values, cmap=SEQ, vmin=0, vmax=vmax,
                       shading="nearest", transform=ccrs.PlateCarree())
    decorate(ax, labels=(k == 0))
    ax.set_title(f"{p['start']:%Y-%m-%d} ~ {p['stop']:%Y-%m-%d}", fontsize=11)
cb = fig.colorbar(im, ax=fig.axes, orientation="horizontal", fraction=0.05, pad=0.08, shrink=0.5)
cb.set_label("Tropospheric NO$_2$ column (µmol/m²)")
fig.suptitle(f"Auckland tropospheric NO$_2$ — TROPOMI {COMPOSITE} mean", fontsize=14, y=0.98)
fig.savefig(OUT_DIR / "01_no2_maps.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 2) Difference map (last - first period)
first, last = periods[0], periods[-1]
diff = last["ds"].no2.values - first["ds"].no2.values
lim = np.nanpercentile(np.abs(diff), 99)
fig = plt.figure(figsize=(7.5, 6.8))
ax = map_axes(fig, (1, 1, 1))
im = ax.pcolormesh(lon, lat, diff, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim),
                   shading="nearest", transform=ccrs.PlateCarree())
decorate(ax, labels=True)
cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.07)
cb.set_label("ΔNO$_2$ (µmol/m²)")
ax.set_title(f"NO$_2$ change: ({last['label']}) - ({first['label']})\n"
             "red = increase, blue = decrease", fontsize=11)
fig.savefig(OUT_DIR / "02_no2_difference.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3) Site comparison (grouped bar)
piv = sites_df.pivot(index="site", columns="period", values="no2_umol_m2").reindex(SITES.keys())
piv_unc = sites_df.pivot(index="site", columns="period", values="no2_unc_umol_m2").reindex(SITES.keys())
order = [p["label"] for p in periods]
fig, ax = plt.subplots(figsize=(11, 5))
n = len(order)
bw = 0.8 / n
x = np.arange(len(piv))
for k, lab in enumerate(order):
    ax.bar(x + (k - (n - 1) / 2) * bw, piv[lab], width=bw * 0.92, color=PERIOD_COLORS[k],
           yerr=piv_unc[lab], error_kw=dict(ecolor=MUTED, elinewidth=0.8, capsize=0),
           label=lab, zorder=3)
ax.set_xticks(x, piv.index, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("NO$_2$ (µmol/m²)")
ax.axhline(0, color="#c3c2b7", lw=1)
ax.grid(axis="x", visible=False)
ax.legend(title="Period", frameon=False, ncols=n, loc="upper right")
ax.set_title("Tropospheric NO$_2$ by site (3×3 pixel mean, error bars = retrieval uncertainty)",
             loc="left", fontsize=12)
fig.savefig(OUT_DIR / "03_site_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 4) Profile by distance from the CBD
bins = np.arange(0, 92, 4)
mid = (bins[:-1] + bins[1:]) / 2
fig, ax = plt.subplots(figsize=(9, 5))
for k, p in enumerate(periods):
    no2 = p["ds"].no2.values
    prof = [np.nanmean(no2[(DIST >= a) & (DIST < b)]) for a, b in zip(bins[:-1], bins[1:])]
    ax.plot(mid, prof, "-o", color=PERIOD_COLORS[k], lw=2, ms=4, label=p["label"])
    if NP <= 3:  # end-of-line labels collide with more series; the legend carries identity
        ax.annotate(p["label"], (mid[-1], prof[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
ax.axvspan(0, URBAN_RADIUS_KM, color="#f0efec", zorder=0)
ax.axvspan(*BG_RADIUS_KM, color="#f6f6f3", zorder=0)
y0 = ax.get_ylim()[0]
ax.text(URBAN_RADIUS_KM / 2, y0, "Urban\n", ha="center", va="bottom", color=MUTED, fontsize=8)
ax.text(np.mean(BG_RADIUS_KM), y0, "Background\n", ha="center", va="bottom", color=MUTED, fontsize=8)
ax.set_xlabel("Distance from Auckland CBD (km)")
ax.set_ylabel("Mean NO$_2$ (µmol/m²)")
ax.set_xlim(0, 100)
ax.legend(frameon=False, title="Period")
ax.set_title("NO$_2$ decreases with distance from the city centre (radial mean profile)",
             loc="left", fontsize=12)
fig.savefig(OUT_DIR / "04_radial_profile.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 5) Distributions (urban vs background)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, mask, title in [(axes[0], URBAN, f"Urban (CBD ≤ {URBAN_RADIUS_KM} km)"),
                        (axes[1], BG, f"Background ({BG_RADIUS_KM[0]}–{BG_RADIUS_KM[1]} km)")]:
    data = [p["ds"].no2.values[mask] for p in periods]
    data = [d[np.isfinite(d)] for d in data]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5))
    for patch, c in zip(bp["boxes"], PERIOD_COLORS):
        patch.set(facecolor=c, alpha=0.85, edgecolor=c)
    long_labels = len(order[0]) > 12  # labels with years overlap unless rotated
    ax.set_xticks(range(1, len(periods) + 1), order, fontsize=9,
                  rotation=30 if long_labels else 0, ha="right" if long_labels else "center")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("NO$_2$ (µmol/m²)")
fig.suptitle("Pixel distributions (box = 25–75%, whiskers = 1.5 IQR)", fontsize=12, x=0.07, ha="left")
fig.savefig(OUT_DIR / "05_distribution.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 6) Data quality: observation weight & relative uncertainty
fig = plt.figure(figsize=(5 * NP, 9.5))
wmax = max(np.nanmax(p["ds"].weight.values) for p in periods)
for k, p in enumerate(periods):
    ax = map_axes(fig, (2, len(periods), k + 1))
    im1 = ax.pcolormesh(lon, lat, p["ds"].weight.values, cmap=SEQ, vmin=0, vmax=wmax,
                        shading="nearest", transform=ccrs.PlateCarree())
    decorate(ax)
    ax.set_title(f"{p['label']}  observation weight (valid obs. count)", fontsize=10)
    ax = map_axes(fig, (2, len(periods), len(periods) + k + 1))
    rel = 100 * p["ds"].no2_unc.values / np.clip(p["ds"].no2.values, 1, None)
    im2 = ax.pcolormesh(lon, lat, rel, cmap=SEQ, vmin=0, vmax=100,
                        shading="nearest", transform=ccrs.PlateCarree())
    decorate(ax)
    ax.set_title(f"{p['label']}  relative uncertainty (%)", fontsize=10)
fig.subplots_adjust(right=0.9, hspace=0.18)
fig.colorbar(im1, cax=fig.add_axes([0.92, 0.55, 0.012, 0.3]), label="weight")
fig.colorbar(im2, cax=fig.add_axes([0.92, 0.13, 0.012, 0.3]), label="uncertainty / NO$_2$ (%)")
fig.suptitle("Data quality — interpret with caution where valid observations are few "
             "and relative uncertainty is high", fontsize=13)
fig.savefig(OUT_DIR / "06_data_quality.png", bbox_inches="tight")
plt.close(fig)

print(f"\nDone. Results saved to: {OUT_DIR}")
