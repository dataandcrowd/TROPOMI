"""
Sentinel-5P TROPOMI L3 (fortnight) tropospheric NO2 — New Zealand-wide analysis

Input  : s5p-l3grd-no2-tropospheric-001-fortnight-*.nc in this folder (global 0.022° grid)
Output : figures (PNG), statistics (CSV) and NetCDF subsets clipped to New Zealand (cache) in output_nz/

Run    : python analyze_nz_no2.py [--data-dir DIR] [--pattern GLOB] [--out-dir DIR]
         e.g. python analyze_nz_no2.py --data-dir "D:/Tropomi NO2" --pattern "*season*.nc"
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import shapely
from shapely.geometry import box
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, PowerNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# ---------------------------------------------------------------- Settings
parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent,
                    help="folder holding the global L3 NetCDF files (default: this script's folder)")
parser.add_argument("--pattern", default="s5p-l3grd-no2-tropospheric-*fortnight*.nc",
                    help="glob for the input files (default: fortnight files)")
parser.add_argument("--out-dir", type=Path, default=None,
                    help="output folder (default: <data-dir>/output_nz)")
args = parser.parse_args()
DATA_DIR = args.data_dir
OUT_DIR = args.out_dir or DATA_DIR / "output_nz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# New Zealand mainland (excluding the Chatham Islands)
LAT_MIN, LAT_MAX = -47.6, -34.0
LON_MIN, LON_MAX = 166.0, 179.0

URBAN_RADIUS_KM = 10      # urban area radius
BG_RADIUS_KM = (40, 80)   # regional background ring
SITE_HALF_WIN = 1         # site value = mean of 3x3 pixels (~7 km)
# Quality filter: exclude pixels whose valid-observation weight over the fortnight is below this
# (e.g. snow-covered Southern Alps in August: weight≈1, outliers of 140–160 µmol/m² NO2)
MIN_WEIGHT = 2.0

# name: (lat, lon, type)
SITES = {
    "Auckland": (-36.848, 174.763, "city"),
    "Hamilton": (-37.787, 175.279, "city"),
    "Tauranga": (-37.687, 176.165, "city"),
    "Rotorua": (-38.137, 176.251, "city"),
    "Whangarei": (-35.725, 174.324, "city"),
    "New Plymouth": (-39.057, 174.075, "city"),
    "Napier-Hastings": (-39.560, 176.880, "city"),
    "Palmerston North": (-40.352, 175.608, "city"),
    "Wellington": (-41.287, 174.776, "city"),
    "Nelson": (-41.271, 173.284, "city"),
    "Christchurch": (-43.532, 172.637, "city"),
    "Dunedin": (-45.874, 170.504, "city"),
    "Queenstown": (-45.031, 168.663, "city"),
    "Invercargill": (-46.413, 168.354, "city"),
    "Huntly power stn": (-37.545, 175.155, "industrial"),
    "Glenbrook steel mill": (-37.205, 174.735, "industrial"),
    "Kawerau pulp mill": (-38.083, 176.700, "industrial"),
    "Tiwai Point smelter": (-46.590, 168.380, "industrial"),
}
LABELS_LEFT = {"Hamilton"}
# Cities for the zoomed maps (centre, half-width in deg)
ZOOM_CITIES = ["Auckland", "Hamilton", "Wellington", "Christchurch"]
ZOOM_HALF = 0.45

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
    """Clip the New Zealand domain from a global file and cache it (output_nz/subset_*.nc)."""
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


def land_mask(lat, lon):
    """Build a grid land mask from the Natural Earth 10m land polygons."""
    bbox = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    geoms = [g.intersection(bbox) for g in shpreader.Reader(
        shpreader.natural_earth("10m", "physical", "land")).geometries()
        if g.intersects(bbox)]
    land = shapely.union_all(geoms)
    lon2, lat2 = np.meshgrid(lon, lat)
    return shapely.contains_xy(land, lon2, lat2)


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
    low = ds.weight < MIN_WEIGHT
    print(f"  Quality filter: excluded {int(low.sum())} pixels with weight < {MIN_WEIGHT} "
          f"({float(low.mean()) * 100:.1f}%)")
    ds["no2"] = ds.no2.where(~low)
    ds["no2_unc"] = ds.no2_unc.where(~low)
    t0 = pd.Timestamp(ds.datetime_start.values.item())
    t1 = pd.Timestamp(ds.datetime_stop.values.item())
    periods.append({"ds": ds, "start": t0, "stop": t1})
# Include the year in labels when the periods span more than one year
fmt = "%m/%d" if len({t.year for p in periods for t in (p["start"], p["stop"])}) == 1 else "%Y/%m/%d"
for p in periods:
    p["label"] = f"{p['start']:{fmt}}–{p['stop']:{fmt}}"
order = [p["label"] for p in periods]
NP = len(periods)
# Composite name for titles, from the mean period length (~14 days vs ~3 months)
COMPOSITE = "seasonal" if np.mean([(p["stop"] - p["start"]).days for p in periods]) > 40 else "fortnightly"

lat = periods[0]["ds"].latitude.values
lon = periods[0]["ds"].longitude.values
LON2, LAT2 = np.meshgrid(lon, lat)
LAND = land_mask(lat, lon)
print(f"Grid {lat.size} x {lon.size}, land pixels {LAND.mean() * 100:.1f}%")


def site_value(arr, la, lo):
    i = int(np.abs(lat - la).argmin())
    j = int(np.abs(lon - lo).argmin())
    w = SITE_HALF_WIN
    return float(np.nanmean(arr[i - w:i + w + 1, j - w:j + w + 1]))


# ---------------------------------------------------------------- Statistics
period_rows, site_rows = [], []
site_dist = {name: haversine_km(la, lo, LAT2, LON2) for name, (la, lo, _) in SITES.items()}
for p in periods:
    no2 = p["ds"].no2.values
    unc = p["ds"].no2_unc.values
    north = LAND & (LAT2 > -40.6) & (LON2 > 172.6)     # North Island (approx.)
    south = LAND & ~north
    period_rows.append({
        "period": p["label"], "start": p["start"].date(), "stop": p["stop"].date(),
        "land_mean": np.nanmean(no2[LAND]), "ocean_mean": np.nanmean(no2[~LAND]),
        "north_island_mean": np.nanmean(no2[north]),
        "south_island_mean": np.nanmean(no2[south]),
        "land_p99": np.nanpercentile(no2[LAND], 99), "max": np.nanmax(no2),
        "max_lat": float(LAT2.flat[np.nanargmax(no2)]),
        "max_lon": float(LON2.flat[np.nanargmax(no2)]),
        "mean_obs_weight_land": np.nanmean(p["ds"].weight.values[LAND]),
    })
    for name, (la, lo, kind) in SITES.items():
        d = site_dist[name]
        urban = np.nanmean(no2[d <= URBAN_RADIUS_KM])
        bg = np.nanmean(no2[(d >= BG_RADIUS_KM[0]) & (d <= BG_RADIUS_KM[1])])
        site_rows.append({
            "period": p["label"], "site": name, "type": kind,
            "no2_umol_m2": site_value(no2, la, lo),
            "no2_unc_umol_m2": site_value(unc, la, lo),
            "urban_mean": urban, "background_mean": bg,
            "enhancement": urban - bg,
        })

stats = pd.DataFrame(period_rows)
sites_df = pd.DataFrame(site_rows)
sites_df["no2_1e15_molec_cm2"] = sites_df["no2_umol_m2"] * UMOL_M2_TO_1E15_MOLEC_CM2
stats.to_csv(OUT_DIR / "period_stats.csv", index=False, encoding="utf-8-sig")
sites_df.to_csv(OUT_DIR / "site_values.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200, "display.max_columns", 30,
              "display.float_format", "{:.2f}".format)
print("\n=== Statistics by period (µmol/m²) ===")
print(stats.set_index("period").T)
print("\n=== NO2 by site (µmol/m², 3x3 pixel mean) ===")
print(sites_df.pivot(index="site", columns="period", values="no2_umol_m2")
      .reindex(SITES.keys()))
print(f"\n=== Urban enhancement = mean within {URBAN_RADIUS_KM} km - "
      f"{BG_RADIUS_KM[0]}–{BG_RADIUS_KM[1]} km background ===")
print(sites_df.pivot(index="site", columns="period", values="enhancement")
      .reindex(SITES.keys()))


# ---------------------------------------------------------------- Shared figure helpers
def map_axes(fig, pos, extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)):
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return ax


def decorate(ax, labels=False, grid=True, fs=7):
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.5, edgecolor="#383835")
    for name, (la, lo, kind) in SITES.items():
        mk = "o" if kind == "city" else "^"
        ax.plot(lo, la, mk, ms=4, mfc="white", mec=INK, mew=0.8,
                transform=ccrs.PlateCarree(), zorder=5)
        if labels and kind == "city":
            left = name in LABELS_LEFT  # put labels that overlap a neighbouring city on the left
            ax.text(lo - 0.15 if left else lo + 0.15, la, name, fontsize=fs, color=INK,
                    va="center", ha="right" if left else "left",
                    transform=ccrs.PlateCarree(), zorder=6, clip_on=True,
                    bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.4))
    if grid:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=GRID, alpha=0.8)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": MUTED}


# ---------------------------------------------------------------- 1) National maps (per period)
# Background values are low and only cities stand out, so use a square-root scale (PowerNorm 0.5)
vmax = np.nanpercentile(np.concatenate([p["ds"].no2.values[LAND] for p in periods]), 99.9)
norm = PowerNorm(0.5, vmin=0, vmax=vmax, clip=True)
fig = plt.figure(figsize=(6 * NP, 6.4))
for k, p in enumerate(periods):
    ax = map_axes(fig, (1, len(periods), k + 1))
    im = ax.pcolormesh(lon, lat, p["ds"].no2.values, cmap=SEQ, norm=norm,
                       shading="nearest", transform=ccrs.PlateCarree(), rasterized=True)
    decorate(ax, labels=(k == 0))
    ax.set_title(f"{p['start']:%Y-%m-%d} ~ {p['stop']:%Y-%m-%d}", fontsize=12)
cb = fig.colorbar(im, ax=fig.axes, orientation="horizontal", fraction=0.04, pad=0.06, shrink=0.45)
cb.set_label("Tropospheric NO$_2$ column (µmol/m², square-root scale)  ○ city  △ industrial site")
fig.suptitle(f"New Zealand tropospheric NO$_2$ — TROPOMI {COMPOSITE} mean", fontsize=15, y=1.0)
fig.savefig(OUT_DIR / "01_nz_no2_maps.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 2) Change map (last - first period)
first, last = periods[0], periods[-1]
diff = last["ds"].no2.values - first["ds"].no2.values
lim = np.nanpercentile(np.abs(diff[LAND]), 99.5)
fig = plt.figure(figsize=(8.5, 9))
ax = map_axes(fig, (1, 1, 1))
im = ax.pcolormesh(lon, lat, diff, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim),
                   shading="nearest", transform=ccrs.PlateCarree(), rasterized=True)
decorate(ax, labels=True)
cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.06)
cb.set_label("ΔNO$_2$ (µmol/m²)")
ax.set_title(f"NO$_2$ change: ({last['label']}) - ({first['label']})\nred = increase, blue = decrease",
             fontsize=12)
fig.savefig(OUT_DIR / "02_nz_difference.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3) Enhancement by city/site (horizontal bars)
enh = (sites_df.pivot(index="site", columns="period", values="enhancement")[order])
enh = enh.loc[enh.mean(axis=1).sort_values().index]   # sort by mean enhancement
fig, ax = plt.subplots(figsize=(10, 9))
n = len(order)
bh = 0.8 / n
y = np.arange(len(enh))
for k, lab in enumerate(order):
    ax.barh(y + (k - (n - 1) / 2) * bh, enh[lab], height=bh * 0.9,
            color=PERIOD_COLORS[k], label=lab, zorder=3)
names = [f"{s}  △" if SITES[s][2] == "industrial" else s for s in enh.index]
ax.set_yticks(y, names, fontsize=9)
ax.axvline(0, color="#c3c2b7", lw=1)
ax.grid(axis="y", visible=False)
ax.set_xlabel(f"NO$_2$ enhancement (µmol/m²) = mean within {URBAN_RADIUS_KM} km - "
              f"{BG_RADIUS_KM[0]}–{BG_RADIUS_KM[1]} km background")
ax.legend(title="Period", frameon=False, loc="lower right")
ax.set_title("NO$_2$ enhancement by city and industrial site (△ = industrial)", loc="left", fontsize=13)
fig.savefig(OUT_DIR / "03_site_enhancement.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 4) Zoomed maps of major cities (city x period)
fig = plt.figure(figsize=(13 / 3 * NP, 3.3 * len(ZOOM_CITIES)))
zvmax = 0
panels = []
for r, city in enumerate(ZOOM_CITIES):
    la, lo, _ = SITES[city]
    ext = (lo - ZOOM_HALF / np.cos(np.radians(la)), lo + ZOOM_HALF / np.cos(np.radians(la)),
           la - ZOOM_HALF, la + ZOOM_HALF)
    for k, p in enumerate(periods):
        ax = map_axes(fig, (len(ZOOM_CITIES), len(periods), r * len(periods) + k + 1), ext)
        im = ax.pcolormesh(lon, lat, p["ds"].no2.values, cmap=SEQ, vmin=0, vmax=35,
                           shading="nearest", transform=ccrs.PlateCarree(), rasterized=True)
        decorate(ax, labels=False, grid=False)
        ax.set_title(f"{city}  |  {p['label']}", fontsize=10)
fig.subplots_adjust(right=0.9, hspace=0.15, wspace=0.05)
fig.colorbar(im, cax=fig.add_axes([0.92, 0.3, 0.012, 0.4]), label="NO$_2$ (µmol/m²)")
fig.suptitle("Major cities zoomed in (~100 km × 100 km, common colour range 0–35)", fontsize=14, y=0.93)
fig.savefig(OUT_DIR / "04_city_zoom.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 5) Land-mean profile by latitude
lat_bins = np.arange(-47.5, -33.99, 0.5)
lat_mid = (lat_bins[:-1] + lat_bins[1:]) / 2
fig, ax = plt.subplots(figsize=(6.5, 8))
for k, p in enumerate(periods):
    no2 = p["ds"].no2.values
    prof = []
    for a, b in zip(lat_bins[:-1], lat_bins[1:]):
        m = LAND & (LAT2 >= a) & (LAT2 < b)
        prof.append(np.nanmean(no2[m]) if m.sum() > 50 else np.nan)
    ax.plot(prof, lat_mid, "-o", color=PERIOD_COLORS[k], lw=2, ms=4, label=p["label"])
for name in ["Auckland", "Hamilton", "Wellington", "Christchurch", "Dunedin", "Invercargill"]:
    ax.axhline(SITES[name][0], color=GRID, lw=0.8, zorder=0)
    ax.text(ax.get_xlim()[1], SITES[name][0], f" {name}", va="center", fontsize=8, color=MUTED)
ax.axhspan(-41.8, -40.4, color="#f6f6f3", zorder=0)
ax.text(ax.get_xlim()[0], -41.1, " Cook Strait", va="center", fontsize=8, color=MUTED)
ax.set_xlabel("Land-mean NO$_2$ (µmol/m²)")
ax.set_ylabel("Latitude (°)")
ax.legend(frameon=False, title="Period", loc="lower right")
ax.set_title("Land-mean NO$_2$ in 0.5° latitude bands", loc="left", fontsize=12)
fig.savefig(OUT_DIR / "05_latitude_profile.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 6) Data quality: observation weight
fig = plt.figure(figsize=(6 * NP, 6.4))
wmax = max(np.nanmax(p["ds"].weight.values) for p in periods)
for k, p in enumerate(periods):
    ax = map_axes(fig, (1, len(periods), k + 1))
    im = ax.pcolormesh(lon, lat, p["ds"].weight.values, cmap=SEQ, vmin=0, vmax=wmax,
                       shading="nearest", transform=ccrs.PlateCarree(), rasterized=True)
    decorate(ax)
    ax.set_title(f"{p['label']}  observation weight", fontsize=11)
cb = fig.colorbar(im, ax=fig.axes, orientation="horizontal", fraction=0.04, pad=0.06, shrink=0.45)
cb.set_label("weight (higher = more valid observations) — low over the often-cloudy west coast and mountains")
fig.suptitle(f"Data quality — pixels with weight < {MIN_WEIGHT:g} are excluded from the analysis "
             "(blank in other maps)", fontsize=14, y=1.0)
fig.savefig(OUT_DIR / "06_nz_data_quality.png", bbox_inches="tight")
plt.close(fig)

print(f"\nDone. Results saved to: {OUT_DIR}")
