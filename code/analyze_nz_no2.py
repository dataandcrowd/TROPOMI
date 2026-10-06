"""
Sentinel-5P TROPOMI L3 (fortnight) 대류권 NO2 — 뉴질랜드 전역 분석

입력 : 이 폴더의 s5p-l3grd-no2-tropospheric-001-fortnight-*.nc (전지구 0.022° 격자)
출력 : output_nz/ 폴더에 그림(PNG), 통계(CSV), 뉴질랜드 영역만 잘라낸 NetCDF(캐시)

실행 : python analyze_nz_no2.py
"""
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

# ---------------------------------------------------------------- 설정
DATA_DIR = Path(__file__).parent
OUT_DIR = DATA_DIR / "output_nz"
OUT_DIR.mkdir(exist_ok=True)

# 뉴질랜드 본토 (채텀 제도 제외)
LAT_MIN, LAT_MAX = -47.6, -34.0
LON_MIN, LON_MAX = 166.0, 179.0

URBAN_RADIUS_KM = 10      # 도시 영역 반경
BG_RADIUS_KM = (40, 80)   # 지역 배경 고리
SITE_HALF_WIN = 1         # 지점 값 = 3x3 픽셀(약 7 km) 평균
# 품질 필터: 2주 동안 유효 관측 가중치가 이보다 작은 픽셀은 제외
# (예: 8월 서던 알프스 적설 지역에서 weight≈1, NO2 140~160 µmol/m² 의 이상값)
MIN_WEIGHT = 2.0

# 이름: (lat, lon, 유형)
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
# 확대 지도에 쓸 도시 (중심, 반폭 deg)
ZOOM_CITIES = ["Auckland", "Hamilton", "Wellington", "Christchurch"]
ZOOM_HALF = 0.45

UMOL_M2_TO_1E15_MOLEC_CM2 = 6.02214076e23 * 1e-6 / 1e4 / 1e15  # = 0.0602

# 차트 색 (dataviz 레퍼런스 팔레트)
PERIOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # 기간 1~3 고정 순서
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                     "#f2a5a4", "#e34948", "#9c1f1f"])
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": ["Malgun Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 110, "savefig.dpi": 160,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- 데이터 로드
def load_period(nc_path: Path) -> xr.Dataset:
    """전지구 파일에서 뉴질랜드 영역만 잘라 캐시(output_nz/subset_*.nc)로 저장."""
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
    """Natural Earth 10m 육지 폴리곤으로 격자 육지 마스크 생성."""
    bbox = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    geoms = [g.intersection(bbox) for g in shpreader.Reader(
        shpreader.natural_earth("10m", "physical", "land")).geometries()
        if g.intersects(bbox)]
    land = shapely.union_all(geoms)
    lon2, lat2 = np.meshgrid(lon, lat)
    return shapely.contains_xy(land, lon2, lat2)


files = sorted(DATA_DIR.glob("s5p-l3grd-no2-tropospheric-*fortnight*.nc"))
if not files:
    raise SystemExit("NetCDF 파일을 찾을 수 없습니다.")

periods = []
for f in files:
    print(f"읽는 중: {f.name}")
    ds = load_period(f)
    low = ds.weight < MIN_WEIGHT
    print(f"  품질 필터: weight < {MIN_WEIGHT} 픽셀 {int(low.sum())}개 "
          f"({float(low.mean()) * 100:.1f}%) 제외")
    ds["no2"] = ds.no2.where(~low)
    ds["no2_unc"] = ds.no2_unc.where(~low)
    t0 = pd.Timestamp(ds.datetime_start.values.item())
    t1 = pd.Timestamp(ds.datetime_stop.values.item())
    periods.append({"ds": ds, "start": t0, "stop": t1, "label": f"{t0:%m/%d}–{t1:%m/%d}"})
order = [p["label"] for p in periods]

lat = periods[0]["ds"].latitude.values
lon = periods[0]["ds"].longitude.values
LON2, LAT2 = np.meshgrid(lon, lat)
LAND = land_mask(lat, lon)
print(f"격자 {lat.size} x {lon.size}, 육지 픽셀 {LAND.mean() * 100:.1f}%")


def site_value(arr, la, lo):
    i = int(np.abs(lat - la).argmin())
    j = int(np.abs(lon - lo).argmin())
    w = SITE_HALF_WIN
    return float(np.nanmean(arr[i - w:i + w + 1, j - w:j + w + 1]))


# ---------------------------------------------------------------- 통계
period_rows, site_rows = [], []
site_dist = {name: haversine_km(la, lo, LAT2, LON2) for name, (la, lo, _) in SITES.items()}
for p in periods:
    no2 = p["ds"].no2.values
    unc = p["ds"].no2_unc.values
    north = LAND & (LAT2 > -40.6) & (LON2 > 172.6)     # 북섬 (대략)
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
print("\n=== 기간별 통계 (µmol/m²) ===")
print(stats.set_index("period").T)
print("\n=== 지점별 NO2 (µmol/m², 3x3 픽셀 평균) ===")
print(sites_df.pivot(index="site", columns="period", values="no2_umol_m2")
      .reindex(SITES.keys()))
print(f"\n=== 도시 증강량 = 반경 {URBAN_RADIUS_KM} km 평균 - {BG_RADIUS_KM[0]}–{BG_RADIUS_KM[1]} km 배경 ===")
print(sites_df.pivot(index="site", columns="period", values="enhancement")
      .reindex(SITES.keys()))


# ---------------------------------------------------------------- 그림 공통
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
            left = name in LABELS_LEFT  # 옆 도시와 겹치는 라벨은 왼쪽에
            ax.text(lo - 0.15 if left else lo + 0.15, la, name, fontsize=fs, color=INK,
                    va="center", ha="right" if left else "left",
                    transform=ccrs.PlateCarree(), zorder=6, clip_on=True,
                    bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.4))
    if grid:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=GRID, alpha=0.8)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": MUTED}


# ---------------------------------------------------------------- 1) 전국 지도 (기간별)
# 배경 값이 낮고 도시만 튀므로 제곱근 스케일(PowerNorm 0.5)로 표시
vmax = np.nanpercentile(np.concatenate([p["ds"].no2.values[LAND] for p in periods]), 99.9)
norm = PowerNorm(0.5, vmin=0, vmax=vmax, clip=True)
fig = plt.figure(figsize=(18, 6.4))
for k, p in enumerate(periods):
    ax = map_axes(fig, (1, len(periods), k + 1))
    im = ax.pcolormesh(lon, lat, p["ds"].no2.values, cmap=SEQ, norm=norm,
                       shading="nearest", transform=ccrs.PlateCarree(), rasterized=True)
    decorate(ax, labels=(k == 0))
    ax.set_title(f"{p['start']:%Y-%m-%d} ~ {p['stop']:%Y-%m-%d}", fontsize=12)
cb = fig.colorbar(im, ax=fig.axes, orientation="horizontal", fraction=0.04, pad=0.06, shrink=0.45)
cb.set_label("대류권 NO$_2$ 칼럼 (µmol/m², 제곱근 스케일)  ○ 도시  △ 산업시설")
fig.suptitle("뉴질랜드 대류권 NO$_2$ — TROPOMI 2주 평균", fontsize=15, y=1.0)
fig.savefig(OUT_DIR / "01_nz_no2_maps.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 2) 변화 지도 (마지막 - 첫 기간)
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
ax.set_title(f"NO$_2$ 변화: ({last['label']}) - ({first['label']})\n빨강 = 증가, 파랑 = 감소",
             fontsize=12)
fig.savefig(OUT_DIR / "02_nz_difference.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3) 도시·시설별 증강량 (가로 막대)
enh = (sites_df.pivot(index="site", columns="period", values="enhancement")[order])
enh = enh.loc[enh.mean(axis=1).sort_values().index]   # 평균 증강량 순 정렬
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
ax.set_xlabel(f"NO$_2$ 증강량 (µmol/m²) = 반경 {URBAN_RADIUS_KM} km 평균 - "
              f"{BG_RADIUS_KM[0]}–{BG_RADIUS_KM[1]} km 배경")
ax.legend(title="기간", frameon=False, loc="lower right")
ax.set_title("도시·산업시설별 NO$_2$ 증강량 (△ = 산업시설)", loc="left", fontsize=13)
fig.savefig(OUT_DIR / "03_site_enhancement.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 4) 주요 도시 확대 지도 (도시 x 기간)
fig = plt.figure(figsize=(13, 3.3 * len(ZOOM_CITIES)))
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
fig.suptitle("주요 도시 확대 (약 100 km × 100 km, 공통 색 범위 0–35)", fontsize=14, y=0.93)
fig.savefig(OUT_DIR / "04_city_zoom.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 5) 위도별 육지 평균 프로파일
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
ax.text(ax.get_xlim()[0], -41.1, " 쿡 해협", va="center", fontsize=8, color=MUTED)
ax.set_xlabel("육지 평균 NO$_2$ (µmol/m²)")
ax.set_ylabel("위도 (°)")
ax.legend(frameon=False, title="기간", loc="lower right")
ax.set_title("위도 0.5° 구간별 육지 평균 NO$_2$", loc="left", fontsize=12)
fig.savefig(OUT_DIR / "05_latitude_profile.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 6) 데이터 품질: 관측 가중치
fig = plt.figure(figsize=(18, 6.4))
wmax = max(np.nanmax(p["ds"].weight.values) for p in periods)
for k, p in enumerate(periods):
    ax = map_axes(fig, (1, len(periods), k + 1))
    im = ax.pcolormesh(lon, lat, p["ds"].weight.values, cmap=SEQ, vmin=0, vmax=wmax,
                       shading="nearest", transform=ccrs.PlateCarree(), rasterized=True)
    decorate(ax)
    ax.set_title(f"{p['label']}  관측 가중치", fontsize=11)
cb = fig.colorbar(im, ax=fig.axes, orientation="horizontal", fraction=0.04, pad=0.06, shrink=0.45)
cb.set_label("weight (유효 관측이 많을수록 큼) — 구름이 잦은 서해안·산지는 값이 낮음")
fig.suptitle(f"데이터 품질 — weight < {MIN_WEIGHT:g} 픽셀은 분석에서 제외(다른 지도에서 공백)",
             fontsize=14, y=1.0)
fig.savefig(OUT_DIR / "06_nz_data_quality.png", bbox_inches="tight")
plt.close(fig)

print(f"\n완료. 결과 저장 위치: {OUT_DIR}")
