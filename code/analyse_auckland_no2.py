"""
Sentinel-5P TROPOMI L3 (fortnight) 대류권 NO2 — 오클랜드(뉴질랜드) 지역 분석

입력 : 이 폴더의 s5p-l3grd-no2-tropospheric-001-fortnight-*.nc (전지구 0.022° 격자)
출력 : output/ 폴더에 그림(PNG), 통계(CSV), 오클랜드 영역만 잘라낸 NetCDF(캐시)

실행 : python analyze_auckland_no2.py
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------- 설정
DATA_DIR = Path(__file__).parent
OUT_DIR = DATA_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# 분석 영역 (오클랜드 광역 + 주변 해역/농촌 배경)
LAT_MIN, LAT_MAX = -37.75, -36.05
LON_MIN, LON_MAX = 173.95, 175.65

CBD = (-36.8485, 174.7633)  # (lat, lon) 오클랜드 도심
URBAN_RADIUS_KM = 15        # 도심 영역 반경
BG_RADIUS_KM = (45, 90)     # 배경(background) 고리 영역

SITES = {  # 이름: (lat, lon)
    "CBD / Port": (-36.8485, 174.7633),
    "Penrose (industrial)": (-36.910, 174.815),
    "Manukau": (-36.993, 174.880),
    "Airport": (-37.008, 174.785),
    "Albany (North Shore)": (-36.728, 174.700),
    "Glenbrook steel mill": (-37.205, 174.735),
    "Waiheke Is. (island)": (-36.800, 175.100),
    "Hunua Ranges (rural)": (-37.070, 175.130),
}
# 지도 라벨 위치 보정 (경도, 위도 오프셋) — 겹치는 지점만
LABEL_OFFSET = {"Airport": (-0.36, -0.05), "Manukau": (0.04, -0.06),
                "Penrose (industrial)": (0.05, 0.0), "CBD / Port": (-0.33, 0.03),
                "Albany (North Shore)": (-0.30, 0.04)}
SITE_HALF_WIN = 1  # 사이트 값 = (2*1+1)^2 = 3x3 픽셀(약 7 km) 평균

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
    """전지구 파일에서 오클랜드 영역만 잘라 캐시(output/subset_*.nc)로 저장."""
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


files = sorted(DATA_DIR.glob("s5p-l3grd-no2-tropospheric-*fortnight*.nc"))
if not files:
    raise SystemExit("NetCDF 파일을 찾을 수 없습니다.")

periods = []
for f in files:
    print(f"읽는 중: {f.name}")
    ds = load_period(f)
    t0 = pd.Timestamp(ds.datetime_start.values.item())
    t1 = pd.Timestamp(ds.datetime_stop.values.item())
    periods.append({"file": f.name, "ds": ds, "start": t0, "stop": t1,
                    "label": f"{t0:%m/%d}–{t1:%m/%d}"})

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


# ---------------------------------------------------------------- 통계
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
print("\n=== 기간별 통계 (단위: µmol/m², 1 µmol/m² = 0.0602×10^15 molec/cm²) ===")
print(stats.set_index("period").T)
print("\n=== 지점별 NO2 (µmol/m², 3x3 픽셀 평균) ===")
print(sites_df.pivot(index="site", columns="period", values="no2_umol_m2")
      .reindex(SITES.keys()))


# ---------------------------------------------------------------- 그림 공통
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


# ---------------------------------------------------------------- 1) 기간별 지도
vmax = np.nanpercentile(np.concatenate([p["ds"].no2.values.ravel() for p in periods]), 99.5)
fig = plt.figure(figsize=(15, 5.6))
for k, p in enumerate(periods):
    ax = map_axes(fig, (1, len(periods), k + 1))
    im = ax.pcolormesh(lon, lat, p["ds"].no2.values, cmap=SEQ, vmin=0, vmax=vmax,
                       shading="nearest", transform=ccrs.PlateCarree())
    decorate(ax, labels=(k == 0))
    ax.set_title(f"{p['start']:%Y-%m-%d} ~ {p['stop']:%Y-%m-%d}", fontsize=11)
cb = fig.colorbar(im, ax=fig.axes, orientation="horizontal", fraction=0.05, pad=0.08, shrink=0.5)
cb.set_label("대류권 NO$_2$ 칼럼 (µmol/m²)")
fig.suptitle("오클랜드 대류권 NO$_2$ — TROPOMI 2주 평균", fontsize=14, y=0.98)
fig.savefig(OUT_DIR / "01_no2_maps.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 2) 차이 지도 (마지막 - 첫 기간)
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
ax.set_title(f"NO$_2$ 변화: ({last['label']}) - ({first['label']})\n"
             "빨강 = 증가, 파랑 = 감소", fontsize=11)
fig.savefig(OUT_DIR / "02_no2_difference.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3) 지점별 비교 (grouped bar)
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
ax.legend(title="기간", frameon=False, ncols=n, loc="upper right")
ax.set_title("지점별 대류권 NO$_2$ (3×3 픽셀 평균, 오차막대 = 산출 불확도)", loc="left", fontsize=12)
fig.savefig(OUT_DIR / "03_site_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 4) 도심 거리별 프로파일
bins = np.arange(0, 92, 4)
mid = (bins[:-1] + bins[1:]) / 2
fig, ax = plt.subplots(figsize=(9, 5))
for k, p in enumerate(periods):
    no2 = p["ds"].no2.values
    prof = [np.nanmean(no2[(DIST >= a) & (DIST < b)]) for a, b in zip(bins[:-1], bins[1:])]
    ax.plot(mid, prof, "-o", color=PERIOD_COLORS[k], lw=2, ms=4, label=p["label"])
    ax.annotate(p["label"], (mid[-1], prof[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK2)
ax.axvspan(0, URBAN_RADIUS_KM, color="#f0efec", zorder=0)
ax.axvspan(*BG_RADIUS_KM, color="#f6f6f3", zorder=0)
y0 = ax.get_ylim()[0]
ax.text(URBAN_RADIUS_KM / 2, y0, "도심\n", ha="center", va="bottom", color=MUTED, fontsize=8)
ax.text(np.mean(BG_RADIUS_KM), y0, "배경 영역\n", ha="center", va="bottom", color=MUTED, fontsize=8)
ax.set_xlabel("오클랜드 CBD로부터 거리 (km)")
ax.set_ylabel("평균 NO$_2$ (µmol/m²)")
ax.set_xlim(0, 100)
ax.legend(frameon=False, title="기간")
ax.set_title("도심에서 멀어질수록 NO$_2$ 감소 (방사형 평균 프로파일)", loc="left", fontsize=12)
fig.savefig(OUT_DIR / "04_radial_profile.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 5) 분포 (도심 vs 배경)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, mask, title in [(axes[0], URBAN, f"도심 (CBD ≤ {URBAN_RADIUS_KM} km)"),
                        (axes[1], BG, f"배경 ({BG_RADIUS_KM[0]}–{BG_RADIUS_KM[1]} km)")]:
    data = [p["ds"].no2.values[mask] for p in periods]
    data = [d[np.isfinite(d)] for d in data]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5))
    for patch, c in zip(bp["boxes"], PERIOD_COLORS):
        patch.set(facecolor=c, alpha=0.85, edgecolor=c)
    ax.set_xticks(range(1, len(periods) + 1), order, fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("NO$_2$ (µmol/m²)")
fig.suptitle("픽셀 분포 비교 (상자 = 25–75%, 수염 = 1.5 IQR)", fontsize=12, x=0.07, ha="left")
fig.savefig(OUT_DIR / "05_distribution.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 6) 데이터 품질: 관측 가중치 & 상대 불확도
fig = plt.figure(figsize=(15, 9.5))
wmax = max(np.nanmax(p["ds"].weight.values) for p in periods)
for k, p in enumerate(periods):
    ax = map_axes(fig, (2, len(periods), k + 1))
    im1 = ax.pcolormesh(lon, lat, p["ds"].weight.values, cmap=SEQ, vmin=0, vmax=wmax,
                        shading="nearest", transform=ccrs.PlateCarree())
    decorate(ax)
    ax.set_title(f"{p['label']}  관측 가중치(유효 관측 수)", fontsize=10)
    ax = map_axes(fig, (2, len(periods), len(periods) + k + 1))
    rel = 100 * p["ds"].no2_unc.values / np.clip(p["ds"].no2.values, 1, None)
    im2 = ax.pcolormesh(lon, lat, rel, cmap=SEQ, vmin=0, vmax=100,
                        shading="nearest", transform=ccrs.PlateCarree())
    decorate(ax)
    ax.set_title(f"{p['label']}  상대 불확도 (%)", fontsize=10)
fig.subplots_adjust(right=0.9, hspace=0.18)
fig.colorbar(im1, cax=fig.add_axes([0.92, 0.55, 0.012, 0.3]), label="weight")
fig.colorbar(im2, cax=fig.add_axes([0.92, 0.13, 0.012, 0.3]), label="불확도 / NO$_2$ (%)")
fig.suptitle("데이터 품질 — 유효 관측 수가 적고 상대 불확도가 큰 곳은 해석 주의", fontsize=13)
fig.savefig(OUT_DIR / "06_data_quality.png", bbox_inches="tight")
plt.close(fig)

print(f"\n완료. 결과 저장 위치: {OUT_DIR}")
