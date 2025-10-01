# auto_align_with_esri.py — rotate+XY refine vs Esri OR local ortho, roads masks, 180° decision, North-Up
import argparse, json, math, pathlib
from io import BytesIO
from typing import Tuple, Optional
import cv2, numpy as np, requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---- optional rasterio (for local ortho) ----
HAS_RASTERIO = True
try:
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    from rasterio.enums import Resampling
except Exception:
    HAS_RASTERIO = False

EARTH_RADIUS_M = 6378137.0
TILE_SIZE = 256
ESRI_TILE_URLS = [
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
]

# ---------------- HTTP session ----------------
_session = requests.Session()
_session.mount("https://", HTTPAdapter(
    max_retries=Retry(total=6, connect=6, read=6, backoff_factor=0.8,
                      status_forcelist=(429, 500, 502, 503, 504), raise_on_status=False),
    pool_connections=16, pool_maxsize=32
))
_session.headers.update({"User-Agent": "align-script/1.0"})

# ---------------- Worldfile utils ----------------
def read_jpw(path: str):
    A, D, B, E, C, F = [float(x.strip()) for x in open(path, "r", encoding="utf-8").read().strip().splitlines()]
    return A, D, B, E, C, F

def write_jpw(path: str, A, D, B, E, C, F):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{A:.12f}\n{D:.12f}\n{B:.12f}\n{E:.12f}\n{C:.12f}\n{F:.12f}\n")

def write_worldfiles_for_jpeg(jpg_path: str, A, D, B, E, C, F):
    p = pathlib.Path(jpg_path)
    write_jpw(str(p.with_suffix(".jpw")), A, D, B, E, C, F)
    write_jpw(str(p.with_suffix(".jgw")), A, D, B, E, C, F)
    aux = pathlib.Path(str(p) + ".aux.xml")
    if aux.exists():
        print(f"[WARN] Found AUX: {aux} → it may override the worldfile in GIS.")

def image_center_px(w: int, h: int) -> Tuple[float, float]:
    return (w - 1) / 2.0, (h - 1) / 2.0

def world_from_pixel(A, D, B, E, C, F, i, j):
    return A * i + B * j + C, D * i + E * j + F

def solve_C_F_to_keep_point(A, D, B, E, i0, j0, x_target, y_target):
    return x_target - (A * i0 + B * j0), y_target - (D * i0 + E * j0)

# ---------------- CRS helpers ----------------
def infer_epsg_from_transform(A, D, B, E, C, F, w, h):
    s = 0.5 * (math.hypot(A, D) + math.hypot(B, E))
    ic, jc = image_center_px(w, h)
    x0, y0 = world_from_pixel(A, D, B, E, C, F, ic, jc)
    if (-180.5 <= x0 <= 180.5) and (-90.5 <= y0 <= 90.5) and s < 1e-3:
        return "EPSG:4326"
    if (abs(x0) > 1000 and abs(y0) > 1000) and (1e-3 <= s <= 10):
        return "EPSG:3857"
    return "UNKNOWN"

def write_prj(path: str, epsg: str):
    if epsg == "EPSG:3857":
        wkt = ('PROJCS["WGS_84_Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",'
               'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
               'UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],'
               'PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],'
               'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
               'UNIT["metre",1]]')
    elif epsg == "EPSG:4326":
        wkt = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
               'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
    else:
        return
    open(path, "w", encoding="utf-8").write(wkt)

# ---------------- WebMerc / Tiles ----------------
def lonlat_to_webmerc(lon, lat):
    return math.radians(lon) * EARTH_RADIUS_M, math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * EARTH_RADIUS_M

def webmerc_to_lonlat(x, y):
    return math.degrees(x / EARTH_RADIUS_M), math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2)

def ground_res_m_per_px(lat, z):
    return (math.cos(math.radians(lat)) * 2 * math.pi * EARTH_RADIUS_M) / (TILE_SIZE * (2 ** z))

def choose_zoom_for_resolution(target, lat, max_z=21, min_z=10):
    for z in range(min_z, max_z + 1):
        if ground_res_m_per_px(lat, z) <= target:
            return z
    return max_z

def webmerc_to_tilexy(x, y, z):
    scale = TILE_SIZE * (2 ** z)
    minx = -math.pi * EARTH_RADIUS_M
    maxy = math.pi * EARTH_RADIUS_M
    px = (x - minx) / (2 * math.pi * EARTH_RADIUS_M) * scale
    py = (maxy - y) / (2 * math.pi * EARTH_RADIUS_M) * scale
    tx, ty = int(px // TILE_SIZE), int(py // TILE_SIZE)
    return tx, ty, int(scale // TILE_SIZE), TILE_SIZE, px - tx * TILE_SIZE, py - ty * TILE_SIZE

def bbox_to_tile_range(min_lon, min_lat, max_lon, max_lat, z):
    x0, y0 = lonlat_to_webmerc(min_lon, min_lat)
    x1, y1 = lonlat_to_webmerc(max_lon, max_lat)
    tx0, ty0, _, _, _, _ = webmerc_to_tilexy(x0, y0, z)
    tx1, ty1, _, _, _, _ = webmerc_to_tilexy(x1, y1, z)
    xmin, xmax = sorted([tx0, tx1])
    ymin, ymax = sorted([ty0, ty1])
    return xmin, ymin, xmax, ymax

def clamp_tile_index(v, z): return max(0, min((1 << z) - 1, v))

def fetch_tile(z, x, y, timeout=30):
    for tpl in ESRI_TILE_URLS:
        try:
            r = _session.get(tpl.format(z=z, y=y, x=x), timeout=timeout)
            if r.status_code == 200:
                return Image.open(BytesIO(r.content)).convert("RGB")
        except requests.RequestException:
            pass
    return None

def stitch_tiles(z, xmin, ymin, xmax, ymax):
    cols, rows = xmax - xmin + 1, ymax - ymin + 1
    big = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    for j, ty in enumerate(range(ymin, ymax + 1)):
        for i, tx in enumerate(range(xmin, xmax + 1)):
            t = fetch_tile(z, tx, ty) or Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
            big.paste(t, (i * TILE_SIZE, j * TILE_SIZE))
    return big

# ---------------- Local Ortho (NEW) ----------------
def _minmax_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-6:
        return np.clip(arr, 0, 255).astype(np.uint8)
    out = (arr - mn) / (mx - mn) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def load_local_ortho_crop(
    ortho_path: str,
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    bbox_crs: str = "EPSG:4326"
) -> np.ndarray:
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio לא מותקן. התקיני: pip install rasterio")
    with rasterio.open(ortho_path) as src:
        if src.crs is None:
            raise ValueError("לקובץ האורתופוטו אין CRS.")
        # המרת גבולות ל-CRS של הרסטר
        if str(src.crs) != bbox_crs:
            bounds_src = transform_bounds(bbox_crs, src.crs, min_lon, min_lat, max_lon, max_lat, densify_pts=16)
        else:
            bounds_src = (min_lon, min_lat, max_lon, max_lat)
        # חלון קריאה
        win = from_bounds(*bounds_src, transform=src.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if win.width < 2 or win.height < 2:
            raise ValueError("החלון המבוקש מחוץ לגבולות האורתופוטו.")
        # קריאת ערוצים
        bands = [1, 2, 3] if src.count >= 3 else [1]
        data = src.read(
            bands,
            window=win,
            resampling=Resampling.bilinear
        )
        # התאמת dtype ל-8bit
        if data.dtype != np.uint8:
            data = _minmax_to_uint8(data)
        # ל-RGB
        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        img = np.transpose(data[:3], (1, 2, 0))  # HWC
        return img

# ---------------- Road mask (optional) ----------------
def _auto_canny_thresholds(img):
    v = float(np.median(img))
    return max(0, int(0.66 * v)), min(255, int(1.33 * v))

def make_road_mask(gray, min_edge_ratio=0.01, min_lines=10):
    g = cv2.GaussianBlur(gray, (5, 5), 1.2)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    g = clahe.apply(g)
    lo, hi = _auto_canny_thresholds(g)
    e = cv2.Canny(g, lo, hi, L2gradient=True)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    e = cv2.dilate(e, k, 1)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, k, iterations=1)
    edge_ratio = e.mean() / 255.0
    lines = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10)
    num_lines = 0 if lines is None else len(lines)
    score = 0.5 * edge_ratio + 0.5 * min(1.0, num_lines / 50.0)
    ok = (edge_ratio >= min_edge_ratio) or (num_lines >= min_lines)
    if not ok:
        return None, 0.0
    mask = cv2.GaussianBlur((e > 0).astype(np.uint8) * 255, (5, 5), 0)
    return mask, float(score)

# ---------------- Utils ----------------
def _resize_same(g1, g2):
    H = min(g1.shape[0], g2.shape[0]); W = min(g1.shape[1], g2.shape[1])
    if g1.shape[:2] != (H, W): g1 = cv2.resize(g1, (W, H))
    if g2.shape[:2] != (H, W): g2 = cv2.resize(g2, (W, H))
    return g1, g2

def _rotate_gray(g, theta_deg):
    h, w = g.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), theta_deg, 1.0)
    return cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

# ---------------- Rotation estimation helpers ----------------
def _histogram_match(src_gray: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
    s = src_gray.ravel()
    r = ref_gray.ravel()
    s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(r, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / s.size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / r.size
    interp_values = np.interp(s_quantiles, r_quantiles, r_values)
    matched = interp_values[bin_idx].reshape(src_gray.shape).astype(np.uint8)
    return matched

def _anms_grid(keypoints, grid=(8,8), shape=None, max_per_cell=50):
    if not keypoints:
        return []
    H, W = shape
    gr, gc = grid
    cell_h = max(1, H // gr)
    cell_w = max(1, W // gc)
    cells = [[[] for _ in range(gc)] for _ in range(gr)]
    for kp in keypoints:
        r = min(gr-1, int(kp.pt[1] // cell_h))
        c = min(gc-1, int(kp.pt[0] // cell_w))
        cells[r][c].append(kp)
    selected = []
    for r in range(gr):
        for c in range(gc):
            ks = sorted(cells[r][c], key=lambda k: -k.response)
            selected.extend(ks[:max_per_cell])
    return selected

def _estimate_rotation_logpolar(g1, g2):
    g1, g2 = _resize_same(g1, g2)
    def spectrum(img):
        f = np.fft.fftshift(np.fft.fft2(img)); m = np.log1p(np.abs(f))
        return cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    s1, s2 = spectrum(g1), spectrum(g2)
    H, W = s1.shape[:2]; center = (W / 2, H / 2); R = int(np.hypot(W / 2, H / 2)); A = 360
    lp1 = cv2.warpPolar(s1, (A, R), center, R, cv2.WARP_POLAR_LOG | cv2.WARP_FILL_OUTLIERS)
    lp2 = cv2.warpPolar(s2, (A, R), center, R, cv2.WARP_POLAR_LOG | cv2.WARP_FILL_OUTLIERS)
    lp1 = cv2.transpose(lp1).astype(np.float32); lp2 = cv2.transpose(lp2).astype(np.float32)
    shift, response = cv2.phaseCorrelate(lp1, lp2)
    angle_per_pix = 360.0 / lp1.shape[1]
    theta = -(shift[0] * angle_per_pix)
    theta = (theta + 180) % 360 - 180
    return float(theta), float(response)

# ---------------- Rotation estimation ----------------
def estimate_rotation_deg(drone_img_gray: np.ndarray, ortho_gray: np.ndarray,
                          mask_drone: Optional[np.ndarray]=None,
                          mask_ortho: Optional[np.ndarray]=None,
                          min_inliers: int = 20) -> float:
    d0 = cv2.GaussianBlur(drone_img_gray, (3,3), 0)
    o0 = cv2.GaussianBlur(ortho_gray,  (3,3), 0)
    d  = _histogram_match(d0, o0)

    H, W = d.shape[:2]

    def detect_and_match(detector_name: str, ratio=0.78):
        if detector_name == "SIFT":
            try:
                det = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02, edgeThreshold=12, sigma=1.2)
                norm = cv2.NORM_L2
            except Exception:
                return None
        elif detector_name == "AKAZE":
            det = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, threshold=0.001)
            norm = cv2.NORM_HAMMING
        else:
            det = cv2.ORB_create(nfeatures=8000, WTA_K=4, edgeThreshold=15, patchSize=31, fastThreshold=7)
            norm = cv2.NORM_HAMMING

        k1 = det.detect(d, mask_drone)
        k2 = det.detect(o0, mask_ortho)
        k1 = _anms_grid(k1, grid=(8,8), shape=(H, W), max_per_cell=30)
        k2 = _anms_grid(k2, grid=(8,8), shape=o0.shape,    max_per_cell=30)

        if len(k1) < 12 or len(k2) < 12:
            return None
        k1, d1 = det.compute(d,  k1)
        k2, d2 = det.compute(o0, k2)
        if d1 is None or d2 is None:
            return None

        if norm == cv2.NORM_L2:
            matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
            d1 = np.float32(d1); d2 = np.float32(d2)
            raw = matcher.knnMatch(d1, d2, k=2)
        else:
            bf = cv2.BFMatcher(norm, crossCheck=False)
            raw = bf.knnMatch(d1, d2, k=2)

        good = [m for m,n in raw if m.distance < ratio * n.distance]
        if len(good) < 12 and ratio < 0.85:
            good = [m for m,n in raw if m.distance < 0.85 * n.distance]
        if len(good) < 12:
            return None

        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=4.0, maxIters=8000, confidence=0.995)
        if M is None:
            return None
        ninl = int(inl.sum()) if inl is not None else 0
        if ninl < min_inliers:
            return None
        a, b = float(M[0,0]), float(M[1,0])
        theta = math.degrees(math.atan2(b, a))
        return theta, ninl

    for name in ("SIFT", "AKAZE", "ORB"):
        out = detect_and_match(name)
        if out is not None:
            theta, ninl = out
            print(f"[INFO] rotation by {name}: {theta:.3f}° CCW (inliers {ninl})")
            return float((theta + 180) % 360 - 180)

    theta_lp, resp = _estimate_rotation_logpolar(d, o0)
    print(f"[INFO] log-polar fallback: {theta_lp:.3f}° (resp {resp:.3f})")
    return float((theta_lp + 180) % 360 - 180)

# -------- coarse sweep (to lock angle) --------
def _prep_for_ecc(g: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (5,5), 1.2)
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    return mag

def coarse_rotation_sweep(drone_gray: np.ndarray,
                          ortho_gray: np.ndarray,
                          theta_seed: float,
                          window_deg: float = 60.0,
                          step_deg: float = 0.5,
                          mask_ortho: Optional[np.ndarray] = None,
                          fallback_full: bool = True) -> float:
    max_side = 700
    def down(g):
        h, w = g.shape[:2]
        s = max(1.0, max(h, w) / max_side)
        if s > 1.0:
            g = cv2.resize(g, (int(w/s), int(h/s)))
        return g
    dg = down(drone_gray)
    og = down(ortho_gray)

    A0 = _prep_for_ecc(dg)
    B0 = _prep_for_ecc(og)
    if mask_ortho is not None:
        msk = cv2.resize(mask_ortho, (B0.shape[1], B0.shape[0])) > 0
        W = cv2.normalize(msk.astype(np.float32), None, 0.3, 1.0, cv2.NORM_MINMAX)
        B0 = B0 * W

    def ecc_for(theta):
        d_rot = _rotate_gray((A0*255).astype(np.uint8), theta).astype(np.float32)/255.0
        B = B0.astype(np.float32)
        warp = np.eye(2,3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)
        try:
            cc, _ = cv2.findTransformECC(B, d_rot, warp, cv2.MOTION_TRANSLATION, crit)
        except cv2.error:
            cc = -1e12
        return float(cc)

    best_theta = theta_seed
    best_cc = -1e12
    start = theta_seed - window_deg
    stop  = theta_seed + window_deg
    t = start
    while t <= stop + 1e-9:
        cc = ecc_for(t)
        if cc > best_cc:
            best_cc, best_theta = cc, t
        t += step_deg

    print(f"[SWEEP] window={window_deg}°, step={step_deg}° → best={best_theta:.3f}° (ECC={best_cc:.4f})")

    if fallback_full and best_cc < 0.002:
        best_theta_full, best_cc_full = best_theta, best_cc
        t = -180.0
        while t <= 180.0 + 1e-9:
            cc = ecc_for(t)
            if cc > best_cc_full:
                best_cc_full, best_theta_full = cc, t
            t += max(2.0, step_deg)
        if best_cc_full > best_cc + 1e-3:
            best_theta, best_cc = best_theta_full, best_cc_full
            print(f"[SWEEP] full [-180,180] → best={best_theta:.3f}° (ECC={best_cc:.4f})")

    return float((best_theta + 180) % 360 - 180)

# ---------------- Rotation refine grid ----------------
def refine_rotation_grid(drone_gray, ortho_gray, theta_init, mask_ortho=None,
                         offsets=(0, 90, -90, 180, -180), local_range=25.0, step=0.5):
    best_theta, best_cc = theta_init, -1e12
    dg0 = drone_gray.copy(); og0 = ortho_gray.copy()
    for off in offsets:
        base = theta_init + off
        a = base - local_range
        while a <= base + local_range + 1e-9:
            try:
                d_rot = cv2.warpAffine(
                    dg0, cv2.getRotationMatrix2D((dg0.shape[1] / 2, dg0.shape[0] / 2), a, 1.0),
                    (dg0.shape[1], dg0.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
                )
                A, B = _resize_same(d_rot.astype(np.float32) / 255.0, og0.astype(np.float32) / 255.0)
                warp = np.eye(2, 3, dtype=np.float32)
                crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)
                msk = cv2.resize(mask_ortho, (B.shape[1], B.shape[0])) if mask_ortho is not None else None
                try:
                    cc, _ = cv2.findTransformECC(B, A, warp, cv2.MOTION_TRANSLATION, crit, msk)
                except TypeError:
                    if msk is not None:
                        W = cv2.normalize(msk.astype(np.float32) / 255.0, None, 0.3, 1.0, cv2.NORM_MINMAX)
                        A *= W; B *= W
                    cc, _ = cv2.findTransformECC(B, A, warp, cv2.MOTION_TRANSLATION, crit)
            except cv2.error:
                cc = -1e12
            if cc > best_cc:
                best_cc, best_theta = cc, a
            a += step
    print(f"[INFO] grid refine: best={best_theta:.3f}° (ECC={best_cc:.4f})")
    return float((best_theta + 180) % 360 - 180)

# ---------------- XY refine ----------------
def refine_xy_translation(drone_rot_gray, ortho_gray, mask_ortho=None):
    d = drone_rot_gray.astype(np.float32) / 255.0
    o = ortho_gray.astype(np.float32) / 255.0
    Hd, Wd = d.shape[:2]; Ho, Wo = o.shape[:2]
    H, W = min(Hd, Ho), min(Wd, Wo)
    d_res = cv2.resize(d, (W, H)) if (Hd, Wd) != (H, W) else d
    o_res = cv2.resize(o, (W, H)) if (Ho, Wo) != (H, W) else o
    sx, sy = (Wd / W), (Hd / H)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
    msk = cv2.resize(mask_ortho, (W, H)) if mask_ortho is not None else None
    try:
        cc, warp = cv2.findTransformECC(o_res, d_res, warp, cv2.MOTION_TRANSLATION, crit, msk)
    except TypeError:
        if msk is not None:
            Wt = cv2.normalize(msk.astype(np.float32) / 255.0, None, 0.3, 1.0, cv2.NORM_MINMAX)
            d_res *= Wt; o_res *= Wt
        cc, warp = cv2.findTransformECC(o_res, d_res, warp, cv2.MOTION_TRANSLATION, crit)
    tx, ty = float(warp[0, 2]) * sx, float(warp[1, 2]) * sy
    print(f"[INFO] XY refine (ECC): tx={tx:.2f}px, ty={ty:.2f}px, cc={cc:.4f}")
    return tx, ty, cc

# ---------------- JSON ----------------
def update_json(json_in_path: Optional[str], json_out_path: Optional[str],
                theta_ccw: float, jpw_out_name: str, crs: Optional[str] = None):
    if not (json_in_path and json_out_path):
        return
    meta = json.load(open(json_in_path, "r", encoding="utf-8"))
    ig = meta.get("ImageGeo", {})
    ig["rotationDegreesCW"] = round(-float(theta_ccw), 4)
    ig["worldFile"] = jpw_out_name
    if crs:
        ig["crs"] = crs
    meta["ImageGeo"] = ig
    json.dump(meta, open(json_out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

# ---------------- Variant choose ----------------
def _post(M, th): A, D, B, E = M; c, s = math.cos(th), math.sin(th); return (A*c + B*s, D*c + E*s, -A*s + B*c, -D*s + E*c)
def _pre(M, th):  A, D, B, E = M; c, s = math.cos(th), math.sin(th); return (c*A - s*D, s*A + c*D, c*B - s*E, s*B + c*E)
def _swap_flip(A, D, B, E, swap=False, fx=False, fy=False):
    if swap: A, D, B, E = D, A, E, B
    if fx:   A, B = -A, -B
    if fy:   D, E = -D, -E
    return A, D, B, E
def _ang(dx, dy):   return math.degrees(math.atan2(dy, dx))
def _angdiff(a, b): return abs((a - b + 180) % 360 - 180)
def _score(A, D, B, E): return _angdiff(_ang(A, D), 0.0) + _angdiff(_ang(B, E), -90.0)
def choose_best_variant(A0, D0, B0, E0, theta_deg):
    th = math.radians(theta_deg); cand = []
    for prepost in ("post", "pre"):
        for sign in (+1, -1):
            M1 = _post((A0, D0, B0, E0), sign * th) if prepost == "post" else _pre((A0, D0, B0, E0), sign * th)
            for sw in (False, True):
                for fx in (False, True):
                    for fy in (False, True):
                        A, D, B, E = _swap_flip(*M1, swap=sw, fx=fx, fy=fy)
                        cand.append((_score(A, D, B, E), A, D, B, E))
    cand.sort(key=lambda x: x[0]); _, A, D, B, E = cand[0]
    return A, D, B, E

# ---------------- 180° pick ----------------
def _ecc_score_for_theta(drone_gray, ortho_gray, theta_deg, mask_ortho=None):
    try:
        d_rot = _rotate_gray(drone_gray, theta_deg)
        A, B = _resize_same(d_rot.astype(np.float32)/255.0, ortho_gray.astype(np.float32)/255.0)
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        msk = cv2.resize(mask_ortho, (B.shape[1], B.shape[0])) if mask_ortho is not None else None
        try:
            cc, _ = cv2.findTransformECC(B, A, warp, cv2.MOTION_TRANSLATION, crit, msk)
        except TypeError:
            if msk is not None:
                W = cv2.normalize(msk.astype(np.float32)/255.0, None, 0.3, 1.0, cv2.NORM_MINMAX)
                A *= W; B *= W
            cc, _ = cv2.findTransformECC(B, A, warp, cv2.MOTION_TRANSLATION, crit)
        return float(cc)
    except cv2.error:
        return -1e12

def choose_theta_with_180(drone_gray, ortho_gray, theta_init, mask_ortho=None, margin=2.0):
    cc0   = _ecc_score_for_theta(drone_gray, ortho_gray, theta_init,       mask_ortho)
    cc180 = _ecc_score_for_theta(drone_gray, ortho_gray, theta_init + 180, mask_ortho)
    print(f"[INFO] 180-check: ECC(θ)={cc0:.4f}, ECC(θ+180)={cc180:.4f}")
    if cc180 > cc0 + 1e-3:
        return float(((theta_init + 180.0) + 180) % 360 - 180)
    return float((theta_init + 180) % 360 - 180)

# ---------------- Core pipeline ----------------
def _run_pipeline(drone_image_path, jpw_in_path, jpw_out_path,
                  json_in_path, json_out_path,
                  resolution_m_per_px, margin_factor, max_zoom,
                  force_epsg, add_rotation_deg, do_xy_refine,
                  nudge_east_m, nudge_north_m,
                  xy_cc_min, xy_max_shift_m,
                  prefer_roads, auto_flip_180, enforce_north_up,
                  ortho_path: Optional[str] = None):

    bgr = cv2.imread(drone_image_path, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(drone_image_path)
    h, w = bgr.shape[:2]; gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    A0, D0, B0, E0, C0, F0 = read_jpw(jpw_in_path)
    epsg_auto = infer_epsg_from_transform(A0, D0, B0, E0, C0, F0, w, h)
    epsg = force_epsg if (force_epsg in ("EPSG:4326", "EPSG:3857")) else epsg_auto
    print(f"[INFO] CRS chosen: {epsg} (auto='{epsg_auto}')")

    # Anchor (UL)
    x_anchor, y_anchor = world_from_pixel(A0, D0, B0, E0, C0, F0, 0.0, 0.0)
    print(f"[CHK] UL BEFORE: ({x_anchor:.6f}, {y_anchor:.6f})")

    # Resolution & center
    res = resolution_m_per_px if resolution_m_per_px is not None else max(0.01, 0.5 * (math.hypot(A0, D0) + math.hypot(B0, E0)))
    ic, jc = image_center_px(w, h); x_c, y_c = world_from_pixel(A0, D0, B0, E0, C0, F0, ic, jc)
    if epsg == "EPSG:4326": xwm, ywm = lonlat_to_webmerc(x_c, y_c); lon_c, lat_c = x_c, y_c
    else:                   xwm, ywm = x_c, y_c;                 lon_c, lat_c = webmerc_to_lonlat(x_c, y_c)

    # Ortho bbox
    z = choose_zoom_for_resolution(res, lat_c, max_z=max_zoom, min_z=10)
    gw, gh = res * w * margin_factor, res * h * margin_factor
    minx, maxx = xwm - gw/2, xwm + gw/2; miny, maxy = ywm - gh/2, ywm + gh/2
    min_lon, min_lat = webmerc_to_lonlat(minx, miny); max_lon, max_lat = webmerc_to_lonlat(maxx, maxy)

    # ---- Ortho source: local OR Esri
    if ortho_path:
        print(f"[INFO] Using LOCAL ortho: {ortho_path}")
        ortho_rgb = load_local_ortho_crop(
            ortho_path=ortho_path,
            min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat,
            bbox_crs="EPSG:4326"  # bbox שחישבנו הוא ב-WGS84
        )
        ortho_gray = cv2.cvtColor(ortho_rgb, cv2.COLOR_RGB2GRAY)
    else:
        xmin, ymin, xmax, ymax = bbox_to_tile_range(min_lon, min_lat, max_lon, max_lat, z)
        xmin, xmax = clamp_tile_index(xmin, z), clamp_tile_index(xmax, z)
        ymin, ymax = clamp_tile_index(ymin, z), clamp_tile_index(ymax, z)
        ortho = stitch_tiles(z, xmin, ymin, xmax, ymax)
        def gpx(lon, lat):
            x, y = lonlat_to_webmerc(lon, lat); tx, ty, _, _, ox, oy = webmerc_to_tilexy(x, y, z)
            return int(round((tx - xmin) * TILE_SIZE + ox)), int(round((ty - ymin) * TILE_SIZE + oy))
        gx0, gy1 = gpx(min_lon, min_lat); gx1, gy0 = gpx(max_lon, max_lat)
        gx0, gx1 = min(gx0, gx1), max(gx0, gx1); gy0, gy1 = min(gy0, gy1), max(gy0, gy1)
        ortho_crop = ortho.crop((gx0, gy0, gx1, gy1))
        ortho_gray = cv2.cvtColor(np.array(ortho_crop), cv2.COLOR_RGB2GRAY)

    # Road masks (optional)
    mask_drone, score_d = (None, 0.0)
    mask_ortho, score_o = (None, 0.0)
    if prefer_roads:
        mask_drone, score_d = make_road_mask(gray)
        mask_ortho, score_o = make_road_mask(ortho_gray)
        if mask_drone is None or mask_ortho is None:
            mask_drone = mask_ortho = None
            print("[INFO] road mask not confident → fallback to full image.")
        else:
            print(f"[INFO] road masks active (scores: drone={score_d:.3f}, ortho={score_o:.3f})")

    # ---------- Rotation ----------
    theta_ccw = estimate_rotation_deg(gray, ortho_gray, mask_drone, mask_ortho)
    theta_ccw += float(add_rotation_deg or 0.0)

    theta_ccw = coarse_rotation_sweep(
        drone_gray=gray, ortho_gray=ortho_gray, theta_seed=theta_ccw,
        window_deg=60.0, step_deg=0.5, mask_ortho=mask_ortho, fallback_full=True
    )
    theta_ccw = refine_rotation_grid(
        gray, ortho_gray, theta_ccw, mask_ortho,
        offsets=(0, 90, -90, 180, -180), local_range=25.0, step=0.5
    )
    print(f"[INFO] final theta (CCW): {theta_ccw:.3f}°")

    if auto_flip_180:
        theta_ccw = choose_theta_with_180(gray, ortho_gray, theta_ccw, mask_ortho, margin=2.0)
        print(f"[INFO] 180-check picked: {theta_ccw:.3f}°")

    # Variant + keep UL
    A1, D1, B1, E1 = choose_best_variant(A0, D0, B0, E0, theta_ccw)
    C1, F1 = solve_C_F_to_keep_point(A1, D1, B1, E1, 0.0, 0.0, x_anchor, y_anchor)

    # Enforce North-Up
    if enforce_north_up and epsg in ("EPSG:4326", "EPSG:3857") and E1 > 0:
        A1, D1, B1, E1 = -A1, -D1, -B1, -E1
        C1, F1 = solve_C_F_to_keep_point(A1, D1, B1, E1, 0.0, 0.0, x_anchor, y_anchor)
        print("[FIX] Flipped 180° to enforce North-Up (E>0 → E<0).")

    # XY refine (gated)
    if do_xy_refine:
        d_rot = _rotate_gray(gray, theta_ccw)
        tx, ty, cc = refine_xy_translation(d_rot, ortho_gray, mask_ortho)
        dC = A1 * tx + B1 * ty; dF = D1 * tx + E1 * ty
        if epsg == "EPSG:3857":
            shift_m = (dC**2 + dF**2)**0.5
        else:
            lat_mid = lat_c
            m_per_deg_lat = 111320.0
            m_per_deg_lon = math.cos(math.radians(lat_mid)) * 111320.0
            shift_m = ((dC * m_per_deg_lon)**2 + (dF * m_per_deg_lat)**2)**0.5
        print(f"[CHK] XY candidate: cc={cc:.4f}, shift≈{shift_m:.2f} m")
        if cc >= xy_cc_min and shift_m <= xy_max_shift_m:
            C1 += dC; F1 += dF; print("[OK] XY refine applied.")
        else:
            print("[SKIP] XY refine ignored.")

    # Manual nudge (E+, N+)
    if abs(nudge_east_m) > 0.0 or abs(nudge_north_m) > 0.0:
        if epsg == "EPSG:3857":
            C1 += nudge_east_m; F1 -= nudge_north_m
        elif epsg == "EPSG:4326":
            lat_mid = lat_c
            C1 += nudge_east_m / (math.cos(math.radians(lat_mid)) * 111320.0)
            F1 -= nudge_north_m / 111320.0

    # Write outputs
    write_worldfiles_for_jpeg(drone_image_path, A1, D1, B1, E1, C1, F1)
    if jpw_out_path:
        write_jpw(jpw_out_path, A1, D1, B1, E1, C1, F1)
    prj = pathlib.Path(drone_image_path).with_suffix(".prj")
    if epsg in ("EPSG:4326", "EPSG:3857"):
        write_prj(str(prj), epsg); print(f"[OK] wrote PRJ: {prj.name} ({epsg})")
    jpw_name = pathlib.Path(jpw_out_path).name if jpw_out_path else pathlib.Path(drone_image_path).with_suffix(".jpw").name
    if json_in_path and json_out_path:
        update_json(json_in_path, json_out_path, theta_ccw, jpw_name, crs=epsg)

    x2, y2 = world_from_pixel(A1, D1, B1, E1, C1, F1, 0.0, 0.0)
    print(f"[CHK] UL AFTER : ({x2:.6f}, {y2:.6f})  Δ=({x2-x_anchor:.6f},{y2-y_anchor:.6f})")
    return {"theta_deg_ccw": float(theta_ccw), "epsg": epsg}

# ---------------- Public API ----------------
def auto_anchor_with_esri(
    drone_image_path: str, jpw_in_path: str, jpw_out_path: str,
    json_in_path: Optional[str] = None, json_out_path: Optional[str] = None,
    resolution_m_per_px: Optional[float] = None, margin_factor: float = 2.4,
    max_zoom: int = 17, force_epsg: Optional[str] = None, add_rotation_deg: float = 0.0,
    do_xy_refine: bool = True, nudge_east_m: float = 0.0, nudge_north_m: float = 0.0,
    xy_cc_min: float = 0.05, xy_max_shift_m: Optional[float] = 2.0,
    prefer_roads: bool = False, auto_flip_180: bool = True, enforce_north_up: bool = True,
    ortho_path: Optional[str] = None
) -> dict:
    if xy_max_shift_m is None and resolution_m_per_px is not None:
        bgr = cv2.imread(drone_image_path, cv2.IMREAD_COLOR)
        H, W = bgr.shape[:2]
        xy_max_shift_m = min(2.0, 0.01 * max(H, W) * resolution_m_per_px)
    return _run_pipeline(
        drone_image_path, jpw_in_path, jpw_out_path,
        json_in_path, json_out_path,
        resolution_m_per_px, margin_factor, max_zoom,
        force_epsg, add_rotation_deg, do_xy_refine,
        nudge_east_m, nudge_north_m,
        xy_cc_min, xy_max_shift_m if xy_max_shift_m is not None else 2.0,
        prefer_roads, auto_flip_180, enforce_north_up,
        ortho_path=ortho_path
    )

# -------- Batch runner: process all images in a folder --------
def batch_anchor_folder(
    folder: str,
    resolution_m_per_px: Optional[float] = None,
    margin_factor: float = 2.4,
    max_zoom: int = 17,
    force_epsg: Optional[str] = "EPSG:4326",
    add_rotation_deg: float = 0.0,
    do_xy_refine: bool = True,
    nudge_east_m: float = 0.0,
    nudge_north_m: float = 0.0,
    xy_cc_min: float = 0.05,
    xy_max_shift_m: float = 2.0,
    prefer_roads: bool = False,
    auto_flip_180: bool = True,
    enforce_north_up: bool = True,
    recursive: bool = False,
    exts: tuple = (".JPG", ".jpg", ".jpeg"),
    ortho_path: Optional[str] = None,
):
    folder = pathlib.Path(folder)
    imgs = []
    for ext in exts:
        imgs.extend(folder.rglob(f"*{ext}") if recursive else folder.glob(f"*{ext}"))
    imgs = sorted(imgs)
    results = []

    print(f"[BATCH] Found {len(imgs)} images in {folder} (recursive={recursive}), ortho={'LOCAL' if ortho_path else 'Esri'}")

    for img in imgs:
        jpw = img.with_suffix(".jpw")
        jgw = img.with_suffix(".jgw")
        if not jpw.exists() and jgw.exists():
            jpw = jgw
        if not jpw.exists():
            msg = "missing JPW/JGW"
            print(f"[SKIP] {img.name}: {msg}")
            results.append((img.name, "skip", "", "", msg))
            continue

        jsn = img.with_suffix(".json")
        json_in  = str(jsn) if jsn.exists() else None
        json_out = str(jsn) if jsn.exists() else None

        try:
            r = auto_anchor_with_esri(
                drone_image_path=str(img),
                jpw_in_path=str(jpw),
                jpw_out_path=str(jpw),
                json_in_path=json_in,
                json_out_path=json_out,
                resolution_m_per_px=resolution_m_per_px,
                margin_factor=margin_factor,
                max_zoom=max_zoom,
                force_epsg=force_epsg,
                add_rotation_deg=add_rotation_deg,
                do_xy_refine=do_xy_refine,
                nudge_east_m=nudge_east_m,
                nudge_north_m=nudge_north_m,
                xy_cc_min=xy_cc_min,
                xy_max_shift_m=xy_max_shift_m,
                prefer_roads=prefer_roads,
                auto_flip_180=auto_flip_180,
                enforce_north_up=enforce_north_up,
                ortho_path=ortho_path,
            )
            results.append((img.name, "ok", f'{r.get("theta_deg_ccw", "")}', r.get("epsg", ""), ""))
            print(f"[OK] {img.name}: θ={r.get('theta_deg_ccw'):.3f}°, {r.get('epsg')}")
        except Exception as e:
            msg = str(e)
            print(f"[ERR] {img.name}: {msg}")
            results.append((img.name, "error", "", "", msg))

    out_csv = folder / "batch_results.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("image,status,theta_deg_ccw,epsg,message\n")
        for name, status, theta, epsg, msg in results:
            f.write(f"{name},{status},{theta},{epsg},{msg.replace(',', ';')}\n")
    print(f"[SUMMARY] wrote {out_csv}  ({sum(1 for r in results if r[1]=='ok')} OK / {len(imgs)} total)")
    return results

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Auto-align (rotate+XY) vs Esri or local ortho; roads-preference, 180° auto-pick, enforce North-Up.")
    # single-file mode
    ap.add_argument("--drone_image", default=None)
    ap.add_argument("--jpw_in", default=None)
    ap.add_argument("--jpw_out", default=None)
    ap.add_argument("--json_in", default=None)
    ap.add_argument("--json_out", default=None)

    # folder mode
    ap.add_argument("--folder", default=None, help="Process all images in this folder")
    ap.add_argument("--recursive", action="store_true", help="Scan subfolders too")
    ap.add_argument("--exts", default=".JPG,.jpg,.jpeg", help="Comma-separated list of image extensions")

    # shared params
    ap.add_argument("--resolution", type=float, default=None)
    ap.add_argument("--margin_factor", type=float, default=2.4)
    ap.add_argument("--max_zoom", type=int, default=17)
    ap.add_argument("--force_epsg", choices=["EPSG:4326", "EPSG:3857"], default="EPSG:4326")
    ap.add_argument("--add_deg", type=float, default=0.0)
    ap.add_argument("--no_xy_refine", action="store_true")
    ap.add_argument("--nudge_east_m", type=float, default=0.0)
    ap.add_argument("--nudge_north_m", type=float, default=0.0)
    ap.add_argument("--xy_cc_min", type=float, default=0.05)
    ap.add_argument("--xy_max_shift_m", type=float, default=2.0)
    ap.add_argument("--prefer_roads", action="store_true")
    ap.add_argument("--no_auto_flip_180", action="store_true")
    ap.add_argument("--no_enforce_north_up", action="store_true")
    ap.add_argument("--ortho_path", default=None, help="Path to local ortho GeoTIFF (optional)")
    args = ap.parse_args()

    if args.folder:
        exts = tuple(e.strip() for e in args.exts.split(",") if e.strip())
        batch_anchor_folder(
            folder=args.folder,
            resolution_m_per_px=args.resolution,
            margin_factor=args.margin_factor,
            max_zoom=args.max_zoom,
            force_epsg=args.force_epsg,
            add_rotation_deg=args.add_deg,
            do_xy_refine=not args.no_xy_refine,
            nudge_east_m=args.nudge_east_m,
            nudge_north_m=args.nudge_north_m,
            xy_cc_min=args.xy_cc_min,
            xy_max_shift_m=args.xy_max_shift_m,
            prefer_roads=args.prefer_roads,
            auto_flip_180=not args.no_auto_flip_180,
            enforce_north_up=not args.no_enforce_north_up,
            recursive=args.recursive,
            exts=exts,
            ortho_path=args.ortho_path,
        )
        return

    if not (args.drone_image and args.jpw_in and args.jpw_out):
        raise SystemExit("For single-file mode, provide --drone_image --jpw_in --jpw_out (or use --folder).")

    # res = auto_anchor_with_esri(
    #     drone_image_path=args.drone_image, jpw_in_path=args.jpw_in, jpw_out_path=args.jpw_out,
    #     json_in_path=args.json_in, json_out_path=args.json_out,
    #     resolution_m_per_px=args.resolution, margin_factor=args.margin_factor, max_zoom=args.max_zoom,
    #     force_epsg=args.force_epsg, add_rotation_deg=args.add_deg,
    #     do_xy_refine=not args.no_xy_refine,
    #     nudge_east_m=args.nudge_east_m, nudge_north_m=args.nudge_north_m,
    #     xy_cc_min=args.xy_cc_min, xy_max_shift_m=args.xy_max_shift_m,
    #     prefer_roads=args.prefer_roads, auto_flip_180=not args.no_auto_flip_180,
    #     enforce_north_up=not args.no_enforce_north_up,
    #     ortho_path=args.ortho_path
    # )
    # print(res)

# if __name__ == "__main__":
#     main()

# ---- batch run (your style) ----
# results = batch_anchor_folder(
#     folder=r"C:\Users\DELL\Desktop\tekem_frame\anchoring\output2",
#     resolution_m_per_px=0.05279,
#     margin_factor=2.2,
#     max_zoom=18,
#     force_epsg="EPSG:4326",
#     prefer_roads=True,
#     auto_flip_180=True,
#     enforce_north_up=True,
#     recursive=False,
#     xy_cc_min=0.05,
#     xy_max_shift_m=2.0,
#     # שימי כאן את האורתופוטו שלך אם תרצי לעבוד מקומי:
#     # ortho_path=r"C:\path\to\your\ortho_mosaic.tif",
#     ortho_path=None
# )
# print(results)
