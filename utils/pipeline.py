# utils/pipeline.py
from __future__ import annotations
import os, shutil, tempfile, pathlib, json
from typing import Iterable, List, Dict, Optional
from datetime import datetime
from pathlib import Path

from image_to_json_generator import (
    process_images_to_individual_json,
    prepare_data_for_qgis,
)

# משתמשים בעוגן החדש שלך
from align_to_orthophoto import auto_anchor_with_esri as align_to_orthophoto

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}

def _is_image(p: str) -> bool:
    return pathlib.Path(p).suffix.lower() in IMAGE_EXT

def _gather_images_in_dir(dir_path: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            full = os.path.join(root, f)
            if _is_image(full):
                out.append(full)
    return out

def _create_session_dir() -> tuple[str, str]:
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(tempfile.gettempdir(), f"whitening_{session_name}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir, session_name

def _image_name_from_json(json_path: str) -> str:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d.get("BasicData", {}).get("imageFile") or (Path(json_path).stem + ".JPG")
    except Exception:
        return Path(json_path).stem + ".JPG"

def _read_resolution_from_json(json_path: str) -> Optional[float]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return (
            d.get("BasicData", {}).get("resolutionMperPx")
            or d.get("ImageGeo", {}).get("resolutionMperPx")
            or d.get("ImageGeo", {}).get("groundResolutionM")
        )
    except Exception:
        return None

def _copy_image_to_output(img_path: str, output_dir: str) -> str:
    """מעתיקה את קובץ התמונה ל-output כדי שיישב לצד ה-JSON/JPW הסופיים."""
    os.makedirs(output_dir, exist_ok=True)
    dst = os.path.join(output_dir, os.path.basename(img_path))
    try:
        # מעדיף להחליף/לרענן עותק ב-output כדי להיות בטוח שהוא קיים ועדכני
        shutil.copy2(img_path, dst)
    except Exception as e:
        print(f"⚠️ שגיאה בהעתקת תמונה ל-output: {img_path} → {dst}: {e}")
    return dst

def run_whitening(
    selected_paths: Iterable[str],
    drone_type: str,
    log_path: str | None = None,
    skip_log: bool = False,
) -> Dict:
    # 1) תיקיית סשן
    session_dir, session_name = _create_session_dir()

    # 2) העתקת תמונות
    copied = 0
    for p in selected_paths:
        if not p:
            continue
        if os.path.isdir(p):
            for img in _gather_images_in_dir(p):
                shutil.copy2(img, os.path.join(session_dir, os.path.basename(img)))
                copied += 1
        elif _is_image(p) and os.path.isfile(p):
            shutil.copy2(p, os.path.join(session_dir, os.path.basename(p)))
            copied += 1
    if copied == 0:
        raise RuntimeError("לא נמצאו תמונות להלבנה.")

    # 3) config.json
    cfg = {"drone_type": drone_type, "log_path": log_path, "skip_log": bool(skip_log)}
    with open(os.path.join(session_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # 4) הפקת JSON לכל תמונה
    session_used = process_images_to_individual_json(session_dir, drone_type=drone_type)

    # 5) מריצים קודם prepare_data_for_qgis כדי ש-JPW ייווצר ב-output
    print("ℹ️ prepare_data_for_qgis (pass #1) — יצירת JPW ראשוני ב-output")
    prepare_data_for_qgis(session_used)

    # 6) עיגון מול אורתופוטו: יעדכן את ה-JPW/JSON בתיקיית output
    output_dir = os.path.join(session_used, "output")
    os.makedirs(output_dir, exist_ok=True)

    for jf in os.listdir(output_dir):
        if not jf.lower().endswith(".json"):
            continue

        json_path = os.path.join(output_dir, jf)
        img_name = _image_name_from_json(json_path)
        img_path_session = os.path.join(session_used, img_name)  # התמונה המקורית יושבת בשורש הסשן

        # ה-JPW אמור להיות ב-output אחרי prepare_data_for_qgis
        jpw_path = os.path.join(output_dir, Path(img_name).stem + ".jpw")
        if not os.path.exists(jpw_path):
            # fallback: אם ה-JPW ליד התמונה
            alt_jpw = os.path.splitext(img_path_session)[0] + ".jpw"
            if os.path.exists(alt_jpw):
                jpw_path = alt_jpw

        # גם אם אין עיגון, נדאג שה-IMAGE ייכנס ל-output
        _copy_image_to_output(img_path_session, output_dir)

        if not (os.path.exists(img_path_session) and os.path.exists(jpw_path)):
            print(f"⚠️ דילוג על עיגון: חסר קובץ (image? {os.path.exists(img_path_session)} / jpw? {os.path.exists(jpw_path)}) עבור {img_name}")
            continue

        try:
            resolution = _read_resolution_from_json(json_path)
            print(f"🔧 Anchoring {Path(img_path_session).name} → יעדכן {Path(jpw_path).name} ואת ה-JSON")
            align_to_orthophoto(
                drone_image_path=img_path_session,
                jpw_in_path=jpw_path,
                jpw_out_path=jpw_path,     # דריסה במקום
                json_in_path=json_path,
                json_out_path=json_path,   # עדכון אותו JSON (שכבר ב-output)
                resolution_m_per_px=resolution,
                margin_factor=2.2,
                max_zoom=18,
                force_epsg="EPSG:4326",
                add_rotation_deg=0.0,
                do_xy_refine=True,
                nudge_east_m=0.0,
                nudge_north_m=0.0,
                xy_cc_min=0.05,
                xy_max_shift_m=2.0,
                prefer_roads=True,
                auto_flip_180=True,
                enforce_north_up=True,
                ortho_path=None,  # לשימוש באורתו מקומי: הציבי כאן נתיב GeoTIFF
            )
        except Exception as e:
            print(f"⚠️ עיגון התמונה נכשל עבור {img_name}: {e}")

        # אחרי העיגון — נוודא שוב שהעתק עדכני של קובץ התמונה נמצא ב-output
        _copy_image_to_output(img_path_session, output_dir)

    # 7) מריצים שוב prepare_data_for_qgis כדי להעתיק את הקבצים המעודכנים ל-TO_QGIS
    print("♻️ prepare_data_for_qgis (pass #2) — העתקת הקבצים המעודכנים ל-TO_QGIS")
    prepare_data_for_qgis(session_used)

    # 8) יצירת ZIP
    to_qgis_dir = os.path.join(session_used, "TO_QGIS")
    zip_path = shutil.make_archive(os.path.join(session_used, "TO_QGIS"), "zip", to_qgis_dir)

    # 9) תוצאות
    fail_dir = os.path.join(session_used, "fail_output")
    results: Dict[str, Dict] = {}
    if os.path.isdir(output_dir):
        for jf in sorted(os.listdir(output_dir)):
            if jf.lower().endswith(".json"):
                jp = os.path.join(output_dir, jf)
                img_name = _image_name_from_json(jp)
                img_out = os.path.join(output_dir, img_name)
                results[img_name] = {
                    "status": "success",
                    "json_path": jp,
                    "image_path": img_out if os.path.exists(img_out) else None,
                }
    if os.path.isdir(fail_dir):
        for jf in sorted(os.listdir(fail_dir)):
            if jf.lower().endswith(".json"):
                jp = os.path.join(fail_dir, jf)
                img_name = _image_name_from_json(jp)
                results.setdefault(img_name, {
                    "status": "failed",
                    "json_path": jp,
                    "image_path": None,
                })

    return {
        "session_dir": session_used,
        "zip_path": zip_path,
        "output_dir": output_dir,
        "fail_output_dir": fail_dir,
        "to_qgis_dir": to_qgis_dir,
        "results": results,
    }
