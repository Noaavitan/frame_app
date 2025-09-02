# utils/pipeline.py
from __future__ import annotations
import os, shutil, tempfile, json, glob
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple

# פונקציות ההלבנה הקיימות שלך:
from image_to_json_generator import process_images_to_individual_json, prepare_data_for_qgis

def _stage_inputs(drone_type: str, log_path: Optional[str], image_paths: Iterable[str]) -> str:
    """
    יוצר תיקיית עבודה זמנית, מעתיק אליה תמונות (ולוג אם קיים),
    וכותב config.json עם פרטי רחפן/לוג.
    """
    workdir = Path(tempfile.mkdtemp(prefix="whitening_")).resolve()
    # תמונות
    for p in image_paths:
        src = Path(p)
        if src.is_file():
            shutil.copy2(src, workdir / src.name)
    # לוג (אופציונלי)
    if log_path:
        lp = Path(log_path)
        if lp.is_file():
            shutil.copy2(lp, workdir / lp.name)
    # קונפיג
    (workdir / "config.json").write_text(
        json.dumps(
            {"drone_type": drone_type, "log_file": Path(log_path).name if log_path else None},
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )
    return str(workdir)

def _collect_results(workdir: str, original_images: List[str]) -> Dict[str, Dict]:
    """
    סורק את תיקיות output / fail_output ומחזיר מפה:
    key = שם קובץ המקור (כולל סיומת), value = {status, json_path[, reason]}.
    התאמה לפי STEM (שם ללא סיומת) כדי לשייך JSON לתמונה.
    """
    wd = Path(workdir)
    out_dir = wd / "output"
    fail_dir = wd / "fail_output"

    results: Dict[str, Dict] = {}
    # מכין מיפוי מה-stem לרשימת שמות מקור (תמונות שנבחרו)
    stem_to_originals: Dict[str, List[str]] = {}
    for img in original_images:
        stem = Path(img).stem
        stem_to_originals.setdefault(stem, []).append(Path(img).name)

    # הצלחות
    for jp in glob.glob(str(out_dir / "*.json")):
        jpath = Path(jp)
        stem = jpath.stem
        originals = stem_to_originals.get(stem, [jpath.stem])
        # ייתכנו כמה המקורים עם אותו STEM – נסמן לכולם הצלחה עם אותו JSON
        for original_name in originals:
            results[original_name] = {
                "status": "success",
                "json_path": str(jpath),
            }

    # כשלונות (לעתים יש JSON ב-fail_output עם נתונים חלקיים)
    for jp in glob.glob(str(fail_dir / "*.json")):
        jpath = Path(jp)
        stem = jpath.stem
        originals = stem_to_originals.get(stem, [jpath.stem])
        for original_name in originals:
            # אל תדרוס הצלחה אם כבר סומנה
            if results.get(original_name, {}).get("status") == "success":
                continue
            results[original_name] = {
                "status": "failed",
                "json_path": str(jpath),
                "reason": "Missing critical fields",
            }

    # תמונות שלא הופיעו כלל בפלט – נסמן ככשלון כללי
    for img in original_images:
        name = Path(img).name
        if name not in results:
            results[name] = {
                "status": "failed",
                "json_path": "",
                "reason": "No JSON produced",
            }

    return results

def run_whitening(drone_type: str, log_path: Optional[str], image_paths: List[str]) -> Dict:
    """
    מריץ את הפייפליין ומחזיר מבנה תוצאות מפורט:
    {
      "zip_path": str,
      "workdir": str,
      "output_dir": str,
      "fail_output_dir": str,
      "results": {
          "<image_name>": {"status": "success"/"failed", "json_path": str, "reason"?: str}
      }
    }
    """
    folder_path = _stage_inputs(drone_type, log_path, image_paths)

    # ההלבנה עצמה:
    process_images_to_individual_json(folder_path)
    prepare_data_for_qgis(folder_path)

    # אריזה ל-ZIP מתוך TO_QGIS
    to_qgis_dir = os.path.join(folder_path, "TO_QGIS")
    os.makedirs(to_qgis_dir, exist_ok=True)
    zip_base = os.path.join(folder_path, "TO_QGIS")
    zip_path = shutil.make_archive(zip_base, "zip", to_qgis_dir)

    # איסוף תוצאות
    results = _collect_results(folder_path, image_paths)
    return {
        "zip_path": zip_path,
        "workdir": folder_path,
        "output_dir": str(Path(folder_path) / "output"),
        "fail_output_dir": str(Path(folder_path) / "fail_output"),
        "results": results,
    }
