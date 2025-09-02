# utils/exiftool_setup.py
from __future__ import annotations
import os, sys
from pathlib import Path
from shutil import which

# Windows-only registry helpers
if os.name == "nt":
    import winreg
    import ctypes

SENTINEL_NAME = ".exiftool_path_set"

def _find_exiftool_dir(base_dir: Path) -> Path | None:
    """
    מחפש את התיקייה שמכילה exiftool.exe יחסית לתיקיית הפרויקט (base_dir).
    """
    # מועמדים נפוצים בתוך הפרויקט
    candidates = [
        base_dir / "exiftool.exe",
        base_dir / "exiftool-13.30_64" / "exiftool.exe",
        base_dir / "exiftool-13.32_64" / "exiftool.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c.parent

    # חיפוש exiftool*/exiftool.exe עד עומק 3
    for exe in base_dir.rglob("exiftool.exe"):
        try:
            depth = len(exe.parent.relative_to(base_dir).parts)
        except ValueError:
            depth = 99
        if depth <= 3 and exe.is_file():
            return exe.parent
    return None

def _get_user_path() -> str:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_READ) as key:
            val, _ = winreg.QueryValueEx(key, "Path")
            return val
    except Exception:
        return ""

def _set_user_path(new_val: str) -> None:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_val)

def _broadcast_env_change():
    HWND_BROADCAST = 0xFFFF
    WM_SETTINGCHANGE = 0x001A
    SMTO_ABORTIFHUNG = 0x0002
    res = ctypes.c_long()
    ctypes.windll.user32.SendMessageTimeoutW(
        HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment",
        SMTO_ABORTIFHUNG, 5000, ctypes.byref(res)
    )

def ensure_exiftool_on_path(base_dir: Path) -> tuple[bool, str]:
    """
    דואג ש-exiftool יהיה ב-PATH של המשתמש.
    מחזיר (success, message).
    ריצה חוזרת לא תבצע שינוי אם כבר עודכן.
    """
    # לא Windows? אין מה לעשות כאן.
    if os.name != "nt":
        return True, "Non-Windows environment – skipping PATH setup."

    sentinel = base_dir / SENTINEL_NAME

    # כבר על PATH?
    if which("exiftool"):
        # ליצור סנטינל כדי לא לבדוק שוב בכל לחיצה
        try:
            sentinel.write_text("ok", encoding="utf-8")
        except Exception:
            pass
        return True, "ExifTool already on PATH."

    # אם כבר הרצנו בעבר (סנטינל), ננסה שוב לבדוק PATH; אם עדיין לא — נמשיך לאיתור מקומי
    # (משאירים את הסנטינל כ"ניסיון" בלבד, לא תנאי עצירה).
    exif_dir = _find_exiftool_dir(base_dir)
    if not exif_dir:
        return False, "לא נמצא exiftool.exe בתיקיית הפרויקט. ודא שהוא קיים."

    # עדכון PATH של המשתמש
    current = _get_user_path()
    parts = [p for p in (current.split(";") if current else []) if p.strip()]
    # בדיקת קיום לוגית (normalize to lower + /)
    target_norm = exif_dir.resolve().as_posix().lower()
    already = False
    for p in parts:
        try:
            if Path(p).exists():
                if Path(p).resolve().as_posix().lower() == target_norm:
                    already = True
                    break
        except Exception:
            if p.replace("\\", "/").lower() == target_norm:
                already = True
                break

    if not already:
        parts.append(str(exif_dir))
        new_val = ";".join(parts)
        _set_user_path(new_val)
        _broadcast_env_change()

    try:
        sentinel.write_text("ok", encoding="utf-8")
    except Exception:
        pass

    # בדיקה מסכמת: which יעבוד רק בתהליכים חדשים; פה נחזיר הצלחה עם הסבר
    return True, f"נוסף ל-PATH של המשתמש: {exif_dir}"
