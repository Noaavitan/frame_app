# screens/opening.py
from __future__ import annotations
import flet as ft
from pathlib import Path
import sys

def _resource(rel_path: str) -> str:
    """
    מאתר קובץ משאב גם בריצה רגילה וגם אחרי אריזה.
    מחפש ב:
    - תיקיית ה-EXE
    - תיקיית _internal (כשארוז)
    - תיקיית הפרויקט
    - תיקיית העבודה הנוכחית
    כולל חיפוש לא-תלוי-רישיות בתיקיית היעד.
    """
    rel = Path(rel_path)

    bases = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        bases += [exe_dir, exe_dir / "_internal"]
    else:
        bases += [Path(__file__).resolve().parent.parent, Path.cwd()]

    # חיפוש ישיר
    for b in bases:
        p = (b / rel).resolve()
        if p.exists():
            return str(p)

    # חיפוש לא-תלוי-רישיות בתוך התיקייה היעד
    target_lower = rel.name.lower()
    for b in bases:
        folder = (b / rel.parent)
        if folder.exists():
            for f in folder.iterdir():
                if f.name.lower() == target_lower:
                    return str(f.resolve())

    # fallback
    return rel_path

DRONE_IMG = _resource("image/Drone.gif")

def build_opening_screen(on_start):
    title = ft.Text("תקן פריים", size=88, weight=ft.FontWeight.BOLD, color="#eeeeee", text_align=ft.TextAlign.CENTER)

    start_btn = ft.ElevatedButton(
        text="בואו נלבין",
        on_click=on_start,
        style=ft.ButtonStyle(
            bgcolor="#dddddd",
            color="#222222",
            padding=ft.Padding(36, 26, 36, 26),
            text_style=ft.TextStyle(size=22, weight=ft.FontWeight.BOLD),
            shape=ft.RoundedRectangleBorder(radius=10),
        ),
    )

    left = ft.Column([title, start_btn], spacing=44, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    right_content = (
        ft.Image(src=DRONE_IMG, width=500, height=500, fit=ft.ImageFit.CONTAIN)
        if Path(DRONE_IMG).exists()
        else ft.Text("התמונה Drone.gif לא נמצאה", color="#ff8a80")
    )

    right = ft.Container(content=right_content, padding=40)

    return ft.Row(
        [ft.Container(content=left, expand=1, padding=20), right],
        expand=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
