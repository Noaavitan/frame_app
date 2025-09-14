# screens/opening.py
from __future__ import annotations
import flet as ft
from pathlib import Path
import sys

def _resource(rel_path: str) -> str:
    rel = Path(rel_path)
    bases = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        bases += [exe_dir, exe_dir / "_internal"]
    else:
        bases += [Path(__file__).resolve().parent.parent, Path.cwd()]

    for b in bases:
        p = (b / rel).resolve()
        if p.exists():
            return str(p)

    target_lower = rel.name.lower()
    for b in bases:
        folder = (b / rel.parent)
        if folder.exists():
            for f in folder.iterdir():
                if f.name.lower() == target_lower:
                    return str(f.resolve())
    return rel_path

DRONE_IMG = _resource("image/Drone.gif")
LOGO_IMG = _resource("image/logo.png")   # ← נוסיף את הלוגו


def build_opening_screen(on_start):
    # לוגו גדול יותר
    title = (
        ft.Image(src=LOGO_IMG, width=950, height=200, fit=ft.ImageFit.CONTAIN)
        if Path(LOGO_IMG).exists()
        else ft.Text("הלוגו logo.png לא נמצא", size=40, color="#ff8a80")
    )

    start_btn = ft.ElevatedButton(
        text="בואו נלבין",
        on_click=on_start,
        style=ft.ButtonStyle(
            bgcolor="#dddddd",
            color="#222222",
            padding=ft.Padding(44, 32, 44, 32),
            text_style=ft.TextStyle(size=26, weight=ft.FontWeight.BOLD),
            shape=ft.RoundedRectangleBorder(radius=12),
        ),
    )

    left = ft.Column(
        [title, start_btn],
        spacing=0,
        alignment=ft.MainAxisAlignment.CENTER,            # מרכז אנכית
        horizontal_alignment=ft.CrossAxisAlignment.CENTER # מרכז אופקית
    )

    # הרחפן קרוב יותר למרכז:
    right_content = (
        ft.Image(src=DRONE_IMG, width=600, height=600, fit=ft.ImageFit.CONTAIN)
        if Path(DRONE_IMG).exists()
        else ft.Text("התמונה Drone.gif לא נמצאה", color="#ff8a80")
    )

    # פחות Padding משמאל + יישור לשמאל, כדי להצמיד לגבול האמצעי
    right = ft.Container(
        content=right_content,
        padding=ft.Padding(150, 20, 20, 20),               # היה 40 מכל הצדדים
        alignment=ft.alignment.center_left                # קרוב לאמצע המסך
    )

    return ft.Row(
        [
            ft.Container(content=left, expand=6, padding=10),  # שמאל צר יותר
            ft.Container(content=right, expand=4),             # ימין רחב יותר
        ],
        expand=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

