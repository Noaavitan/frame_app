# screens/opening.py
from __future__ import annotations
import flet as ft
from pathlib import Path
import sys, os

def _resource(rel_path: str) -> str:
    rel = Path(rel_path)

    if hasattr(sys, "_MEIPASS"):  #  exe
        base = Path(sys._MEIPASS)
        return str(base / rel)

    #  כשמריצים ישירות עם python
    return str(Path(__file__).resolve().parent.parent / rel)

    target_lower = rel.name.lower()
    for b in bases:
        folder = (b / rel.parent)
        if folder.exists():
            for f in folder.iterdir():
                if f.name.lower() == target_lower:
                    return str(f.resolve())
    return rel_path

DRONE_IMG = _resource("image/Drone.gif")
LOGO_IMG = _resource("image/logo.png")   #  הלוגו


def build_opening_screen(on_start):

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
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )


    right_content = (
        ft.Image(src=DRONE_IMG, width=600, height=600, fit=ft.ImageFit.CONTAIN)
        if Path(DRONE_IMG).exists()
        else ft.Text("התמונה Drone.gif לא נמצאה", color="#ff8a80")
    )


    right = ft.Container(
        content=right_content,
        padding=ft.Padding(150, 20, 20, 20),
        alignment=ft.alignment.center_left
    )

    return ft.Row(
        [
            ft.Container(content=left, expand=6, padding=10),
            ft.Container(content=right, expand=4),
        ],
        expand=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

