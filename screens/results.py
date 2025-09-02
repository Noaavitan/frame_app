# screens/results.py
from __future__ import annotations
import flet as ft
import os, sys, subprocess
from pathlib import Path
from typing import Dict

def _open_native(path: str, page: ft.Page, select: bool = False):
    try:
        if os.name == "nt":
            if select and Path(path).exists():
                subprocess.Popen(["explorer", "/select,", path])
            else:
                os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            if select:
                subprocess.Popen(["open", "-R", path])
            else:
                subprocess.Popen(["open", path])
        else:
            # select לא נתמך אמין בלינוקס – נפתח תיקייה או קובץ
            subprocess.Popen(["xdg-open", path if Path(path).exists() else str(Path(path).parent)])
    except Exception as e:
        page.snack_bar = ft.SnackBar(ft.Text(f"שגיאה בפתיחה: {e}"))
        page.snack_bar.open = True
        page.update()

def _status_chip(ok: bool):
    return ft.Container(
        padding=ft.padding.symmetric(horizontal=10, vertical=4),
        bgcolor="#0b3d0b" if ok else "#3d0b0b",
        border_radius=999,
        content=ft.Row(
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Icon(ft.Icons.CHECK_CIRCLE, size=16) if ok else ft.Icon(ft.Icons.ERROR, size=16),
                ft.Text("הצלחה" if ok else "כשלון", size=12, color="#e8e8e8")
            ],
        ),
    )

def build_results_screen(page: ft.Page, processing_result: Dict, on_again):
    """
    processing_result:
      {
        "zip_path": str,
        "workdir": str,
        "output_dir": str,
        "fail_output_dir": str,
        "results": { "<image_name>": {"status": "success"/"failed", "json_path": str, "reason"?: str} }
      }
    on_again: callback לחזרה למסך ההלבנה
    """
    header = ft.Text("תוצאות ההלבנה", size=32, weight=ft.FontWeight.BOLD, color="white")

    zip_path = processing_result.get("zip_path", "")
    workdir = processing_result.get("workdir", "")
    output_dir = processing_result.get("output_dir", "")
    fail_output_dir = processing_result.get("fail_output_dir", "")
    results: Dict = processing_result.get("results", {})

    list_tiles = []
    for image_name, info in results.items():
        ok = info.get("status") == "success"
        json_path = info.get("json_path") or ""
        reason = info.get("reason", "")
        list_tiles.append(
            ft.Card(
                content=ft.Container(
                    padding=12,
                    content=ft.Column(
                        spacing=8,
                        controls=[
                            ft.Row(
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                controls=[
                                    ft.Text(image_name, size=16, weight=ft.FontWeight.W_600, color="white"),
                                    _status_chip(ok),
                                ],
                            ),
                            ft.Row(
                                spacing=8,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    ft.Text("JSON:", size=12, color="#9aa0a6"),
                                    ft.Text(json_path or "לא קיים", size=12, color="#cfcfcf", selectable=True),
                                ],
                            ),
                            ft.Row(
                                spacing=8,
                                controls=[
                                    ft.TextButton("פתח JSON", disabled=not json_path, on_click=lambda _, p=json_path: _open_native(p, page, select=True)),
                                    ft.TextButton("פתח תיקיית JSON", disabled=not json_path, on_click=lambda _, p=json_path: _open_native(str(Path(p).parent), page)),
                                ],
                            ),
                            (ft.Text(f"סיבה: {reason}", size=12, color="#d18181") if (not ok and reason) else ft.Container()),
                        ],
                    ),
                )
            )
        )

    summary_bar = ft.Row(
        spacing=10,
        controls=[
            ft.TextButton("פתח ZIP", disabled=not zip_path, on_click=lambda _: _open_native(zip_path, page, select=True)),
            ft.TextButton("פתח תיקיית פלט", disabled=not output_dir, on_click=lambda _: _open_native(output_dir, page)),
            ft.TextButton("פתח תיקיית כשלונות", disabled=not fail_output_dir, on_click=lambda _: _open_native(fail_output_dir, page)),
            ft.TextButton("פתח תיקיית עבודה", disabled=not workdir, on_click=lambda _: _open_native(workdir, page)),
        ],
    )

    again_btn = ft.ElevatedButton(
        text="הלבנה נוספת",
        icon=ft.Icons.AUTORENEW,   # חץ מעגלי
        on_click=on_again,
        bgcolor="#374151",
        color="white",
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
    )

    body = ft.Column(
        spacing=16,
        width=820,
        controls=[
            header,
            ft.Container(padding=8, content=summary_bar),
            ft.Divider(opacity=0.1),
            ft.Text("סטטוס לפי תמונה:", size=14, color="#e0e0e0"),
            ft.ListView(controls=list_tiles, height=420, spacing=10, auto_scroll=False),
            ft.Row([again_btn], alignment=ft.MainAxisAlignment.CENTER),
        ],
    )

    card = ft.Card(content=ft.Container(padding=20, content=body))
    return ft.Container(expand=True, alignment=ft.alignment.center, content=card)
