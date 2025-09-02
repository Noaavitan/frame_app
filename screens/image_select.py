# screens/image_select.py
from __future__ import annotations
import flet as ft
import asyncio, os, sys, pathlib
from typing import List, Set
from utils.pipeline import run_whitening
from screens.results import build_results_screen

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}

def _is_image(p: str) -> bool:
    return pathlib.Path(p).suffix.lower() in IMAGE_EXT

def _gather_images_in_dir(dir_path: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            full = os.path.join(root, f)
            if _is_image(full):
                paths.append(full)
    return paths

def build_image_select_screen(page: ft.Page):
    # --- State ---
    selected_drone = ft.Dropdown(
        options=[
            ft.dropdown.Option("DJI Mavic 350"),
            ft.dropdown.Option("DJI Padam"),
        ],
        label="סוג הרחפן",
        hint_text="בחרו את הדגם",
        autofocus=True,
        width=320,
    )

    log_file = {"path": None}
    selected_log_path = ft.Text("לא נבחר קובץ log", color="#9aa0a6", size=13)
    selected_files: Set[str] = set()

    files_counter = ft.Text("נבחרו 0 קבצי תמונה", size=14, color="#cccccc")

    # דיאלוג התקדמות
    progress_dlg = ft.AlertDialog(
        modal=True,
        content=ft.Column(
            [ft.ProgressRing(), ft.Text("מריץ הלבנה ואריזה ל-ZIP...", size=16)],
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=16,
        ),
    )

    # --- Pickers ---
    log_picker = ft.FilePicker(
        on_result=lambda e: (
            log_file.__setitem__("path", e.files[0].path if (e.files and e.files[0].path) else None),
            setattr(selected_log_path, "value",
                    e.files[0].path if (e.files and e.files[0].path) else "לא נבחר קובץ log"),
            setattr(selected_log_path, "color", "#9aa0a6"),
            page.update(),
        )
    )
    page.overlay.append(log_picker)

    imgs_picker = ft.FilePicker(
        on_result=lambda e: (
            selected_files.update([f.path for f in (e.files or []) if f.path and _is_image(f.path)]),
            refresh_files_ui(),
        )
    )
    page.overlay.append(imgs_picker)

    dir_picker = ft.FilePicker(
        on_result=lambda e: (
            selected_files.update(_gather_images_in_dir(e.path)) if (getattr(e, "path", None)) else None,
            refresh_files_ui(),
        )
    )
    page.overlay.append(dir_picker)

    # --- "ריבוע" המרכז – עכשיו מציג שמות קבצים (עם גלילה) ---
    placeholder_text = ft.Text(
        "גררו תמונות/תיקיות לכאן או השתמשו בכפתורים למעלה",
        color="#9aa0a6", size=12, text_align=ft.TextAlign.CENTER
    )

    drop_list = ft.ListView(height=180, spacing=4, auto_scroll=False)  # יתעדכן דינמית
    drop_area = ft.Container(
        height=200,
        bgcolor="#0f0f0f",
        border=ft.border.all(1, "#303030"),
        border_radius=10,
        alignment=ft.alignment.center,
        content=placeholder_text,  # כשיהיו קבצים – נחליף ל-drop_list
        padding=10,
    )

    def refresh_files_ui():
        count = len(selected_files)
        files_counter.value = f"נבחרו {count} קבצי תמונה"
        if count > 0:
            drop_list.controls = [
                ft.Text(pathlib.Path(p).name, size=12, color="#bdbdbd", tooltip=p)
                for p in sorted(selected_files)
            ]
            drop_area.content = drop_list
        else:
            drop_area.content = placeholder_text
        page.update()

    # --- Drag & Drop מכל האפליקציה (OS → האפליקציה) ---
    def handle_page_drop(e):
        added = 0
        for f in (e.files or []):
            if f.path:
                if os.path.isdir(f.path):
                    for p in _gather_images_in_dir(f.path):
                        if p not in selected_files:
                            selected_files.add(p); added += 1
                elif _is_image(f.path) and f.path not in selected_files:
                    selected_files.add(f.path); added += 1
        if added == 0:
            page.snack_bar = ft.SnackBar(ft.Text("לא נוספו קבצים (ודאו שמדובר בתמונות/תיקיות)"))
            page.snack_bar.open = True
        refresh_files_ui()

    try:
        page.on_drop = handle_page_drop  # עדיין מאפשר גרירה "בכללי"
    except Exception:
        pass  # בסביבה שלא תומכת – פשוט נשאר עם בחירה דרך הכפתורים

    # --- Controls ---
    pick_log_btn = ft.TextButton(
        "בחרו קובץ לוג…",
        on_click=lambda _: log_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["log", "txt", "csv"],
        ),
    )

    def on_no_log_toggle(e):
        if no_log_cb.value:
            pick_log_btn.disabled = True
            selected_log_path.value = "לא נדרש קובץ log"
            selected_log_path.color = "#7fd37f"
            log_file["path"] = None
        else:
            pick_log_btn.disabled = False
            selected_log_path.value = "לא נבחר קובץ log"
            selected_log_path.color = "#9aa0a6"
        page.update()

    no_log_cb = ft.Checkbox(label="אין קובץ לוג (דלג)", value=False, on_change=on_no_log_toggle)

    add_folder_btn = ft.FilledButton("בחרו תיקייה…", on_click=lambda _: dir_picker.get_directory_path())
    add_files_btn  = ft.OutlinedButton("בחרו תמונות…", on_click=lambda _: imgs_picker.pick_files(allow_multiple=True, file_type=ft.FilePickerFileType.IMAGE))
    clear_btn      = ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="נקה בחירה", on_click=lambda _: (selected_files.clear(), refresh_files_ui()))

    async def on_submit_clicked(e):
        # ולידציות
        if not selected_drone.value:
            page.snack_bar = ft.SnackBar(ft.Text("בחרו סוג רחפן")); page.snack_bar.open = True; page.update(); return
        if len(selected_files) == 0:
            page.snack_bar = ft.SnackBar(ft.Text("לא נבחרו תמונות")); page.snack_bar.open = True; page.update(); return
        if (not no_log_cb.value) and (not log_file["path"]):
            page.snack_bar = ft.SnackBar(ft.Text("בחרו קובץ log או סמנו 'אין קובץ לוג'")); page.snack_bar.open = True; page.update(); return

        # דיאלוג התקדמות
        page.dialog = progress_dlg
        progress_dlg.open = True
        page.update()

        try:
            proc = await asyncio.to_thread(
                run_whitening,
                selected_drone.value,
                None if no_log_cb.value else log_file["path"],
                list(selected_files)
            )
        except Exception as err:
            progress_dlg.open = False; page.update()
            page.snack_bar = ft.SnackBar(ft.Text(f"שגיאה בעיבוד: {err}")); page.snack_bar.open = True; page.update()
            return

        progress_dlg.open = False
        page.update()

        # ניווט למסך תוצאות
        def back_to_select(_):
            page.controls.clear()
            page.add(build_image_select_screen(page))
            page.update()

        page.controls.clear()
        page.add(build_results_screen(page, proc, on_again=back_to_select))
        page.update()

    submit_btn = ft.ElevatedButton(
        "שלח להלבנה",
        bgcolor="#3b82f6",
        color="white",
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
        on_click=on_submit_clicked,
    )

    # --- Layout ---
    header = ft.Text("בחירת התמונות", size=32, weight=ft.FontWeight.BOLD, color="white")

    left_col = ft.Column(
        [
            header,
            selected_drone,

            ft.Row([pick_log_btn, no_log_cb], spacing=12, alignment=ft.MainAxisAlignment.START),
            ft.Text("קובץ log נבחר:", size=12, color="#9aa0a6"),
            selected_log_path,

            ft.Divider(opacity=0.1),

            ft.Text("הוספת תמונות", size=14, weight=ft.FontWeight.W_600, color="#e0e0e0"),
            ft.Row([add_folder_btn, add_files_btn, clear_btn], spacing=10),

            # מונה + הריבוע שמציג את השמות
            ft.Row([files_counter], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            drop_area,

            ft.Container(height=6),
            ft.Row([submit_btn], alignment=ft.MainAxisAlignment.CENTER),
        ],
        spacing=12,
        width=640,
        horizontal_alignment=ft.CrossAxisAlignment.START,
    )

    card = ft.Card(content=ft.Container(padding=24, content=left_col))
    return ft.Container(expand=True, alignment=ft.alignment.center, content=card)
