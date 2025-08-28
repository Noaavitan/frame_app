import flet as ft
import asyncio, os, shutil, sys, subprocess
from pathlib import Path
from image_to_json_generator import process_images_to_individual_json, prepare_data_for_qgis

# תמונת רחפן במסך הפתיחה (אם תריצי בדפדפן, הריצי עם assets_dir="assets")
DRONE_IMG = "image/drone_bkrnd2.png"

# אייקון העתקה – תואם גם לגרסאות בלי ft.icons
ICON_COPY = getattr(getattr(ft, "Icons", object), "CONTENT_COPY", None)


def run_pipeline(folder_path: str) -> str:
    """מריץ את הפייפליין שלך ומחזיר נתיב ל-ZIP שנוצר."""
    process_images_to_individual_json(folder_path)
    prepare_data_for_qgis(folder_path)
    to_qgis_dir = os.path.join(folder_path, "TO_QGIS")
    os.makedirs(to_qgis_dir, exist_ok=True)
    zip_base = os.path.join(folder_path, "TO_QGIS")
    zip_path = shutil.make_archive(zip_base, "zip", to_qgis_dir)
    return zip_path


def open_path_native(path: str, page: ft.Page):
    """פותח קובץ/תיקייה מקומית בצורה נייטיבית לפי מערכת ההפעלה."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        page.snack_bar = ft.SnackBar(ft.Text(f"שגיאה בפתיחה: {e}"))
        page.snack_bar.open = True
        page.update()


def copy_to_clipboard(text: str, page: ft.Page):
    page.set_clipboard(text)
    page.snack_bar = ft.SnackBar(ft.Text("הנתיב הועתק ללוח"))
    page.snack_bar.open = True
    page.update()


def build_loader(message="טוען...", subtext="זה עשוי לקחת רגע"):
    # מסך טעינה ממורכז לחלוטין
    return ft.Container(
        expand=True,
        alignment=ft.alignment.center,
        content=ft.Column(
            controls=[
                ft.ProgressRing(),
                ft.Text(message, size=22, weight=ft.FontWeight.W_500, color="white"),
                ft.Text(subtext, size=12, color="#9aa0a6"),
            ],
            spacing=16,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


def build_opening_screen(on_click):
    start_btn = ft.ElevatedButton(
        text="בואו נלבין",
        on_click=on_click,
        style=ft.ButtonStyle(
            bgcolor="#dddddd",
            color="#222222",
            padding=ft.Padding(35, 25, 35, 25),
            text_style=ft.TextStyle(size=22, weight=ft.FontWeight.BOLD),
            shape=ft.RoundedRectangleBorder(radius=8),
        ),
    )

    left_column = ft.Column(
        [
            ft.Text("Frame App", size=90, weight=ft.FontWeight.BOLD, color="#eeeeee"),
            start_btn,
        ],
        spacing=50,
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    right_image = ft.Image(src=DRONE_IMG, fit=ft.ImageFit.CONTAIN)

    return ft.Row(
        [
            ft.Container(content=left_column, expand=4, padding=20),
            ft.Container(content=right_image, expand=3, padding=20),
        ],
        expand=True,
    )


def build_image_input_screen(page: ft.Page):
    """מסך בחירת קובץ → טען תמונה — ממורכז, עם תצוגה מקדימה קטנה ותוצאות עוטפות."""
    selected_path = {"value": None}
    PREVIEW_W, PREVIEW_H = 360, 240

    title = ft.Text(
        "טעינת תמונה",
        size=32,
        weight=ft.FontWeight.BOLD,
        color="white",
        text_align=ft.TextAlign.CENTER,
    )
    help_text = ft.Text(
        "בחרו תמונה להלבנה",
        color="#9aa0a6",
        text_align=ft.TextAlign.CENTER,
    )

    selected_label = ft.Text("לא נבחרה תמונה", color="#cccccc")

    preview = ft.Container(
        width=PREVIEW_W,
        height=PREVIEW_H,
        bgcolor="#111111",
        border_radius=8,
        alignment=ft.alignment.center,
        content=ft.Text("תצוגה מקדימה תופיע כאן", color="#666"),
    )

    # טקסט תוצאה שעוטף נתיב ארוך
    result_text = ft.Text(
        "",
        color="#cccccc",
        size=14,
        no_wrap=False,
        max_lines=2,
        overflow=ft.TextOverflow.ELLIPSIS,
        selectable=True,
        text_align=ft.TextAlign.CENTER,
    )

    # כפתורי פעולה (ZIP/תיקייה/העתקה)
    result_buttons = ft.Row([], alignment=ft.MainAxisAlignment.CENTER, spacing=16)

    progress_dlg = ft.AlertDialog(
        modal=True,
        content=ft.Column(
            [ft.ProgressRing(), ft.Text("מריץ עיבוד תמונות ואריזה ל-ZIP...", size=16)],
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=16,
        ),
    )

    def on_file_picked(e: ft.FilePickerResultEvent):
        if e.files and e.files[0].path:
            p = e.files[0].path
            selected_path["value"] = p
            selected_label.value = f"נבחרה תמונה: {Path(p).name}"
            preview.content = ft.Image(
                src=p, fit=ft.ImageFit.CONTAIN, width=PREVIEW_W, height=PREVIEW_H
            )
        else:
            selected_label.value = "לא נבחרה תמונה / בדפדפן אין גישה לנתיב מקומי."
            preview.content = ft.Text("תצוגה מקדימה תופיע כאן", color="#666")
        page.update()

    file_picker = ft.FilePicker(on_result=on_file_picked)
    page.overlay.append(file_picker)

    pick_button = ft.FilledButton(
        "בחר/י תמונה...",
        on_click=lambda e: file_picker.pick_files(
            allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE
        ),
    )

    async def on_load_clicked(e):
        if not selected_path["value"]:
            page.snack_bar = ft.SnackBar(ft.Text("לא נבחר קובץ"))
            page.snack_bar.open = True
            page.update()
            return

        folder_path = os.path.dirname(selected_path["value"])

        page.dialog = progress_dlg
        progress_dlg.open = True
        page.update()

        try:
            zip_path = await asyncio.to_thread(run_pipeline, folder_path)
        except Exception as err:
            progress_dlg.open = False
            page.update()
            page.snack_bar = ft.SnackBar(ft.Text(f"שגיאה בעיבוד: {err}"))
            page.snack_bar.open = True
            page.update()
            return

        progress_dlg.open = False
        page.update()

        # מציגים נתיב (יעטף), וכפתורים לפתיחה נייטיבית
        result_text.value = f"נוצר ZIP: {zip_path}"

        # כפתור העתקה לפי זמינות אייקון
        copy_button = (
            ft.IconButton(
                icon=ICON_COPY,
                tooltip="העתיקי נתיב ZIP",
                on_click=lambda _: copy_to_clipboard(zip_path, page),
            )
            if ICON_COPY
            else ft.TextButton(
                "העתיקי נתיב", on_click=lambda _: copy_to_clipboard(zip_path, page)
            )
        )

        result_buttons.controls = [
            ft.TextButton("פתחי ZIP", on_click=lambda _: open_path_native(zip_path, page)),
            ft.TextButton(
                "פתחי תיקייה", on_click=lambda _: open_path_native(folder_path, page)
            ),
            copy_button,
        ]
        page.update()

    load_button = ft.ElevatedButton(text="טען תמונה (הרץ עיבוד)", on_click=on_load_clicked)

    # תוכן המסך — בתוך Card ממורכז וצר יותר
    card_content = ft.Column(
        controls=[
            title,
            help_text,
            ft.Row([pick_button, selected_label], alignment=ft.MainAxisAlignment.CENTER),
            ft.Divider(opacity=0.1),
            ft.Row([load_button], alignment=ft.MainAxisAlignment.CENTER),
            ft.Container(height=10),
            preview,
            ft.Container(height=10),
            ft.Container(width=520, content=result_text, alignment=ft.alignment.center),
            ft.Container(height=4),
            result_buttons,
        ],
        spacing=16,
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    card = ft.Card(
        content=ft.Container(
            width=560,  # רוחב הכרטיס
            padding=24,
            content=card_content,
        )
    )

    # כל המסך ממורכז באמצע הדף
    return ft.Container(expand=True, alignment=ft.alignment.center, content=card)


def main(page: ft.Page):
    page.title = "Frame App"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#000000"

    async def go_loading_then_input(e):
        # מרכוז הטעינה
        page.horizontal_alignment = "center"
        page.vertical_alignment = "center"

        page.controls.clear()
        page.add(build_loader(message="מכין את עמוד טעינת התמונה...", subtext="רגע אחד..."))
        page.update()
        await asyncio.sleep(0.5)

        # מסך הקלט – גם כן ממורכז
        page.controls.clear()
        page.add(build_image_input_screen(page))
        page.update()

    page.add(build_opening_screen(on_click=go_loading_then_input))


if __name__ == "__main__":
    # להרצה בדפדפן (אם צריך להציג את תמונת הרחפן): בטלי את השורה למטה והשתמשי בשורה המודגמת
    # ft.app(target=main, view=ft.AppView.WEB_BROWSER, assets_dir="assets")
    ft.app(target=main)
