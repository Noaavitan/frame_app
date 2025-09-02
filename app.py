# app.py
import flet as ft
import asyncio
from pathlib import Path
from screens.opening import build_opening_screen
from screens.image_select import build_image_select_screen
from utils.exiftool_setup import ensure_exiftool_on_path

def main(page: ft.Page):
    page.title = "תקן פריים"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#000000"
    page.horizontal_alignment = "center"
    page.vertical_alignment = "center"
    page.rtl = True

    # דיאלוג קצר בזמן הכנה
    setup_dlg = ft.AlertDialog(
        modal=True,
        content=ft.Column(
            controls=[ft.ProgressRing(), ft.Text("מכין סביבת עבודה (ExifTool)...", size=16)],
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=12,
        ),
    )

    async def go_to_select(e):
        # ריצה פעם-ראשונה (מהירה אם כבר מוגדר)
        page.dialog = setup_dlg
        setup_dlg.open = True
        page.update()

        try:
            base_dir = Path(__file__).resolve().parent  # תיקיית הפרויקט שלך (frame_app)
            ok, msg = await asyncio.to_thread(ensure_exiftool_on_path, base_dir)
        except Exception as err:
            ok, msg = False, f"שגיאה בהכנת ExifTool: {err}"

        setup_dlg.open = False
        page.update()

        # משוב קצר (לא חוסם מעבר)
        page.snack_bar = ft.SnackBar(ft.Text(msg))
        page.snack_bar.open = True
        page.update()

        # מעבר למסך בחירת התמונות
        page.controls.clear()
        page.add(build_image_select_screen(page))
        page.update()

    page.add(build_opening_screen(on_start=go_to_select))

if __name__ == "__main__":
    ft.app(target=main)
