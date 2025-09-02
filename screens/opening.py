import flet as ft

# תשלים את התמונה בעצמך — כאן רק שמירה על מקום
DRONE_IMG = "image/drone_bkrnd2.png"

def build_opening_screen(on_start):
    title = ft.Text("תקן פריים", size=88, weight=ft.FontWeight.BOLD, color="#eeeeee")

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

    left = ft.Column(
        [title, start_btn],
        spacing=44,
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.START,
    )

    right = ft.Container(
        content=ft.Image(src=DRONE_IMG, fit=ft.ImageFit.CONTAIN),
        expand=1, padding=20,
    )

    layout = ft.Row(
        [
            ft.Container(content=left, expand=1, padding=20),
            right
        ],
        expand=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    return layout
