import fasthtml.common as ft
import monsterui.all as mui
import os
import json

DATASET_DIR = os.path.join(os.path.dirname(__file__), "golden_dataset")

app, rt = mui.fast_app(hdrs=mui.Theme.blue.headers())

def list_traces():
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
    files.sort(reverse=True)
    items = []
    for fname in files:
        path = os.path.join(DATASET_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        msg = data["request"]["messages"][0]["content"]
        dt = fname.split('_')[1] + ' ' + fname.split('_')[2]
        items.append(
            ft.Li(ft.A(f"{dt}: {msg[:60]}...", href=annotate.to(fname=fname), cls=mui.AT.classic))
        )
    return ft.Ul(*items, cls=mui.ListT.bullet)

@rt
def index():
    return mui.Container(
        mui.H2("Golden Dataset Traces"),
        list_traces()
    )

def chat_bubble(m):
    is_user = m["role"] == "user"
    return ft.Div(
        ft.Div(
            mui.render_md(m["content"]),
            cls=f"chat-bubble {'chat-bubble-primary' if is_user else 'chat-bubble-secondary'}"
        ),
        cls=f"chat {'chat-end' if is_user else 'chat-start'}"
    )

@rt
def annotate(fname:str):
    path = os.path.join(DATASET_DIR, fname)
    with open(path) as f:
        data = json.load(f)
    chat =  data["response"]["messages"]
    bubbles = []
    bubbles = [chat_bubble(m) for m in chat]
    notes = data.get("open_coding", "")
    return mui.Container(
        mui.Grid(
            ft.Div(*bubbles),
            mui.Form(
                mui.TextArea(notes, name="notes", value=notes, rows=20),
                mui.Button("Save", type="submit"),
                action=save_annotation.to(fname=fname), method="post",
                cls='w-full flex flex-col gap-2'
            ),
        )
    )

@rt
def save_annotation(fname:str, notes:str):
    path = os.path.join(DATASET_DIR, fname)
    with open(path) as f:
        data = json.load(f)
    data["open_coding"] = notes
    with open(path, "w") as f:
        json.dump(data, f)
    return ft.Redirect(annotate.to(fname=fname))

@rt
def theme():
    return mui.ThemePicker()

ft.serve()






