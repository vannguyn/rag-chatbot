import json
import os

HISTORY_PATH = "data/chat_history/history.json"


def load_history():

    if not os.path.exists(HISTORY_PATH):
        return []

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history):

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_message(role, content):

    history = load_history()

    history.append({
        "role": role,
        "content": content
    })

    save_history(history)