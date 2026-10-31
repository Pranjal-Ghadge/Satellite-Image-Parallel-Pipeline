import threading
import time
import webbrowser

import uvicorn


def open_dashboard() -> None:
    time.sleep(1.0)
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    threading.Thread(target=open_dashboard, daemon=True).start()
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
