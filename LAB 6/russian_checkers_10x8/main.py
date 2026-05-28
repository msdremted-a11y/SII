from __future__ import annotations

from pathlib import Path

from ui_pygame import PygameApp


def main() -> None:
    q_path = Path(__file__).with_name("qtable.pkl")
    app = PygameApp(q_path=q_path)
    app.run()


if __name__ == "__main__":
    main()

