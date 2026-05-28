from __future__ import annotations

import queue
import random
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Optional, Tuple

from rl_agent import QConfig, QLearningAgent, TrainConfig, TrainProgress, train_against_minimax


class TrainingWindow:
    """Отдельное окно Tkinter: настройки обучения и прогресс."""

    def __init__(
        self,
        agent: QLearningAgent,
        q_cfg: QConfig,
        rng: random.Random,
        on_depth_change: Optional[Callable[[int], None]] = None,
        on_training_done: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.agent = agent
        self.q_cfg = q_cfg
        self.rng = rng
        self.on_depth_change = on_depth_change
        self.on_training_done = on_training_done

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ui_events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Обучение RL — настройки")
        self.root.geometry("480x620")
        self.root.minsize(440, 560)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._refresh_q_status()

    def show(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def pump_events(self) -> None:
        """Вызывать из цикла pygame, чтобы окно Tk не зависало."""
        self._drain_ui_events()
        try:
            self.root.update()
        except tk.TclError:
            pass

    def _drain_ui_events(self) -> None:
        # Все обновления Tk делаем строго из главного потока (pygame loop).
        while True:
            try:
                kind, payload = self._ui_events.get_nowait()
            except queue.Empty:
                return
            if kind == "progress":
                self._update_progress_ui(payload)
            elif kind == "finished":
                msg, stats = payload
                self._training_finished(msg, stats)

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Параметры обучения", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        self.var_games = tk.IntVar(value=10_000)
        self.var_depth = tk.IntVar(value=3)
        self.var_max_half = tk.IntVar(value=220)
        self.var_alpha = tk.DoubleVar(value=self.q_cfg.alpha)
        self.var_gamma = tk.DoubleVar(value=self.q_cfg.gamma)
        self.var_eps_start = tk.DoubleVar(value=self.q_cfg.epsilon_start)
        self.var_eps_end = tk.DoubleVar(value=self.q_cfg.epsilon_end)
        self.var_eps_decay = tk.IntVar(value=self.q_cfg.epsilon_decay_games)
        self.var_fixed_white = tk.BooleanVar(value=False)
        self.var_autosave = tk.IntVar(value=2000)
        self.var_play_depth = tk.IntVar(value=4)
        self.var_eval_every = tk.IntVar(value=1000)
        self.var_eval_games = tk.IntVar(value=200)
        self.var_eval_depth = tk.IntVar(value=3)
        self.var_target_wr = tk.DoubleVar(value=0.60)
        self.var_stop_on_target = tk.BooleanVar(value=True)

        row = 1
        row = self._row_spin(frm, row, "Число игр", self.var_games, 100, 200_000, 500)
        row = self._row_spin(frm, row, "Глубина минимакса (соперник)", self.var_depth, 1, 8, 1)
        row = self._row_spin(frm, row, "Макс. полуходов в партии", self.var_max_half, 50, 400, 10)

        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(frm, text="Q-learning").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        row = self._row_float(frm, row, "Alpha (скорость обучения)", self.var_alpha, 0.01, 1.0, 0.01)
        row = self._row_float(frm, row, "Gamma (дисконт)", self.var_gamma, 0.5, 0.999, 0.01)
        row = self._row_float(frm, row, "Epsilon (старт)", self.var_eps_start, 0.0, 1.0, 0.01)
        row = self._row_float(frm, row, "Epsilon (конец)", self.var_eps_end, 0.0, 1.0, 0.01)
        row = self._row_spin(frm, row, "Затухание epsilon (игр)", self.var_eps_decay, 100, 100_000, 500)

        ttk.Checkbutton(frm, text="RL всегда играет белыми", variable=self.var_fixed_white).grid(
            row=row, column=0, columnspan=2, sticky="w", **pad
        )
        row += 1

        row = self._row_spin(frm, row, "Автосохранение каждые N игр (0=выкл)", self.var_autosave, 0, 50_000, 500)

        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(frm, text="Контроль (оценка силы)").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        row = self._row_spin(frm, row, "Оценка каждые N игр (0=выкл)", self.var_eval_every, 0, 50_000, 250)
        row = self._row_spin(frm, row, "Контрольных игр", self.var_eval_games, 20, 5000, 50)
        row = self._row_spin(frm, row, "Глубина минимакса на контроле", self.var_eval_depth, 1, 6, 1)
        row = self._row_float(frm, row, "Цель win-rate (0..1)", self.var_target_wr, 0.0, 1.0, 0.02)
        ttk.Checkbutton(frm, text="Остановить, когда цель достигнута", variable=self.var_stop_on_target).grid(
            row=row, column=0, columnspan=2, sticky="w", **pad
        )
        row += 1

        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(frm, text="Игра (минимакс)").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        row = self._row_spin(frm, row, "Глубина минимакса в игре", self.var_play_depth, 1, 9, 1)

        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        ttk.Label(frm, text="Прогресс").grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.progress = ttk.Progressbar(frm, mode="determinate", maximum=100)
        self.progress.grid(row=row, column=0, columnspan=2, sticky="ew", **pad)
        row += 1

        self.lbl_status = ttk.Label(frm, text="Ожидание запуска…")
        self.lbl_status.grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        self.lbl_stats = ttk.Label(frm, text="", justify=tk.LEFT)
        self.lbl_stats.grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        self.lbl_q = ttk.Label(frm, text="")
        self.lbl_q.grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=row, column=0, columnspan=2, sticky="ew", pady=12)
        row += 1

        self.btn_start = ttk.Button(btn_frm, text="▶ Начать обучение", command=self._start_training)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop = ttk.Button(btn_frm, text="■ Остановить", command=self._stop_training, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(btn_frm, text="Сохранить Q-table", command=self._save_q).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frm, text="Загрузить Q-table", command=self._load_q).pack(side=tk.LEFT)

        frm.columnconfigure(1, weight=1)

    def _row_spin(
        self, parent: ttk.Frame, row: int, label: str, var: tk.Variable, frm: int, to: int, step: int
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=4)
        sp = ttk.Spinbox(parent, from_=frm, to=to, increment=step, textvariable=var, width=12)
        sp.grid(row=row, column=1, sticky="e", padx=10, pady=4)
        return row + 1

    def _row_float(
        self, parent: ttk.Frame, row: int, label: str, var: tk.DoubleVar, frm: float, to: float, step: float
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=4)
        sp = ttk.Spinbox(parent, from_=frm, to=to, increment=step, textvariable=var, width=12, format="%.2f")
        sp.grid(row=row, column=1, sticky="e", padx=10, pady=4)
        return row + 1

    def _apply_q_cfg(self) -> None:
        self.q_cfg.alpha = float(self.var_alpha.get())
        self.q_cfg.gamma = float(self.var_gamma.get())
        self.q_cfg.epsilon_start = float(self.var_eps_start.get())
        self.q_cfg.epsilon_end = float(self.var_eps_end.get())
        self.q_cfg.epsilon_decay_games = int(self.var_eps_decay.get())

    def _build_train_config(self) -> TrainConfig:
        return TrainConfig(
            games=int(self.var_games.get()),
            max_halfmoves=int(self.var_max_half.get()),
            minimax_depth=int(self.var_depth.get()),
            fixed_side=bool(self.var_fixed_white.get()),
            progress_every=25,
            autosave_every=int(self.var_autosave.get()),
            eval_every=int(self.var_eval_every.get()),
            eval_games=int(self.var_eval_games.get()),
            eval_depth=int(self.var_eval_depth.get()),
            stop_on_target=bool(self.var_stop_on_target.get()),
            target_win_rate=float(self.var_target_wr.get()),
        )

    def _start_training(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._apply_q_cfg()
        if self.on_depth_change is not None:
            self.on_depth_change(int(self.var_play_depth.get()))

        games = int(self.var_games.get())
        if games < 1:
            messagebox.showerror("Ошибка", "Число игр должно быть ≥ 1")
            return

        self._stop_event.clear()
        self.progress["maximum"] = games
        self.progress["value"] = 0
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_stop.config(text="■ Остановить")
        self.lbl_status.config(text="Обучение запущено…")

        cfg = self._build_train_config()

        def worker() -> None:
            try:
                stats = train_against_minimax(
                    self.agent,
                    cfg,
                    self.q_cfg,
                    self.rng,
                    on_progress=self._on_progress_thread,
                    stop_event=self._stop_event,
                )
                self.agent.save()
                msg = (
                    f"Готово: игр={int(stats['games'])}, win_rate={stats['win_rate']:.2%}, "
                    f"ничьи={int(stats['draws'])}, Q={int(stats['q_size'])}, "
                    f"время={stats['elapsed_sec']:.1f}с"
                )
                if stats.get("stopped", 0) >= 1:
                    msg = "Остановлено. " + msg
            except Exception as e:
                msg = f"Ошибка: {e}"
                stats = None
            self._ui_events.put(("finished", (msg, stats)))

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def _on_progress_thread(self, p: TrainProgress) -> None:
        self._ui_events.put(("progress", p))

    def _update_progress_ui(self, p: TrainProgress) -> None:
        self.progress["value"] = p.game
        pct = 100.0 * p.game / max(1, p.total_games)
        wr = p.rl_wins / max(1, p.game)
        ev = ""
        if p.eval_win_rate is not None and p.eval_depth is not None and p.eval_games is not None:
            ev = f" | eval d{p.eval_depth}: {p.eval_win_rate:.1%} ({p.eval_games} игр)"
        self.lbl_status.config(
            text=f"Игра {p.game} / {p.total_games}  ({pct:.1f}%)  |  {p.elapsed_sec:.0f} с{ev}"
        )
        self.lbl_stats.config(
            text=(
                f"Win-rate RL: {wr:.1%}\n"
                f"Победы: белые {p.white_wins} | чёрные {p.black_wins} | ничьи {p.draws}\n"
                f"Размер Q-table: {p.q_size}"
            )
        )
        self._refresh_q_status()

    def _training_finished(self, msg: str, stats: Optional[dict]) -> None:
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_stop.config(text="■ Остановить")
        self.lbl_status.config(text=msg)
        self._refresh_q_status()
        if self.on_training_done is not None:
            self.on_training_done(msg)

    def _stop_training(self) -> None:
        self._stop_event.set()
        self.lbl_status.config(text="Останавливаем после текущей партии…")
        # Не блокируем кнопку: остановка может занять время, пока доигрывается партия.
        self.btn_stop.config(text="■ Останавливаем…")

    def _save_q(self) -> None:
        self.agent.save()
        self._refresh_q_status()
        messagebox.showinfo("Сохранено", f"Q-table: {len(self.agent.q)} записей")

    def _load_q(self) -> None:
        ok = self.agent.load()
        self._refresh_q_status()
        if ok:
            messagebox.showinfo("Загружено", f"Q-table: {len(self.agent.q)} записей")
        else:
            messagebox.showwarning("Не найдено", "Файл qtable.pkl не найден или повреждён")

    def _refresh_q_status(self) -> None:
        self.lbl_q.config(text=f"Q-table: {len(self.agent.q)} записей | файл: {self.agent.q_path.name}")

    def _on_close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            if messagebox.askyesno("Обучение", "Остановить обучение и закрыть окно?"):
                self._stop_event.set()
            else:
                return
        self.root.withdraw()

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass
