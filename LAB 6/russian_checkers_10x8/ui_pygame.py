from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pygame

from checkers_engine import BOARD_H, BOARD_W, GameState, Move, Player, is_dark
from minimax_ai import MinimaxConfig, choose_minimax_move
from rl_agent import QConfig, QLearningAgent
from training_window import TrainingWindow


@dataclass
class UIConfig:
    cell: int = 70
    margin: int = 40
    fps: int = 60


class PygameApp:
    def __init__(self, q_path: Path) -> None:
        pygame.init()
        pygame.display.set_caption("Русские шашки 10x8 (pygame) + RL")

        self.cfg = UIConfig()
        self.w = self.cfg.margin * 2 + BOARD_W * self.cfg.cell
        self.h = self.cfg.margin * 2 + BOARD_H * self.cfg.cell + 90
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Segoe UI", 18)
        self.font_big = pygame.font.SysFont("Segoe UI", 26, bold=True)

        self.rng = random.Random()
        self.rng.seed(time.time_ns())

        self.q_cfg = QConfig()
        self.agent = QLearningAgent(q_path=q_path, cfg=self.q_cfg, rng=self.rng)
        self.q_loaded = self.agent.load()

        self.mm_depth = 4
        self.mode = "menu"  # menu | play | demo
        self.opponent = "rl"  # rl | minimax (used in play)
        self.demo_mm_depth = 3
        self.demo_rl_epsilon = 0.03
        self.demo_delay_ms = 250
        self._demo_next_move_at_ms = 0

        self.state = GameState()
        self.human_side = Player.WHITE
        self.selected: Optional[Tuple[int, int]] = None
        self.legal_from_selected: List[Move] = []
        self.last_msg: str = ""
        self._train_win: Optional[TrainingWindow] = None

    def run(self) -> None:
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    self._on_key(ev.key)
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self._on_click(ev.pos)

            if self._train_win is not None:
                self._train_win.pump_events()

            self._maybe_bot_move()
            self._draw()
            pygame.display.flip()
            self.clock.tick(self.cfg.fps)

        if self._train_win is not None:
            self._train_win.destroy()
        pygame.quit()

    def _reset_game(self) -> None:
        self.state = GameState()
        self.selected = None
        self.legal_from_selected = []
        self.last_msg = ""

    def _on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.mode = "menu"
            self.selected = None
            self.legal_from_selected = []
            return

        if self.mode == "menu":
            if key == pygame.K_1:
                self.opponent = "rl"
                self.human_side = Player.WHITE
                self.mode = "play"
                self._reset_game()
            elif key == pygame.K_2:
                self.opponent = "minimax"
                self.human_side = Player.WHITE
                self.mode = "play"
                self._reset_game()
            elif key == pygame.K_4:
                # demo: RL vs minimax
                self.mode = "demo"
                self._reset_game()
                self._demo_next_move_at_ms = pygame.time.get_ticks()
            elif key in (pygame.K_3, pygame.K_t):
                self._open_training_window()
            elif key in (pygame.K_q, pygame.K_s):
                self.agent.save()
                self.last_msg = f"Q-table сохранён: {len(self.agent.q)} записей"
            elif pygame.K_5 <= key <= pygame.K_9:
                self.mm_depth = int(pygame.key.name(key))
            return

        # play/demo mode
        if key in (pygame.K_r,):
            self._reset_game()
        elif self.mode == "demo" and key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_0):
            # demo speed control:
            # [  slower (more delay)
            # ]  faster (less delay)
            # 0  instant
            if key == pygame.K_0:
                self.demo_delay_ms = 0
            elif key == pygame.K_LEFTBRACKET:
                self.demo_delay_ms = min(2000, self.demo_delay_ms + 100)
            elif key == pygame.K_RIGHTBRACKET:
                self.demo_delay_ms = max(0, self.demo_delay_ms - 100)
        elif key in (pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9):
            if self.mode == "play":
                self.mm_depth = int(pygame.key.name(key))
            elif self.mode == "demo":
                self.demo_mm_depth = int(pygame.key.name(key))

    def _open_training_window(self) -> None:
        if self._train_win is None:
            self._train_win = TrainingWindow(
                agent=self.agent,
                q_cfg=self.q_cfg,
                rng=self.rng,
                on_depth_change=self._set_mm_depth,
                on_training_done=lambda msg: setattr(self, "last_msg", msg),
            )
            self._train_win.var_play_depth.set(self.mm_depth)
        else:
            self._train_win.var_play_depth.set(self.mm_depth)
        self._train_win.show()

    def _set_mm_depth(self, depth: int) -> None:
        self.mm_depth = max(1, min(9, depth))

    def _maybe_bot_move(self) -> None:
        if self.mode == "demo":
            self._maybe_demo_move()
            return
        if self.mode != "play":
            return
        if self.state.game_result() is not None:
            return
        if self.state.turn == self.human_side:
            return

        if self.opponent == "minimax":
            mv = choose_minimax_move(self.state, MinimaxConfig(depth=self.mm_depth), self.rng)
        else:
            # trained play: low epsilon for variety
            mv = self.agent.choose_move(self.state, epsilon=0.03)
        if mv is not None:
            self.state.apply_move(mv)
        self.selected = None
        self.legal_from_selected = []

    def _maybe_demo_move(self) -> None:
        if self.state.game_result() is not None:
            return
        now = pygame.time.get_ticks()
        if now < self._demo_next_move_at_ms:
            return

        # White = RL, Black = minimax (demo)
        if self.state.turn == Player.WHITE:
            mv = self.agent.choose_move(self.state, epsilon=self.demo_rl_epsilon)
        else:
            mv = choose_minimax_move(self.state, MinimaxConfig(depth=self.demo_mm_depth), self.rng)

        if mv is not None:
            self.state.apply_move(mv)

        delay = max(0, int(self.demo_delay_ms))
        self._demo_next_move_at_ms = now + delay

    def _on_click(self, pos: Tuple[int, int]) -> None:
        if self.mode != "play":
            return
        if self.state.game_result() is not None:
            return
        if self.state.turn != self.human_side:
            return

        bx, by = self._screen_to_board(pos)
        if bx is None:
            return

        if self.selected is None:
            p = self.state.piece_at(bx, by)
            if p is None or p.owner != self.human_side:
                return
            self.selected = (bx, by)
            self.legal_from_selected = [m for m in self.state.legal_moves() if m.start == (bx, by)]
            return

        # if clicking own piece -> reselect
        p = self.state.piece_at(bx, by)
        if p is not None and p.owner == self.human_side:
            self.selected = (bx, by)
            self.legal_from_selected = [m for m in self.state.legal_moves() if m.start == (bx, by)]
            return

        # attempt to make a move landing on clicked square
        candidates = [m for m in self.legal_from_selected if m.end == (bx, by)]
        if not candidates:
            return
        if len(candidates) == 1:
            mv = candidates[0]
        else:
            # if multiple (rare), pick the one with longer capture chain
            mv = max(candidates, key=lambda m: len(m.captures))
        self.state.apply_move(mv)
        self.selected = None
        self.legal_from_selected = []

    def _screen_to_board(self, pos: Tuple[int, int]) -> Tuple[Optional[int], Optional[int]]:
        x, y = pos
        ox = self.cfg.margin
        oy = self.cfg.margin
        if not (ox <= x < ox + BOARD_W * self.cfg.cell and oy <= y < oy + BOARD_H * self.cfg.cell):
            return (None, None)
        bx = (x - ox) // self.cfg.cell
        by = (y - oy) // self.cfg.cell
        if not is_dark(bx, by):
            return (None, None)
        return (int(bx), int(by))

    def _draw(self) -> None:
        self.screen.fill((22, 22, 24))
        self._draw_board()
        self._draw_hud()

    def _draw_board(self) -> None:
        ox = self.cfg.margin
        oy = self.cfg.margin
        c = self.cfg.cell
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                rect = pygame.Rect(ox + x * c, oy + y * c, c, c)
                if is_dark(x, y):
                    col = (88, 88, 95)
                else:
                    col = (200, 200, 205)
                pygame.draw.rect(self.screen, col, rect)

        # highlight selection
        if self.selected is not None:
            sx, sy = self.selected
            rect = pygame.Rect(ox + sx * c, oy + sy * c, c, c)
            pygame.draw.rect(self.screen, (255, 220, 90), rect, 4)

            # show legal destinations
            for mv in self.legal_from_selected:
                ex, ey = mv.end
                center = (ox + ex * c + c // 2, oy + ey * c + c // 2)
                pygame.draw.circle(self.screen, (70, 200, 120), center, 10)

        # pieces
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                p = self.state.piece_at(x, y)
                if p is None:
                    continue
                center = (ox + x * c + c // 2, oy + y * c + c // 2)
                base = (235, 235, 240) if p.owner == Player.WHITE else (35, 35, 40)
                edge = (0, 0, 0)
                pygame.draw.circle(self.screen, base, center, int(c * 0.36))
                pygame.draw.circle(self.screen, edge, center, int(c * 0.36), 2)
                if p.king:
                    pygame.draw.circle(self.screen, (200, 160, 40), center, int(c * 0.18), 3)

    def _draw_hud(self) -> None:
        pad_y = self.cfg.margin + BOARD_H * self.cfg.cell + 12
        if self.mode == "menu":
            title = self.font_big.render("Меню", True, (240, 240, 245))
            self.screen.blit(title, (self.cfg.margin, pad_y))
            lines = [
                "1 — Играть против RL бота (Q-learning)",
                "2 — Играть против Минимакса",
                "T / 3 — Окно обучения (настройки + прогресс)",
                "4 — ДЕМО: RL (белые) против Минимакса (чёрные)",
                "5..9 — глубина минимакса в игре (сейчас: %d)" % self.mm_depth,
                "S — сохранить Q-table сейчас",
                "ESC — в меню / выход из игры",
            ]
            y = pad_y + 34
            for ln in lines:
                surf = self.font.render(ln, True, (230, 230, 235))
                self.screen.blit(surf, (self.cfg.margin, y))
                y += 22

            status = "Q-table: загружен" if self.q_loaded else "Q-table: не найден (будет создан после обучения)"
            surf = self.font.render(status, True, (200, 200, 210))
            self.screen.blit(surf, (self.cfg.margin, y + 6))
        else:
            winner = self.state.game_result()
            if winner is None:
                turn_txt = "Ход: Белые" if self.state.turn == Player.WHITE else "Ход: Чёрные"
            else:
                turn_txt = "Победа: Белые" if winner == Player.WHITE else "Победа: Чёрные"
            if self.mode == "demo":
                info = (
                    f"{turn_txt} | ДЕМО: RL(белые) vs MM(чёрные) depth={self.demo_mm_depth} | "
                    f"скорость [ ] (0=быстро) | depth 4..9 | R — рестарт | ESC — меню"
                )
            else:
                info = f"{turn_txt} | Оппонент: {self.opponent} | depth={self.mm_depth} | R — рестарт | ESC — меню"
            surf = self.font.render(info, True, (235, 235, 240))
            self.screen.blit(surf, (self.cfg.margin, pad_y))

        if self.last_msg:
            msg = self.font.render(self.last_msg, True, (170, 210, 255))
            self.screen.blit(msg, (self.cfg.margin, pad_y + 60))

