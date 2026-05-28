from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from checkers_engine import GameState, Move, Player


@dataclass
class MinimaxConfig:
    depth: int = 4
    # For "not always same path": pick randomly among best moves within epsilon window.
    tie_break_random: bool = True
    tie_break_eps: float = 0.05  # relative window
    # Add tiny noise to evaluation to avoid deterministic repetition.
    eval_noise: float = 0.02


def choose_minimax_move(state: GameState, cfg: MinimaxConfig, rng: random.Random) -> Optional[Move]:
    moves = state.legal_moves()
    if not moves:
        return None

    perspective = state.turn
    scored: List[Tuple[float, Move]] = []
    for mv in moves:
        s2 = state.clone()
        s2.apply_move(mv)
        score = _minimax(s2, cfg.depth - 1, -math.inf, math.inf, perspective, rng, cfg)
        scored.append((score, mv))

    best = max(scored, key=lambda t: t[0])[0]
    if not cfg.tie_break_random:
        for sc, mv in scored:
            if sc == best:
                return mv
        return scored[0][1]

    # relative epsilon window; if best ~ 0, fall back to abs
    window = abs(best) * cfg.tie_break_eps
    if window < 1e-6:
        window = cfg.tie_break_eps
    candidates = [mv for sc, mv in scored if sc >= best - window]
    return rng.choice(candidates) if candidates else max(scored, key=lambda t: t[0])[1]


def _minimax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    perspective: Player,
    rng: random.Random,
    cfg: MinimaxConfig,
) -> float:
    winner = state.game_result()
    if winner is not None:
        return 1e6 if winner == perspective else -1e6
    if depth <= 0:
        return _eval(state, perspective, rng, cfg)

    moves = state.legal_moves()
    if not moves:
        return -1e6 if state.turn == perspective else 1e6

    maximizing = state.turn == perspective
    if maximizing:
        v = -math.inf
        for mv in moves:
            s2 = state.clone()
            s2.apply_move(mv)
            v = max(v, _minimax(s2, depth - 1, alpha, beta, perspective, rng, cfg))
            alpha = max(alpha, v)
            if beta <= alpha:
                break
        return v
    else:
        v = math.inf
        for mv in moves:
            s2 = state.clone()
            s2.apply_move(mv)
            v = min(v, _minimax(s2, depth - 1, alpha, beta, perspective, rng, cfg))
            beta = min(beta, v)
            if beta <= alpha:
                break
        return v


def _eval(state: GameState, perspective: Player, rng: random.Random, cfg: MinimaxConfig) -> float:
    base = float(state.material_score(perspective))
    # Prefer mobility a bit
    mobility = len(state.legal_moves())
    base += 0.05 * mobility if state.turn == perspective else -0.05 * mobility
    if cfg.eval_noise > 0:
        base += rng.uniform(-cfg.eval_noise, cfg.eval_noise)
    return base

