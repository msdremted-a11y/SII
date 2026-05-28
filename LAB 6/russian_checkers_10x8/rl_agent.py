from __future__ import annotations

import pickle
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from checkers_engine import GameState, Move, Player
from minimax_ai import MinimaxConfig, choose_minimax_move


@dataclass
class QConfig:
    alpha: float = 0.15
    gamma: float = 0.98
    epsilon_start: float = 0.25
    epsilon_end: float = 0.05
    epsilon_decay_games: int = 15_000
    # encourage variety even after training:
    action_tie_random: bool = True
    action_tie_eps: float = 1e-6  # absolute window in Q


class QLearningAgent:
    def __init__(self, q_path: Path, cfg: QConfig, rng: random.Random) -> None:
        self.q_path = q_path
        self.cfg = cfg
        self.rng = rng
        self.q: Dict[Tuple[str, str], float] = {}

    def load(self) -> bool:
        if not self.q_path.exists():
            return False
        try:
            with self.q_path.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                self.q = obj
                return True
        except Exception:
            return False
        return False

    def save(self) -> None:
        self.q_path.parent.mkdir(parents=True, exist_ok=True)
        with self.q_path.open("wb") as f:
            pickle.dump(self.q, f)

    def choose_move(self, state: GameState, epsilon: float) -> Optional[Move]:
        moves = state.legal_moves()
        if not moves:
            return None
        if self.rng.random() < epsilon:
            return self.rng.choice(moves)

        s = state.encode_state()
        best = None
        best_v = float("-inf")
        scored: List[Tuple[float, Move]] = []
        for mv in moves:
            v = self.q.get((s, mv.encode()), 0.0)
            scored.append((v, mv))
            if v > best_v:
                best_v = v
                best = mv

        if best is None:
            return self.rng.choice(moves)

        if not self.cfg.action_tie_random:
            return best

        # pick randomly among near-best to avoid identical lines
        candidates = [mv for v, mv in scored if v >= best_v - self.cfg.action_tie_eps]
        return self.rng.choice(candidates) if candidates else best

    def update(self, s: str, a_key: str, reward: float, s_next: Optional[str], next_moves: Optional[List[Move]]) -> None:
        key = (s, a_key)
        q_sa = self.q.get(key, 0.0)
        if s_next is None or not next_moves:
            target = reward
        else:
            best_next = max(self.q.get((s_next, mv.encode()), 0.0) for mv in next_moves)
            target = reward + self.cfg.gamma * best_next
        self.q[key] = (1.0 - self.cfg.alpha) * q_sa + self.cfg.alpha * target


@dataclass
class TrainConfig:
    games: int = 20_000
    max_halfmoves: int = 220
    minimax_depth: int = 3
    # If True, RL plays as white always; else random side each game.
    fixed_side: bool = False
    progress_every: int = 25
    autosave_every: int = 0  # 0 = only at end
    # periodic evaluation vs minimax to decide early stop
    eval_every: int = 1000  # 0 = disabled
    eval_games: int = 200
    eval_depth: int = 3
    eval_epsilon: float = 0.0  # epsilon during evaluation (0=greedy)
    stop_on_target: bool = True
    target_win_rate: float = 0.60  # stop if eval win-rate >= this


@dataclass
class TrainProgress:
    game: int
    total_games: int
    rl_wins: int
    draws: int
    white_wins: int
    black_wins: int
    q_size: int
    elapsed_sec: float
    stopped: bool = False
    eval_win_rate: Optional[float] = None
    eval_depth: Optional[int] = None
    eval_games: Optional[int] = None


ProgressCallback = Callable[[TrainProgress], None]


def evaluate_vs_minimax(
    agent: QLearningAgent,
    *,
    games: int,
    depth: int,
    rng: random.Random,
    max_halfmoves: int = 220,
    fixed_side: bool = False,
    epsilon: float = 0.0,
) -> Dict[str, float]:
    mm_cfg = MinimaxConfig(depth=depth)
    rl_wins = 0
    draws = 0
    played = 0
    for g in range(games):
        st = GameState()
        rl_side = Player.WHITE if fixed_side else (Player.WHITE if rng.random() < 0.5 else Player.BLACK)
        halfmoves = 0
        while True:
            winner = st.game_result()
            if winner is not None:
                played += 1
                if winner == rl_side:
                    rl_wins += 1
                break
            if halfmoves >= max_halfmoves:
                played += 1
                draws += 1
                break
            if st.turn == rl_side:
                mv = agent.choose_move(st, epsilon=epsilon)
                if mv is None:
                    played += 1
                    break
                st.apply_move(mv)
            else:
                mv = choose_minimax_move(st, mm_cfg, rng)
                if mv is None:
                    played += 1
                    rl_wins += 1
                    break
                st.apply_move(mv)
            halfmoves += 1
    if played <= 0:
        played = 1
    return {
        "played": float(played),
        "rl_wins": float(rl_wins),
        "draws": float(draws),
        "win_rate": float(rl_wins) / float(played),
    }


def train_against_minimax(
    agent: QLearningAgent,
    train_cfg: TrainConfig,
    q_cfg: QConfig,
    rng: random.Random,
    on_progress: Optional[ProgressCallback] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, float]:
    mm_cfg = MinimaxConfig(depth=train_cfg.minimax_depth)
    wins = {Player.WHITE: 0, Player.BLACK: 0}
    rl_wins = 0
    draws = 0
    t0 = time.perf_counter()
    stopped = False
    games_played = 0

    last_eval_wr: Optional[float] = None

    def _report(g: int, force: bool = False) -> None:
        if on_progress is None:
            return
        if not force and train_cfg.progress_every > 0 and g % train_cfg.progress_every != 0 and g != train_cfg.games:
            return
        on_progress(
            TrainProgress(
                game=g,
                total_games=train_cfg.games,
                rl_wins=rl_wins,
                draws=draws,
                white_wins=wins[Player.WHITE],
                black_wins=wins[Player.BLACK],
                q_size=len(agent.q),
                elapsed_sec=time.perf_counter() - t0,
                stopped=stopped,
                eval_win_rate=last_eval_wr,
                eval_depth=train_cfg.eval_depth if train_cfg.eval_every else None,
                eval_games=train_cfg.eval_games if train_cfg.eval_every else None,
            )
        )

    for g in range(train_cfg.games):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        games_played = g + 1

        st = GameState()
        rl_side = Player.WHITE if train_cfg.fixed_side else (Player.WHITE if rng.random() < 0.5 else Player.BLACK)
        epsilon = _epsilon_for_game(g, q_cfg)

        # play one game
        last_state_action: Optional[Tuple[str, str]] = None
        halfmoves = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break

            winner = st.game_result()
            if winner is not None:
                # terminal reward
                if last_state_action is not None:
                    s, a = last_state_action
                    r = 1.0 if winner == rl_side else -1.0
                    agent.update(s, a, r, None, None)
                wins[winner] += 1
                if winner == rl_side:
                    rl_wins += 1
                break

            if halfmoves >= train_cfg.max_halfmoves:
                # draw-ish
                if last_state_action is not None:
                    s, a = last_state_action
                    agent.update(s, a, 0.0, None, None)
                draws += 1
                break

            if st.turn == rl_side:
                s = st.encode_state()
                mv = agent.choose_move(st, epsilon=epsilon)
                if mv is None:
                    # no moves => loss
                    if last_state_action is not None:
                        s0, a0 = last_state_action
                        agent.update(s0, a0, -1.0, None, None)
                    wins[st.turn.other()] += 1
                    break
                a_key = mv.encode()
                st2 = st.clone()
                st2.apply_move(mv)
                # small step reward: captures slightly positive
                step_reward = 0.02 * len(mv.captures)
                s_next = st2.encode_state()
                next_moves = st2.legal_moves() if st2.turn == rl_side else []
                agent.update(s, a_key, step_reward, s_next, next_moves)
                last_state_action = (s, a_key)
                st = st2
            else:
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break
                mv = choose_minimax_move(st, mm_cfg, rng)
                if mv is None:
                    # opponent no move => RL wins
                    if last_state_action is not None:
                        s0, a0 = last_state_action
                        agent.update(s0, a0, 1.0, None, None)
                    wins[st.turn.other()] += 1
                    rl_wins += 1
                    break
                st.apply_move(mv)

            halfmoves += 1

        if stopped:
            break

        if train_cfg.autosave_every > 0 and (g + 1) % train_cfg.autosave_every == 0:
            agent.save()

        # periodic evaluation and early-stop
        if train_cfg.eval_every and train_cfg.eval_every > 0 and (g + 1) % train_cfg.eval_every == 0:
            ev = evaluate_vs_minimax(
                agent,
                games=train_cfg.eval_games,
                depth=train_cfg.eval_depth,
                rng=rng,
                max_halfmoves=train_cfg.max_halfmoves,
                fixed_side=train_cfg.fixed_side,
                epsilon=train_cfg.eval_epsilon,
            )
            last_eval_wr = float(ev["win_rate"])
            if train_cfg.stop_on_target and last_eval_wr >= float(train_cfg.target_win_rate):
                stopped = True
                _report(g + 1, force=True)
                break

        _report(g + 1)

    _report(games_played if games_played else train_cfg.games, force=True)

    played = float(games_played) if games_played > 0 else 1.0

    return {
        "games": played,
        "win_rate": (wins[Player.WHITE] / played) if train_cfg.fixed_side else (rl_wins / played),
        "white_wins": float(wins[Player.WHITE]),
        "black_wins": float(wins[Player.BLACK]),
        "draws": float(draws),
        "q_size": float(len(agent.q)),
        "stopped": float(1 if stopped else 0),
        "elapsed_sec": time.perf_counter() - t0,
    }


def _epsilon_for_game(g: int, cfg: QConfig) -> float:
    if cfg.epsilon_decay_games <= 0:
        return cfg.epsilon_end
    t = min(1.0, max(0.0, g / float(cfg.epsilon_decay_games)))
    return cfg.epsilon_start + t * (cfg.epsilon_end - cfg.epsilon_start)

