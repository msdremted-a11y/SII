from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BOARD_W = 10
BOARD_H = 8


class Player(int, Enum):
    WHITE = 1
    BLACK = -1

    def other(self) -> "Player":
        return Player.WHITE if self == Player.BLACK else Player.BLACK


@dataclass(frozen=True)
class Piece:
    owner: Player
    king: bool = False


@dataclass(frozen=True)
class Move:
    path: Tuple[Tuple[int, int], ...]  # includes start and all landings
    captures: Tuple[Tuple[int, int], ...]  # captured squares in order

    @property
    def start(self) -> Tuple[int, int]:
        return self.path[0]

    @property
    def end(self) -> Tuple[int, int]:
        return self.path[-1]

    @property
    def is_capture(self) -> bool:
        return len(self.captures) > 0

    def encode(self) -> str:
        # stable action key for Q-table
        p = ";".join(f"{x},{y}" for x, y in self.path)
        c = ";".join(f"{x},{y}" for x, y in self.captures)
        return f"P:{p}|C:{c}"


def is_dark(x: int, y: int) -> bool:
    return (x + y) % 2 == 1


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < BOARD_W and 0 <= y < BOARD_H


DIAGS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


class GameState:
    def __init__(self) -> None:
        self.board: List[List[Optional[Piece]]] = [
            [None for _ in range(BOARD_W)] for _ in range(BOARD_H)
        ]
        self.turn: Player = Player.WHITE
        self.halfmove_clock: int = 0
        self._setup_initial()

    def clone(self) -> "GameState":
        g = GameState.__new__(GameState)
        g.board = [row[:] for row in self.board]
        g.turn = self.turn
        g.halfmove_clock = self.halfmove_clock
        return g

    def _setup_initial(self) -> None:
        # Place pieces on dark squares: 3 rows each side.
        # White at bottom (higher y), moves "up" (dy=-1).
        for y in range(0, 3):
            for x in range(BOARD_W):
                if is_dark(x, y):
                    self.board[y][x] = Piece(Player.BLACK, king=False)
        for y in range(BOARD_H - 3, BOARD_H):
            for x in range(BOARD_W):
                if is_dark(x, y):
                    self.board[y][x] = Piece(Player.WHITE, king=False)

    def piece_at(self, x: int, y: int) -> Optional[Piece]:
        return self.board[y][x]

    def set_piece(self, x: int, y: int, p: Optional[Piece]) -> None:
        self.board[y][x] = p

    def iter_pieces(self, owner: Optional[Player] = None) -> Iterable[Tuple[int, int, Piece]]:
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                p = self.board[y][x]
                if p is None:
                    continue
                if owner is None or p.owner == owner:
                    yield x, y, p

    def _promote_if_needed(self, x: int, y: int) -> None:
        p = self.piece_at(x, y)
        if p is None or p.king:
            return
        if p.owner == Player.WHITE and y == 0:
            self.set_piece(x, y, Piece(p.owner, king=True))
        elif p.owner == Player.BLACK and y == BOARD_H - 1:
            self.set_piece(x, y, Piece(p.owner, king=True))

    def apply_move(self, mv: Move) -> None:
        sx, sy = mv.start
        ex, ey = mv.end
        p = self.piece_at(sx, sy)
        if p is None:
            raise ValueError("No piece on move start")
        self.set_piece(sx, sy, None)
        self.set_piece(ex, ey, p)
        for cx, cy in mv.captures:
            self.set_piece(cx, cy, None)

        self._promote_if_needed(ex, ey)
        self.turn = self.turn.other()
        self.halfmove_clock = 0 if mv.is_capture else (self.halfmove_clock + 1)

    def game_result(self) -> Optional[Player]:
        # Returns winner, or None if ongoing. No-draw logic: treat long no-capture as draw -> None
        w = any(True for _ in self.iter_pieces(Player.WHITE))
        b = any(True for _ in self.iter_pieces(Player.BLACK))
        if not w:
            return Player.BLACK
        if not b:
            return Player.WHITE
        if len(self.legal_moves()) == 0:
            return self.turn.other()
        return None

    def legal_moves(self) -> List[Move]:
        caps = self._all_captures(self.turn)
        if caps:
            # Russian checkers: capturing is mandatory. Also: choose moves with maximum captures.
            max_c = max(len(m.captures) for m in caps)
            caps = [m for m in caps if len(m.captures) == max_c]
            return caps
        return self._all_quiet_moves(self.turn)

    def _all_quiet_moves(self, pl: Player) -> List[Move]:
        res: List[Move] = []
        for x, y, p in self.iter_pieces(pl):
            if p.king:
                res.extend(self._king_quiet_moves(x, y))
            else:
                res.extend(self._man_quiet_moves(x, y, pl))
        return res

    def _man_quiet_moves(self, x: int, y: int, pl: Player) -> List[Move]:
        dy = -1 if pl == Player.WHITE else 1
        res: List[Move] = []
        for dx in (-1, 1):
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny) and is_dark(nx, ny) and self.piece_at(nx, ny) is None:
                res.append(Move(path=((x, y), (nx, ny)), captures=()))
        return res

    def _king_quiet_moves(self, x: int, y: int) -> List[Move]:
        res: List[Move] = []
        for dx, dy in DIAGS:
            nx, ny = x + dx, y + dy
            while in_bounds(nx, ny) and is_dark(nx, ny) and self.piece_at(nx, ny) is None:
                res.append(Move(path=((x, y), (nx, ny)), captures=()))
                nx += dx
                ny += dy
        return res

    def _all_captures(self, pl: Player) -> List[Move]:
        res: List[Move] = []
        for x, y, p in self.iter_pieces(pl):
            if p.king:
                res.extend(self._king_capture_sequences_from(x, y))
            else:
                res.extend(self._man_capture_sequences_from(x, y))
        return res

    def _would_promote(self, owner: Player, y: int) -> bool:
        return (owner == Player.WHITE and y == 0) or (owner == Player.BLACK and y == BOARD_H - 1)

    def _man_capture_sequences_from(self, x: int, y: int) -> List[Move]:
        p = self.piece_at(x, y)
        if p is None:
            return []

        sequences: List[Move] = []

        def dfs(cx: int, cy: int, board: List[List[Optional[Piece]]], path: List[Tuple[int, int]], caps: List[Tuple[int, int]]) -> None:
            found = False
            for dx, dy in DIAGS:
                mx, my = cx + dx, cy + dy
                lx, ly = cx + 2 * dx, cy + 2 * dy
                if not (in_bounds(lx, ly) and in_bounds(mx, my)):
                    continue
                if not (is_dark(mx, my) and is_dark(lx, ly)):
                    continue
                mid = board[my][mx]
                if mid is None or mid.owner == p.owner:
                    continue
                if board[ly][lx] is not None:
                    continue

                found = True
                # simulate
                b2 = [r[:] for r in board]
                b2[cy][cx] = None
                b2[my][mx] = None
                became_king = self._would_promote(p.owner, ly)
                b2[ly][lx] = Piece(p.owner, king=became_king)
                if became_king:
                    # Russian checkers: if a man reaches the last rank during a capture,
                    # it becomes a king immediately and must continue capturing as a king.
                    self._king_capture_dfs_from(
                        owner=p.owner,
                        cx=lx,
                        cy=ly,
                        board=b2,
                        path=path + [(lx, ly)],
                        caps=caps + [(mx, my)],
                        out=sequences,
                    )
                else:
                    dfs(lx, ly, b2, path + [(lx, ly)], caps + [(mx, my)])

            if not found and caps:
                sequences.append(Move(path=tuple(path), captures=tuple(caps)))

        dfs(x, y, [r[:] for r in self.board], [(x, y)], [])
        return sequences

    def _king_capture_sequences_from(self, x: int, y: int) -> List[Move]:
        p = self.piece_at(x, y)
        if p is None:
            return []

        sequences: List[Move] = []
        self._king_capture_dfs_from(
            owner=p.owner,
            cx=x,
            cy=y,
            board=[r[:] for r in self.board],
            path=[(x, y)],
            caps=[],
            out=sequences,
        )
        return sequences

    def _king_capture_dfs_from(
        self,
        owner: Player,
        cx: int,
        cy: int,
        board: List[List[Optional[Piece]]],
        path: List[Tuple[int, int]],
        caps: List[Tuple[int, int]],
        out: List[Move],
    ) -> None:
        found_any = False
        for dx, dy in DIAGS:
            # scan for first opponent piece, then land beyond it
            tx, ty = cx + dx, cy + dy
            seen_enemy: Optional[Tuple[int, int]] = None
            while in_bounds(tx, ty) and is_dark(tx, ty):
                cur = board[ty][tx]
                if cur is None:
                    if seen_enemy is not None:
                        found_any = True
                        ex, ey = seen_enemy
                        b2 = [r[:] for r in board]
                        b2[cy][cx] = None
                        b2[ey][ex] = None
                        b2[ty][tx] = Piece(owner, king=True)
                        self._king_capture_dfs_from(
                            owner=owner,
                            cx=tx,
                            cy=ty,
                            board=b2,
                            path=path + [(tx, ty)],
                            caps=caps + [(ex, ey)],
                            out=out,
                        )
                    tx += dx
                    ty += dy
                    continue

                if cur.owner == owner:
                    break
                if seen_enemy is not None:
                    break
                seen_enemy = (tx, ty)
                tx += dx
                ty += dy

        if not found_any and caps:
            out.append(Move(path=tuple(path), captures=tuple(caps)))

    def encode_state(self) -> str:
        # Compact hashable state representation for Q-table.
        # Encoding only dark squares in row-major.
        # '.' empty, 'w' white man, 'W' white king, 'b' black man, 'B' black king
        chars: List[str] = []
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                if not is_dark(x, y):
                    continue
                p = self.board[y][x]
                if p is None:
                    chars.append(".")
                else:
                    if p.owner == Player.WHITE:
                        chars.append("W" if p.king else "w")
                    else:
                        chars.append("B" if p.king else "b")
        chars.append("T" if self.turn == Player.WHITE else "t")
        return "".join(chars)

    def material_score(self, perspective: Player) -> int:
        # simple eval for minimax
        score = 0
        for _, _, p in self.iter_pieces():
            v = 3 if p.king else 1
            score += v if p.owner == perspective else -v
        return score

