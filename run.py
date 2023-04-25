from __future__ import annotations

import io
import os
import sys
import time
import enum
import termios
import select
import textwrap
import itertools
from typing import NamedTuple
from contextlib import contextmanager
import argparse


# Data structures


class Depth(enum.IntEnum):
    LOWER = 0
    UPPER = 1


class Direction(enum.Enum):
    NORTH = (-1, 0)
    EAST = (0, 1)
    SOUTH = (1, 0)
    WEST = (0, -1)

    @property
    def x(self) -> int:
        return self.value[0]

    @property
    def y(self) -> int:
        return self.value[1]

    @property
    def opposite(self) -> Direction:
        return OPPOSITES[self]


class Action(enum.IntEnum):
    STEP = 0
    INVERT = 1
    WAIT = 2
    CLEAR = 3
    DISPLAY = 4


OPPOSITES = {
    Direction.NORTH: Direction.SOUTH,
    Direction.EAST: Direction.WEST,
    Direction.SOUTH: Direction.NORTH,
    Direction.WEST: Direction.EAST,
}

PREFERRED_DIRECTIONS = [
    Direction.EAST,
    Direction.SOUTH,
    Direction.NORTH,
    Direction.WEST,
]


class Position(NamedTuple):
    x: int
    y: int

    def then(self, d: Direction):
        return Position(self.x + d.x, self.y + d.y)


class Marble(NamedTuple):
    p: Position
    z: Depth
    d: Direction

    @property
    def upper(self) -> bool:
        return self.z == Depth.UPPER

    @property
    def lower(self) -> bool:
        return self.z == Depth.LOWER


MARBLE_UPPER = "●"
MARBLE_LOWER = "○"
MARBLES = MARBLE_LOWER + MARBLE_UPPER

GRID_ON = "█"
GRID_OFF = "┼"
DISPLAY_ON = "▣"
DISPLAY_OFF = "□"
DISPLAYS = DISPLAY_ON + DISPLAY_OFF + GRID_ON + GRID_OFF

EXIT = "☒"

IO_0 = "◇"
IO_1 = "◆"
IOS = IO_0 + IO_1

TRACKS = {
    # 2-connector horizontal
    "═": {Direction.EAST, Direction.WEST},
    "━": {Direction.EAST, Direction.WEST},
    "╒": {Direction.EAST, Direction.WEST},
    "╕": {Direction.EAST, Direction.WEST},
    "╘": {Direction.EAST, Direction.WEST},
    "╛": {Direction.EAST, Direction.WEST},
    "╤": {Direction.EAST, Direction.WEST},
    "╧": {Direction.EAST, Direction.WEST},
    # 2-connector vertical
    "║": {Direction.NORTH, Direction.SOUTH},
    "┃": {Direction.NORTH, Direction.SOUTH},
    "╓": {Direction.NORTH, Direction.SOUTH},
    "╖": {Direction.NORTH, Direction.SOUTH},
    "╙": {Direction.NORTH, Direction.SOUTH},
    "╜": {Direction.NORTH, Direction.SOUTH},
    "╟": {Direction.NORTH, Direction.SOUTH},
    "╢": {Direction.NORTH, Direction.SOUTH},
    # 2-connector 90° turn
    "╚": {Direction.NORTH, Direction.EAST},
    "╔": {Direction.EAST, Direction.SOUTH},
    "╗": {Direction.SOUTH, Direction.WEST},
    "╝": {Direction.WEST, Direction.NORTH},
    # 4-connector
    "╬": {Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST},
}

INVERTORS = "━┃"
CONDITIONAL_CLEAR = {
    "╒": Direction.SOUTH,
    "╕": Direction.SOUTH,
    "╘": Direction.NORTH,
    "╛": Direction.NORTH,
    "╓": Direction.EAST,
    "╖": Direction.WEST,
    "╙": Direction.EAST,
    "╜": Direction.WEST,
}
CONTROL = {
    "╤": Direction.SOUTH,
    "╧": Direction.NORTH,
    "╟": Direction.EAST,
    "╢": Direction.WEST,
}

TERM_SIZE = os.terminal_size((0, 0))
OFFSET_CHANGED = True
X_OFFSET = 0
Y_OFFSET = 0

# Input/Output


class InputOutput:
    input_bits: list[bool] = []
    output_bits: list[bool] = []
    input_data: io.RawIOBase | None = None

    @classmethod
    def write(cls, value: bool) -> None:
        cls.output_bits.append(value)
        if len(cls.output_bits) != 8:
            return
        byte: int = 0
        for x in reversed(cls.output_bits):
            byte <<= 1
            byte += int(x)
        sys.stderr.buffer.write(bytes([byte]))
        sys.stderr.buffer.flush()
        cls.output_bits = []

    @classmethod
    def read(cls) -> bool | None:
        if not cls.input_bits:
            char = cls.get_char()
            if char is None:
                return None
            cls.input_bits = list(map(bool, map(int, format(ord(char), "08b"))))
        return cls.input_bits.pop()

    @classmethod
    def get_char(cls) -> bytes | None:
        if cls.input_data is not None:
            byte = cls.input_data.read(1)
            if not byte:
                return None
            return byte
        if sys.stdin.isatty():
            return None
        return get_char_from_stdin()


def get_char_from_stdin() -> bytes | None:
    fd = sys.stdin.fileno()
    read, _, _ = select.select([fd], [], [], 0)
    if fd not in read:
        return None
    char = sys.stdin.buffer.read(1)
    if not char:
        exit()
    return char


# Grid helpers


def get_character(grid: dict[int, dict[int, str]], p: Position) -> str:
    return grid.get(p.x, {}).get(p.y, " ")


def set_character(grid: dict[int, dict[int, str]], p: Position, value: str):
    grid[p.x][p.y] = value


def get_inversion(grid: dict[int, dict[int, str]], p: Position) -> bool:
    return get_character(grid, p) in INVERTORS


def get_directions(grid: dict[int, dict[int, str]], p: Position) -> set[Direction]:
    return TRACKS.get(get_character(grid, p), set())


def guess_directions(grid: dict[int, dict[int, str]], p: Position) -> set[Direction]:
    result = set(get_directions(grid, p))
    for d in Direction:
        if d.opposite in get_directions(grid, p.then(d)):
            result |= {d}
    return result


# Game logic


def display(
    grid: dict[int, dict[int, str]],
    marble: Marble,
) -> set[Position]:
    def _rec(p: Position) -> set[Position]:
        result = set()
        c = get_character(grid, p)
        if c == DISPLAY_ON and marble.lower:
            set_character(grid, p, DISPLAY_OFF)
        elif c == DISPLAY_OFF and marble.upper:
            set_character(grid, p, DISPLAY_ON)
        elif c == GRID_ON and marble.lower:
            set_character(grid, p, GRID_OFF)
        elif c == GRID_OFF and marble.upper:
            set_character(grid, p, GRID_ON)
        else:
            return result
        result |= {p}
        for d in Direction:
            result |= _rec(p.then(d))
        return result

    character = get_character(grid, marble.p)
    first = marble.p.then(CONTROL[character])
    return _rec(first)


def get_action(
    grid: dict[int, dict[int, str]],
    marble: Marble,
    lower_marbles: set[Position],
    upper_marbles: set[Position],
) -> Action:
    character = get_character(grid, marble.p)
    # Invertor block
    if character in INVERTORS:
        return Action.INVERT
    # Control block
    if character in CONTROL:
        d = CONTROL.get(character)
        assert d is not None
        neighbor = marble.p.then(d)
        neighbor_char = get_character(grid, neighbor)
        lower = neighbor in lower_marbles
        upper = neighbor in upper_marbles
        # To conditional clear
        if neighbor_char in CONDITIONAL_CLEAR:
            assert CONDITIONAL_CLEAR[neighbor_char] == d.opposite
            return Action.STEP if lower or upper else Action.WAIT
        # To display
        elif neighbor_char in DISPLAYS:
            return Action.DISPLAY
        # To exit
        elif neighbor_char in EXIT:
            if marble.upper:
                exit()
            return Action.STEP
        # To output
        elif neighbor_char in IOS:
            if marble.upper and neighbor_char == IO_0:
                InputOutput.write(False)
            elif marble.upper and neighbor_char == IO_1:
                InputOutput.write(True)
            return Action.STEP
        # Unrecognized pattern
        else:
            assert False, f"`{character}` + `{neighbor_char}` not supported"
    # Conditional-clear block
    if character in CONDITIONAL_CLEAR:
        d = CONDITIONAL_CLEAR.get(character)
        assert d is not None
        neighbor = marble.p.then(d)
        neighbor_char = get_character(grid, neighbor)
        lower = neighbor in lower_marbles
        upper = neighbor in upper_marbles
        # Connected to another synchro
        if neighbor_char in CONTROL:
            assert CONTROL[neighbor_char] == d.opposite
            return Action.STEP if upper else Action.CLEAR if lower else Action.WAIT
        # Connected to static marble
        elif neighbor_char == MARBLE_LOWER:
            return Action.CLEAR
        elif neighbor_char == MARBLE_UPPER:
            return Action.STEP
        # Connected to input
        elif neighbor_char in IOS:
            if marble.lower:
                return Action.STEP
            bit = InputOutput.read()
            return (
                Action.WAIT
                if bit is None
                else Action.CLEAR
                if bit == 0
                else Action.STEP
            )
        # Unrecognized pattern
        else:
            assert False, f"`{character}` + `{neighbor_char}` not supported"
    # Any other block
    return Action.STEP


def tick(
    grid: dict[int, dict[int, str]], marbles: set[Marble], waiting: dict[Position, Marble],
) -> tuple[set[Marble], set[Position]]:
    # Prepare structures
    changes = set()
    new_marbles = set()
    lower_marbles = {marble.p for marble in marbles if marble.lower}
    upper_marbles = {marble.p for marble in marbles if marble.upper}

    # Iterate over marbles
    for marble in marbles:

        # Get the possible directions
        directions = get_directions(grid, marble.p)

        # Case 0: Crash
        if len(directions) == 0:
            raise RuntimeError("Out of track")
        # Case 2: Follow
        elif len(directions) == 2:
            d1, d2 = directions
            if marble.d.opposite == d1:
                new_d = d2
            elif marble.d.opposite == d2:
                new_d = d1
            else:
                assert False
        # Case 4: Cross
        elif len(directions) == 4:
            new_d = marble.d
        else:
            assert False

        # Perform action
        action = get_action(grid, marble, lower_marbles, upper_marbles)
        if action == Action.STEP:
            new_z = marble.z
            new_p = marble.p.then(new_d)
            new_marbles |= {Marble(new_p, new_z, new_d)}
        elif action == Action.DISPLAY:
            changes |= display(grid, marble)
            new_z = marble.z
            new_p = marble.p.then(new_d)
            new_marbles |= {Marble(new_p, new_z, new_d)}
        elif action == Action.INVERT:
            new_z = Depth(not marble.z.value)
            new_p = marble.p.then(new_d)
            new_marbles |= {Marble(new_p, new_z, new_d)}
        elif action == Action.WAIT:
            new_z = marble.z
            new_p = marble.p
            waiting[new_p] = Marble(new_p, new_z, new_d)
        elif action == Action.CLEAR:
            new_z = Depth.LOWER
            new_p = marble.p.then(new_d)
            new_marbles |= {Marble(new_p, new_z, new_d)}
        else:
            assert False

        # Wake up neighbors
        for d in Direction:
            marble = waiting.pop(new_p.then(d), None)
            if marble is not None:
                new_marbles |= {marble}
                marble = waiting.pop(new_p, None)
                if marble is not None:
                    new_marbles |= {marble}

    return new_marbles, changes


# Drawing helpers


def draw_char(i: int, j: int, char: str):
    i += X_OFFSET
    j += Y_OFFSET
    if not 0 <= i < TERM_SIZE.lines:
        return ""
    if not 0 <= j < TERM_SIZE.columns:
        return ""
    return f"\033[{i+1};{j+1}H{char or ' '}"


def draw_grid(grid: dict[int, dict[int, str]]) -> str:
    result = "\x1b[2J"
    for i, row in grid.items():
        for j, char in row.items():
            result += draw_char(i, j, char)
    return result


def draw_changes(grid: dict[int, dict[int, str]], changes: set[Position]) -> str:
    result = ""
    for p in changes:
        char = get_character(grid, p)
        result += draw_char(p.x, p.y, char)
    return result


def draw_marbles(
    grid: dict[int, dict[int, str]],
    new_marbles: set[Marble],
    show_LOWER: bool = True,
):
    result = ""
    for marble in new_marbles:
        if marble.z == Depth.UPPER:
            result += draw_char(marble.p.x, marble.p.y, MARBLE_UPPER)
        elif show_LOWER:
            result += draw_char(marble.p.x, marble.p.y, MARBLE_LOWER)
    return result


@contextmanager
def drawing_context():
    import tty
    # Enable alternative buffer
    print("\033[?1049h", end="", flush=True)
    # Hide cursor
    print("\033[?25l", end="", flush=True)
    fd = sys.stdin.fileno()
    info = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        # Restore
        termios.tcsetattr(fd, termios.TCSAFLUSH, info)
        # Disable alternative buffer
        print("\033[?1049h", end="", flush=True)
        # Show cursor
        print("\033[?25h", end="", flush=True)
        # Line feed
        print()

@contextmanager
def debug_context():
    buffer = io.StringIO()
    yield buffer
    string = buffer.getvalue()
    lines = textwrap.wrap(
        string, width=TERM_SIZE.columns, initial_indent="┃", subsequent_indent="┃"
    )
    print(
        f"\033[{TERM_SIZE.lines - 10};{0}H" + "┏" + "━" * (TERM_SIZE.columns - 2) + "┓"
    )
    for line in lines:
        print(line.ljust(TERM_SIZE.columns - 1) + "┃")
    print("┗" + "━" * (TERM_SIZE.columns - 2) + "┛")


# Main routine


def main(file: io.TextIOBase, speed: float = 10.0, fps: float = 60.0, input_data: io.RawIOBase | None = None):
    InputOutput.input_data = input_data

    grid: dict[int, dict[int, str]] = {
        i: {j: char for j, char in enumerate(line) if char.strip()}
        for i, line in enumerate(file)
        if line.strip()
    }

    waiting: dict[Position, Marble] = {}
    changes: set[Position] = set()
    marbles: set[Marble] = set()
    old_marbles: set[Marble] = set()
    for i, row in grid.items():
        for j, char in row.items():
            p = Position(i, j)
            if char in MARBLES:
                depth = Depth.UPPER if char == MARBLE_UPPER else Depth.LOWER
                directions = guess_directions(grid, p)
                if len(directions) == 0:
                    continue
                elif len(directions) == 1:
                    assert False
                elif len(directions) == 2:
                    d1, d2 = sorted(directions, key=PREFERRED_DIRECTIONS.index)
                    marbles |= {Marble(p, depth, d2.opposite)}
                    for key, value in TRACKS.items():
                        if value == directions:
                            grid[i][j] = key
                            break
                elif len(directions) == 3:
                    assert False
                elif len(directions) == 4:
                    direction = PREFERRED_DIRECTIONS[0]
                    marbles |= {Marble(p, depth, direction)}
                    grid[i][j] = "╬"



    with drawing_context():
        global TERM_SIZE, OFFSET_CHANGED, X_OFFSET, Y_OFFSET
        deadline: float = time.time()
        deadline_display: float = time.time()

        # Iterator over ticks
        for t in itertools.count():

            # Control
            if sys.stdin.isatty():
                while True:
                    char = get_char_from_stdin()
                    if char == b"\x03":
                        raise KeyboardInterrupt
                    elif char == b"\x1b":
                        char = sys.stdin.buffer.read(1)
                        assert char == b"[", char
                        char = sys.stdin.buffer.read(1)
                        assert char in b"ABCD"
                        OFFSET_CHANGED = True
                        if char == b"A":
                            X_OFFSET += 1
                        elif char == b"B":
                            X_OFFSET -= 1
                        elif char == b"C":
                            Y_OFFSET -= 1
                        elif char == b"D":
                            Y_OFFSET += 1
                    elif char == b"v":
                        speed /= 1.5
                    elif char == b"b":
                        speed *= 1.5
                    elif char is not None:
                        pass
                    else:
                        break


            # Draw to the screen
            if time.time() > deadline_display:
                deadline_display += 1 / fps
                all_marbles = marbles | set(waiting.values())
                if os.get_terminal_size() != TERM_SIZE or OFFSET_CHANGED:
                    OFFSET_CHANGED = False
                    TERM_SIZE = os.get_terminal_size()
                    print(draw_grid(grid), end="", flush=True)
                else:
                    changes |= {marble.p for marble in old_marbles - all_marbles}
                    print(draw_changes(grid, changes), end="", flush=True)
                print(draw_marbles(grid, all_marbles), end="", flush=True)
                old_marbles = marbles
                changes = set()

            # Compute the next tick
            marbles, new_changes = tick(grid, marbles, waiting)
            changes |= new_changes

            # Wait for the next tick
            delta = deadline - time.time()
            if delta > 0:
                time.sleep(delta)
            deadline += 1 / speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=argparse.FileType())
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--input", type=argparse.FileType("rb"), default=None)
    namespace = parser.parse_args()
    main(namespace.file, namespace.speed, namespace.fps, namespace.input)
