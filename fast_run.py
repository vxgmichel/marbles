from __future__ import annotations
import bisect
import functools

import io
import os
import sys
import time
import enum
import typing
import termios
import select
import heapq
import itertools
from typing import NamedTuple
from contextlib import contextmanager
import argparse


# Data structures


class Depth(enum.IntEnum):
    LOWER = 0
    UPPER = 1

    @property
    def inverse(self):
        return Depth(not self.value)


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


# Characters

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

# Logic helpers


def get_character(grid: list[dict[int, str]], p: Position) -> str:
    if not 0 <= p.x < len(grid):
        return " "
    return grid[p.x].get(p.y, " ")


def set_character(grid: list[dict[int, str]], p: Position, value: str):
    grid[p.x][p.y] = value


def get_inversion(grid: list[dict[int, str]], p: Position) -> bool:
    return get_character(grid, p) in INVERTORS


def get_directions(grid: list[dict[int, str]], p: Position) -> set[Direction]:
    return TRACKS.get(get_character(grid, p), set())


def guess_directions(grid: list[dict[int, str]], p: Position) -> set[Direction]:
    result = set(get_directions(grid, p))
    for d in Direction:
        if d.opposite in get_directions(grid, p.then(d)):
            result |= {d}
    return result


# Game logic


class Event:
    def __init__(self, index: int, inverted: bool, neighbor: Position | None = None):
        self.index = index
        self.inverted = inverted
        self.neighbor = neighbor

    def process(self, simulation: Simulation, circuit: Circuit) -> bool:
        if self.inverted:
            circuit.depth = circuit.depth.inverse
        return False


class Start(Event):
    def process(self, simulation: Simulation, circuit: Circuit) -> bool:
        super().process(simulation, circuit)
        if circuit.last_start is None:
            circuit.last_start = 0
            return False
        cycle = simulation.tick - circuit.last_start
        circuit.last_start = simulation.tick
        simulation.register_cycle(circuit, cycle)
        return False


class ConditionalClear(Event):
    def process(self, simulation: Simulation, circuit: Circuit) -> bool:
        super().process(simulation, circuit)
        assert self.neighbor is not None
        other_circuit = simulation.pop_waiting_circuit(self.neighbor)
        if other_circuit is None:
            return True
        if other_circuit.depth == Depth.LOWER and circuit.depth == Depth.UPPER:
            circuit.depth = Depth.LOWER
        other_circuit.waiting = False
        other_circuit.last_tick = simulation.tick
        simulation.add_running_circuit(other_circuit)
        return False


class UnconditionalClear(Event):
    def process(self, simulation: Simulation, circuit: Circuit) -> bool:
        circuit.depth = Depth.LOWER
        return False


class Control(Event):
    def process(self, simulation: Simulation, circuit: Circuit) -> bool:
        super().process(simulation, circuit)
        assert self.neighbor is not None
        other_circuit = simulation.pop_waiting_circuit(self.neighbor)
        if other_circuit is None:
            return True
        if other_circuit.depth == Depth.UPPER and circuit.depth == Depth.LOWER:
            other_circuit.depth = Depth.LOWER
        other_circuit.waiting = False
        other_circuit.last_tick = simulation.tick
        simulation.add_running_circuit(other_circuit)
        return False


class Display(Event):
    def process(self, simulation: Simulation, circuit: Circuit) -> bool:
        super().process(simulation, circuit)
        assert self.neighbor is not None
        circuit.set_display_value(self.neighbor, circuit.depth)
        return False


class Exit(Event):
    def process(self, simulation: Simulation, circuit: Circuit) -> bool:
        super().process(simulation, circuit)
        if circuit.depth == Depth.UPPER:
            exit()
        return False


class ReadBit(Event):
    def process(self, simulation: Simulation, circuit: Circuit) -> bool:
        super().process(simulation, circuit)
        if circuit.depth == Depth.UPPER:
            if not simulation.read_bit():
                circuit.depth = Depth.LOWER
        return False


class WriteZero(Event):
    def process(self, simulation: Simulation, circuit: Circuit) -> bool:
        super().process(simulation, circuit)
        if circuit.depth == Depth.UPPER:
            simulation.write_bit(0)
        return False


class WriteOne(Event):
    def process(self, simulation: Simulation, circuit: Circuit) -> bool:
        super().process(simulation, circuit)
        if circuit.depth == Depth.UPPER:
            simulation.write_bit(1)
        return False


@functools.total_ordering
class DisplayInfo:
    def __init__(
        self, positions: set[Position], char_off: str, char_on: str, initial_value: bool
    ):
        self.char_on = char_on
        self.char_off = char_off
        self.value = initial_value
        self.on_chars = {(p, char_on) for p in positions}
        self.off_chars = {(p, char_off) for p in positions}

        # X positioning
        self.min_x = min(p.x for p in positions)
        self.max_x = max(p.x for p in positions)
        self.span = self.max_x - self.min_x + 1
        self.max_span = 2 ** (self.span - 1).bit_length()

    def __lt__(self, other):
        if isinstance(other, DisplayInfo):
            return self.min_x.__lt__(other.min_x)
        if isinstance(other, int):
            return self.min_x.__lt__(other)
        raise NotImplementedError(other)

    def get_display_chars(self) -> set[tuple[Position, str]]:
        return self.on_chars if self.value else self.off_chars


@functools.total_ordering
class Circuit:
    def __init__(
        self,
        marble: Marble,
        positions: list[Position],
        events: list[Event],
        invertors: list[int],
        displays: dict[Position, DisplayInfo],
    ):
        self.init_marble = marble
        self.positions = positions
        self.events = events
        self.length = len(self.positions)
        self.displays = displays
        self.invertors = invertors

        # X positioning
        self.min_x = min(p.x for p in self.positions)
        self.max_x = max(p.x for p in positions)
        self.span = self.max_x - self.min_x + 1
        self.max_span = 2 ** (self.span - 1).bit_length()

        # Mutable state
        self.waiting: bool = False
        self.depth: Depth = marble.z
        self.last_index: int = 0
        self.last_tick: int = 0
        self.last_start: int | None = None

    def __lt__(self, other):
        if isinstance(other, Circuit):
            return self.min_x.__lt__(other.min_x)
        if isinstance(other, int):
            return self.min_x.__lt__(other)
        raise NotImplementedError(other)

    def set_display_value(self, position: Position, value: Depth) -> None:
        self.displays[position].value = bool(value.value)

    def get_marble(self, tick: int) -> Marble:
        if self.waiting:
            position = self.last_position
            depth = self.depth
        else:
            p_index = self.events[self.last_index].index
            i_start = bisect.bisect_left(self.invertors, p_index)
            p_index += tick - self.last_tick
            i_stop = bisect.bisect_left(self.invertors, p_index)
            p_index %= self.length
            position = self.positions[p_index]
            invert = (i_stop - i_start) % 2
            depth = self.depth.inverse if invert else self.depth

        return Marble(position, depth, Direction.WEST)

    @property
    def last_position(self) -> Position:
        return self.positions[self.events[self.last_index].index]

    @property
    def next_index(self) -> int:
        return (self.last_index + 1) % len(self.events)

    @property
    def next_tick(self) -> int | None:
        if self.waiting:
            return None
        delta = self.events[self.next_index].index - self.events[self.last_index].index
        if delta <= 0:
            delta += self.length
        return self.last_tick + delta

    def process_event(self, simulation: Simulation):
        assert not self.waiting
        self.last_index, self.last_tick = self.next_index, simulation.tick
        self.waiting = self.events[self.last_index].process(simulation, self)
        if self.waiting:
            simulation.add_waiting_circuit(self)
        else:
            simulation.add_running_circuit(self)


class Simulation:
    def __init__(
        self,
        grid: list[dict[int, str]],
        circuits: list[Circuit],
        input_stream: typing.IO[bytes],
        output_stream: typing.IO[bytes],
    ):
        self.tick = 0
        self.grid = grid
        self.circuits = circuits
        self.waiting_circuit: dict[Position, Circuit] = {}
        self.priority_queue: list[tuple[int, int, Circuit]] = []
        self.counter = iter(itertools.count())
        self.displays: dict[Position, bool] = {}
        self.display_info: dict[Position, tuple[set[Position], str, str]] = {}
        self.sorted_circuits: dict[int, list[Circuit]] = {}
        self.sorted_display: dict[int, list[DisplayInfo]] = {}

        # IO
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.input_bits: list[int] = []
        self.output_bits: list[int] = []

        # Cycles
        self.last_cycles: dict[Circuit, int] = {}
        self.cycle_counter: dict[int, int] = {}
        self.global_cycle: int | None = None
        self.global_cycle_start_tick: int = 0
        self.global_cycle_start_positions: set[Position] = set()
        self.global_cycle_reference_circuit: Circuit | None = None
        self.global_cycle_count: int = 0

        for circuit in circuits:
            for display_info in circuit.displays.values():
                self.sorted_display.setdefault(display_info.max_span, []).append(
                    display_info
                )
            self.add_running_circuit(circuit)
            self.sorted_circuits.setdefault(circuit.max_span, []).append(circuit)

        # Sort circuit using min_x
        for lst in self.sorted_circuits.values():
            lst.sort()

    def register_cycle(self, circuit: Circuit, cycle: int) -> None:
        if self.global_cycle is not None:
            if circuit is self.global_cycle_reference_circuit:
                assert (self.tick - self.global_cycle_start_tick) % self.global_cycle == 0
                new_positions = {
                    circuit.get_marble(self.tick).p for circuit in self.circuits
                }
                assert new_positions == self.global_cycle_start_positions
                self.global_cycle_count += 1
            return
        last_cycle = self.last_cycles.get(circuit)
        self.last_cycles[circuit] = cycle
        if last_cycle is not None:
            self.cycle_counter[last_cycle] -= 1
            if self.cycle_counter[last_cycle] == 0:
                del self.cycle_counter[last_cycle]
        self.cycle_counter.setdefault(cycle, 0)
        self.cycle_counter[cycle] += 1
        first_key, first_value = next(iter(self.cycle_counter.items()))
        if first_value == len(self.circuits):
            assert len(self.cycle_counter) == 1
            self.global_cycle = first_key
            self.global_cycle_start_tick = self.tick
            self.global_cycle_reference_circuit = circuit
            self.global_cycle_start_positions = {
                circuit.get_marble(self.tick).p for circuit in self.circuits
            }

    @functools.lru_cache(maxsize=16)
    def get_circuits(self, min_x: int, max_x: int) -> set[Circuit]:
        result = set()
        for max_span, circuits in self.sorted_circuits.items():
            index = bisect.bisect_left(circuits, min_x - max_span)
            for circuit in itertools.islice(circuits, index, None, None):
                if circuit.min_x > max_x:
                    break
                if min_x <= circuit.max_x:
                    result.add(circuit)
        return result

    @functools.lru_cache(maxsize=16)
    def get_display_info(self, min_x: int, max_x: int) -> set[DisplayInfo]:
        result = set()
        for max_span, displays in self.sorted_display.items():
            index = bisect.bisect_left(displays, min_x - max_span)
            for display in itertools.islice(displays, index, None, None):
                if display.min_x > max_x:
                    break
                if min_x <= display.max_x:
                    result.add(display)
        return result

    def get_marbles(self, min_x: int, max_x: int) -> set[Marble]:
        return {
            circuit.get_marble(self.tick) for circuit in self.get_circuits(min_x, max_x)
        }

    def run_until(self, tick: int, deadline: float):
        while True:
            if not self.priority_queue:
                raise RuntimeError("Deadlock!")
            next_tick, _, next_circuit = self.priority_queue[0]
            if next_tick > tick:
                self.tick = tick
                return
            heapq.heappop(self.priority_queue)
            self.tick = next_tick
            next_circuit.process_event(self)
            if time.time() > deadline:
                return

    def add_waiting_circuit(self, circuit: Circuit) -> None:
        assert circuit.last_position not in self.waiting_circuit
        self.waiting_circuit[circuit.last_position] = circuit

    def pop_waiting_circuit(self, position: Position) -> Circuit | None:
        return self.waiting_circuit.pop(position, None)

    def add_running_circuit(self, circuit: Circuit) -> None:
        next_tick = circuit.next_tick
        assert next_tick is not None
        item = (next_tick, next(self.counter), circuit)
        heapq.heappush(self.priority_queue, item)

    def write_bit(self, value: int) -> None:
        self.output_bits.append(value)
        if len(self.output_bits) != 8:
            return
        # Get byte
        byte: int = 0
        for x in reversed(self.output_bits):
            byte <<= 1
            byte += int(x)
        self.output_bits = []
        # Write to stream
        if self.output_stream is None:
            return
        self.output_stream.write(bytes([byte]))
        self.output_stream.flush()

    def read_bit(self) -> int:
        if not self.input_bits:
            char = self.input_stream.read(1)
            if not char:
                raise EOFError
            self.input_bits = list(map(bool, map(int, format(ord(char), "08b"))))
        return self.input_bits.pop()

    def get_display_chars(self, x_min: int, x_max: int) -> set[tuple[Position, str]]:
        result: set[tuple[Position, str]] = set()
        for display_info in self.get_display_info(x_min, x_max):
            result.update(display_info.get_display_chars())
        return result


def build_display_info(grid: list[dict[int, str]], position: Position) -> DisplayInfo:
    c = get_character(grid, position)
    if c in DISPLAY_ON + DISPLAY_OFF:
        initial_value = c == DISPLAY_ON
        return DisplayInfo({position}, DISPLAY_OFF, DISPLAY_ON, initial_value)
    assert c in GRID_ON + GRID_OFF

    initial_value = c == GRID_ON
    queue: list[Position] = [position]
    positions = set()

    while queue:
        p = queue.pop()
        if p in positions:
            continue
        c = get_character(grid, p)
        if c not in GRID_ON + GRID_OFF:
            continue
        positions.add(p)
        queue.extend(p.then(d) for d in Direction)

    return DisplayInfo(positions, GRID_OFF, GRID_ON, initial_value)


def build_circuit(grid: list[dict[int, str]], marble: Marble) -> Circuit:
    current_p = marble.p
    current_d = marble.d

    positions: list[Position] = []
    invertors: list[int] = []
    events: list[Event] = [Start(0, False)]
    displays: dict[Position, DisplayInfo] = {}
    inverted: bool = False

    # Loop over steps
    for i in itertools.count():
        positions.append(current_p)

        # Get new direction
        directions = get_directions(grid, current_p)
        if len(directions) == 2:
            d1, d2 = directions
            assert current_d.opposite in (d1, d2)
            new_d = d2 if current_d.opposite == d1 else d1
        elif len(directions) == 4:
            new_d = current_d
        else:
            assert False

        # Check character
        c = get_character(grid, current_p)

        # Invertors
        if c in INVERTORS:
            inverted = not inverted
            invertors.append(i)

        # Conditional clear
        if c in CONDITIONAL_CLEAR:
            d = CONDITIONAL_CLEAR[c]
            neighbor = current_p.then(d)
            neighbor_char = get_character(grid, neighbor)
            # Connected to control
            if neighbor_char in CONTROL:
                assert CONTROL[neighbor_char] == d.opposite
                events.append(ConditionalClear(i, inverted, neighbor))
                inverted = False
            # Connected to static marble
            elif neighbor_char == MARBLE_LOWER:
                events.append(UnconditionalClear(i, inverted))
                inverted = False
            elif neighbor_char == MARBLE_UPPER:
                pass
            # Connected to input
            elif neighbor_char in IOS:
                events.append(ReadBit(i, inverted))
                inverted = False
            # Unsupported
            else:
                assert False, f"`{c}` + `{neighbor_char}` not supported"

        # Control
        if c in CONTROL:
            d = CONTROL[c]
            neighbor = current_p.then(d)
            neighbor_char = get_character(grid, neighbor)
            # To conditional clear
            if neighbor_char in CONDITIONAL_CLEAR:
                assert CONDITIONAL_CLEAR[neighbor_char] == d.opposite
                events.append(Control(i, inverted, neighbor))
                inverted = False
            # To display
            elif neighbor_char in DISPLAYS:
                events.append(Display(i, inverted, neighbor))
                displays[neighbor] = build_display_info(grid, neighbor)
                inverted = False
            # To exit
            elif neighbor_char in EXIT:
                events.append(Exit(i, inverted))
                inverted = False
            # To output
            elif neighbor_char in IO_0:
                events.append(WriteZero(i, inverted))
                inverted = False
            elif neighbor_char == IO_1:
                events.append(WriteOne(i, inverted))
                inverted = False
            # Unrecognized pattern
            else:
                assert False, f"`{c}` + `{neighbor_char}` not supported"

        # Get new position
        new_p = current_p.then(new_d)

        # Check for stop condition
        if new_p == marble.p:
            break
        current_p = new_p
        current_d = new_d

    events[0].inverted = inverted
    return Circuit(marble, positions, events, invertors, displays)


# Drawing helpers


def get_char_from_stdin() -> bytes | None:
    fd = sys.stdin.fileno()
    read, _, _ = select.select([fd], [], [], 0)
    if fd not in read:
        return None
    char = sys.stdin.buffer.read(1)
    if not char:
        exit()
    return char


class ScreenDisplay:
    def __init__(self, simulation: Simulation, speed: float, fps: float):
        self.fps = fps
        self.speed = speed
        self.simulation = simulation

        self.x_offset = 0
        self.y_offset = 0
        self.termsize = os.terminal_size((0, 0))
        self.changed = True
        self.old_marbles: set[Marble] = set()
        self.old_displays: set[tuple[Position, str]] = set()

        self.frame: int
        self.start_time: float
        self.start_frame: int
        self.start_simulation_tick: int

    def run(self):
        self.start_frame = 0
        self.start_time = time.time()
        self.start_simulation_tick = 0

        # Iterator over steps
        for self.frame in itertools.count(1):
            previous_time = time.time()
            previous_simulation_tick = self.simulation.tick

            # Control
            self.check_stdin()

            step1 = time.time() - previous_time

            # Show simulation
            self.show()

            step2 = time.time() - previous_time - step1

            delta_frame = self.frame - self.start_frame
            simulation_tick = self.start_simulation_tick + int(
                delta_frame * self.speed / self.fps
            )
            deadline = self.start_time + delta_frame / self.fps
            self.simulation.run_until(simulation_tick, deadline)

            step3 = time.time() - previous_time - step1 - step2

            # Wait for the next tick
            delta = deadline - time.time()
            if delta > 0:
                time.sleep(delta)

            step4 = time.time() - previous_time - step1 - step2 - step3

            actual_fps = 1 / (time.time() - previous_time)
            actual_speed_per_frame = self.simulation.tick - previous_simulation_tick
            steps = step1, step2, step3, step4
            self.print_status_bar(actual_fps, actual_speed_per_frame, steps)

    def print_status_bar(self, fps, speed_per_frame, steps):
        info = f"FPS: {fps:7.2f} "
        info += f"  | Tick per frame: {speed_per_frame:5d}"
        info += f"  | Speed: {int(speed_per_frame * fps):6d}"
        info += f"  | Frame: {self.frame: 8d}"
        sum_steps = sum(steps)
        stdin, show, run, sleep = [int(round(s * 100 / sum_steps)) for s in steps]
        info += f"  | Display: {show:3d} % CPU"
        info += f"  | Simulation: {run:3d} % CPU"
        if self.simulation.global_cycle is None:
            info += f"  | Looking for global cycle..."
        else:
            info += f"  | Global cycle: {self.simulation.global_cycle_count:4d} ({self.simulation.global_cycle:5d} ticks)"
        i = self.termsize.lines - 2
        j = 0
        string = f"\033[{i+1};{j+1}H"
        string += "━" * self.termsize.columns
        i += i
        string += f"\033[{i+1};{j+1}H"
        string += info[: self.termsize.columns].ljust(self.termsize.columns)
        print(string, end="", flush=True)

    def check_stdin(self):
        if not sys.stdin.isatty():
            return
        while True:
            char = get_char_from_stdin()
            if char == b"\x03":
                raise KeyboardInterrupt
            elif char == b"\x1b":
                char = sys.stdin.buffer.read(1)
                assert char == b"[", char
                char = sys.stdin.buffer.read(1)
                assert char in b"ABCD"
                self.changed = True
                if char == b"A":
                    self.x_offset += 1
                elif char == b"B":
                    self.x_offset -= 1
                elif char == b"C":
                    self.y_offset -= 1
                elif char == b"D":
                    self.y_offset += 1
            elif char == b"b" or char == b"v":
                if char == b"b":
                    self.speed *= 1.5
                else:
                    self.speed /= 1.5
                self.start_time = time.time()
                self.start_frame = self.frame
                self.start_simulation_tick = self.simulation.tick
            else:
                break

    def show(self) -> None:
        simulation = self.simulation
        termsize = os.get_terminal_size()
        if termsize != self.termsize:
            self.termsize = termsize
            self.changed = True
        if self.changed:
            self.changed = False
            print(self.draw_grid(), end="")
            self.old_marbles = set()
            self.old_displays = set()
        min_x = -self.x_offset
        max_x = self.termsize.lines - 2 - 1 - self.x_offset
        new_marbles = simulation.get_marbles(min_x, max_x)
        unchanged = new_marbles & self.old_marbles
        to_clear = self.old_marbles - unchanged
        to_draw = new_marbles - unchanged
        print(self.clear_marbles(to_clear), end="")
        print(self.draw_marbles(to_draw), end="")
        new_displays = simulation.get_display_chars(min_x, max_x)
        print(
            self.draw_displays(new_displays - self.old_displays),
            end="",
        )
        print(end="", flush=True)
        self.old_marbles = new_marbles
        self.old_displays = new_displays

    def draw_char(self, i: int, j: int, char: str):
        i += self.x_offset
        j += self.y_offset
        if not 0 <= i < self.termsize.lines - 2:
            return ""
        if not 0 <= j < self.termsize.columns:
            return ""
        return f"\033[{i+1};{j+1}H{char or ' '}"

    def draw_displays(
        self,
        displays: set[tuple[Position, str]]
    ) -> str:
        result = ""
        for position, char in displays:
            result += self.draw_char(position.x, position.y, char)
        return result

    def draw_grid(self) -> str:
        grid = self.simulation.grid
        i = self.termsize.lines - 2 - 1
        j = self.termsize.columns - 1
        result = f"\033[{i+1};{j+1}H\033[1J"
        for i in range(self.termsize.lines):
            i -= self.x_offset
            if not 0 <= i < len(grid):
                continue
            row = grid[i]
            for j, char in row.items():
                result += self.draw_char(i, j, char)
        return result

    def clear_marbles(self, marbles: set[Marble]) -> str:
        grid = self.simulation.grid
        result = ""
        for marble in marbles:
            p = marble.p
            char = get_character(grid, p)
            result += self.draw_char(p.x, p.y, char)
        return result

    def draw_marbles(
        self,
        new_marbles: set[Marble],
        show_LOWER: bool = True,
    ):
        result = ""
        for marble in new_marbles:
            if marble.z == Depth.UPPER:
                result += self.draw_char(marble.p.x, marble.p.y, MARBLE_UPPER)
            elif show_LOWER:
                result += self.draw_char(marble.p.x, marble.p.y, MARBLE_LOWER)
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


# Main loop


def main(
    file: io.TextIOBase,
    speed: float = 10.0,
    fps: float = 60.0,
    input_stream: typing.IO[bytes] | None = None,
    output_stream: typing.IO[bytes] | None = None,
):
    # Input/Output
    if input_stream is None and not sys.stdin.isatty():
        input_stream = sys.stdin.buffer
    if output_stream is None and not sys.stderr.isatty():
        output_stream = sys.stderr.buffer

    if input_stream is None:
        input_stream = io.BytesIO()
    if output_stream is None:
        output_stream = io.BytesIO()

    # Make grid
    grid: list[dict[int, str]] = [
        {j: char for j, char in enumerate(line) if char.strip()}
        for line in file
        if line.strip()
    ]

    # Extract moving marbles from grid
    marbles: set[Marble] = set()
    for i, row in enumerate(grid):
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

    # Find circuit for each marble
    circuits = []
    for marble in marbles:
        circuit = build_circuit(grid, marble)
        circuits.append(circuit)
    simulation = Simulation(grid, circuits, input_stream, output_stream)
    display = ScreenDisplay(simulation, speed, fps)

    try:
        with drawing_context():
            display.run()

    # Output captured IO if necessary
    finally:
        if isinstance(output_stream, io.BytesIO):
            sys.stdout.buffer.write(output_stream.getvalue())
            print()


def test_run():
    main(open("true_hello_world.txt"), 1000000, 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=argparse.FileType())
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--input", type=argparse.FileType("rb"), default=None)
    parser.add_argument("--output", type=argparse.FileType("wb"), default=None)
    namespace = parser.parse_args()
    main(
        namespace.file,
        namespace.speed,
        namespace.fps,
        namespace.input,
        namespace.output,
    )