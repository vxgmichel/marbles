from __future__ import annotations
import bisect
import contextlib
import hashlib

import io
import enum
import heapq
import math
import os
import operator
import pathlib
import pickle
import select
import string
import argparse
import itertools
import functools
import sys
import termios
import time
import tty
from typing import Callable, NamedTuple, TYPE_CHECKING

import tqdm

# Compatibility

try:
    math.lcm
except AttributeError:
    lcm = lambda a, b: abs(a * b) // math.gcd(a, b)
    math.lcm = lambda *args: functools.reduce(lcm, args)


# Types

if TYPE_CHECKING:
    Grid = list[dict[int, str]]


# Data structures


class Depth(enum.IntEnum):
    LOWER = 0
    UPPER = 1

    @property
    def inverse(self):
        return Depth(not self.value)


class Direction(enum.Enum):
    EAST = (0, 1)
    SOUTH = (1, 0)
    NORTH = (-1, 0)
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


class EventType(enum.Enum):
    START = enum.auto()
    UNCONDITIONAL_CLEAR = enum.auto()
    CONDITIONAL_CLEAR = enum.auto()
    CONTROL = enum.auto()
    DISPLAY = enum.auto()
    EXIT = enum.auto()
    READ_BIT = enum.auto()
    WRITE_ZERO = enum.auto()
    WRITE_ONE = enum.auto()


class Event:
    def __init__(
        self,
        event_type: EventType,
        index: int,
        inverted: bool,
        neighbor: Position | None = None,
    ):
        self.index = index
        self.event_type = event_type
        self.inverted = inverted
        self.neighbor = neighbor


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
        circuit_id: int,
        marble: Marble,
        positions: list[Position],
        events: list[Event],
        invertors: list[int],
        displays: dict[Position, DisplayInfo],
    ):
        self.id = circuit_id
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

    def __lt__(self, other):
        if isinstance(other, Circuit):
            return self.min_x.__lt__(other.min_x)
        if isinstance(other, int):
            return self.min_x.__lt__(other)
        raise NotImplementedError(other)


class Group:
    def __init__(
        self,
        circuits: list[Circuit],
        sorted_circuits: dict[int, list[Circuit]],
        cycle_start: int,
        cycle_length: int,
        ticks_to_event_indexes: dict[Circuit, list[tuple[int, int, bool]]],
    ):
        self.circuits = circuits
        self.cycle_start = cycle_start
        self.cycle_length = cycle_length
        self.ticks_to_event_indexes = ticks_to_event_indexes
        self.sorted_circuits = sorted_circuits


# Characters

MARBLE_UPPER = "●"
MARBLE_LOWER = "○"
MARBLES = MARBLE_LOWER + MARBLE_UPPER

GRID_ON = "█"
GRID_OFF = "┼"
GRIDS = GRID_ON + GRID_OFF
DISPLAY_ON = "▣"
DISPLAY_OFF = "□"
DISPLAYS = DISPLAY_ON + DISPLAY_OFF
DISPLAYS_OR_GRIDS = GRIDS + DISPLAYS

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

DIRECTIONS_TO_TRACKS: dict[tuple[Direction, ...], str] = {}
for char, directions in TRACKS.items():
    key = tuple(sorted(directions, key=list(Direction).index))
    DIRECTIONS_TO_TRACKS.setdefault(key, char)

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

# Helpers


def get_character(grid: list[dict[int, str]], p: Position) -> str:
    if not 0 <= p.x < len(grid):
        return " "
    return grid[p.x].get(p.y, " ")


def get_directions(grid: Grid, p: Position) -> set[Direction]:
    return TRACKS.get(get_character(grid, p), set())


def guess_directions(grid: Grid, p: Position) -> set[Direction]:
    result = set(get_directions(grid, p))
    for d in Direction:
        if d.opposite in get_directions(grid, p.then(d)):
            result |= {d}
    return result


# Analysis


def create_grid(file: io.TextIOBase) -> Grid:
    return [
        {i: char for i, char in enumerate(line) if char not in string.whitespace}
        for line in tqdm.tqdm(file, desc="Creating grid", unit=" lines")
    ]


def extract_marbles(grid: Grid) -> list[Marble]:
    with tqdm.tqdm(desc="Extracting marbles", unit=" marbles") as progress_bar:
        marbles = []

        # Loop over rows
        for i, row in enumerate(grid):

            # Loop over columns
            for j, char in row.items():

                # Not a marble
                if char not in MARBLES:
                    continue

                # Get info
                p = Position(i, j)
                depth = Depth.UPPER if char == MARBLE_UPPER else Depth.LOWER
                directions = guess_directions(grid, p)

                # Check directions
                if len(directions) == 0:
                    pass
                elif len(directions) == 1:
                    assert False
                elif len(directions) == 2:
                    d1, d2 = sorted(directions, key=list(Direction).index)
                    marbles.append(Marble(p, depth, d2.opposite))
                    grid[i][j] = DIRECTIONS_TO_TRACKS[(d1, d2)]
                    progress_bar.update(1)
                elif len(directions) == 3:
                    assert False
                elif len(directions) == 4:
                    direction = list(Direction)[0]
                    marbles.append(Marble(p, depth, direction))
                    progress_bar.update(1)
                    grid[i][j] = "╬"

        return marbles


def build_display_info(grid: list[dict[int, str]], position: Position) -> DisplayInfo:
    c = get_character(grid, position)
    if c in DISPLAYS:
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
        if c not in GRIDS:
            continue
        positions.add(p)
        queue.extend(p.then(d) for d in Direction)

    return DisplayInfo(positions, GRID_OFF, GRID_ON, initial_value)


def build_circuit(
    grid: list[dict[int, str]], marble: Marble, circuit_id: int
) -> Circuit:
    current_p = marble.p
    current_d = marble.d

    positions: list[Position] = []
    invertors: list[int] = []
    events: list[Event] = [Event(EventType.START, 0, False)]
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
                events.append(Event(EventType.CONDITIONAL_CLEAR, i, inverted, neighbor))
                inverted = False
            # Connected to static marble
            elif neighbor_char == MARBLE_LOWER:
                events.append(
                    Event(EventType.UNCONDITIONAL_CLEAR, i, inverted, neighbor)
                )
                inverted = False
            elif neighbor_char == MARBLE_UPPER:
                pass
            # Connected to input
            elif neighbor_char in IOS:
                events.append(Event(EventType.READ_BIT, i, inverted, neighbor))
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
                events.append(Event(EventType.CONTROL, i, inverted, neighbor))
                inverted = False
            # To display
            elif neighbor_char in DISPLAYS_OR_GRIDS:
                events.append(Event(EventType.DISPLAY, i, inverted, neighbor))
                displays[neighbor] = build_display_info(grid, neighbor)
                inverted = False
            # To exit
            elif neighbor_char in EXIT:
                events.append(Event(EventType.EXIT, i, inverted, neighbor))
                inverted = False
            # To output
            elif neighbor_char in IO_0:
                events.append(Event(EventType.WRITE_ZERO, i, inverted, neighbor))
                inverted = False
            elif neighbor_char == IO_1:
                events.append(Event(EventType.WRITE_ONE, i, inverted, neighbor))
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
    return Circuit(circuit_id, marble, positions, events, invertors, displays)


def build_circuits(grid: Grid, marbles: list[Marble]) -> list[Circuit]:
    return [
        build_circuit(grid, marble, circuit_id)
        for circuit_id, marble in enumerate(
            tqdm.tqdm(
                marbles, desc="Building circuits", unit=" circuit", colour="green"
            )
        )
    ]


def build_groups(circuits: list[Circuit]) -> list[list[Circuit]]:
    # Build a mapping
    mapping: dict[Position, Circuit] = {}
    for circuit in tqdm.tqdm(
        circuits,
        desc="Prepare mapping for group detection",
        unit=" circuit",
        colour="green",
    ):
        for event in circuit.events:
            if event.neighbor is None:
                continue
            position = circuit.positions[event.index]
            mapping[position] = circuit

    # Prepare groups
    with tqdm.tqdm(desc="Detecting groups", unit=" group") as progress_bar:
        groups = []
        unseen = set(circuits)

        # Loop over groups
        while unseen:

            # Prepare group
            current_group = [unseen.pop()]
            todo = set(current_group)

            # Loop over circuit
            while todo:
                circuit = todo.pop()
                for event in circuit.events:
                    if event.neighbor is None:
                        continue
                    other = mapping.get(event.neighbor)
                    if other is not None and other in unseen:
                        todo.add(other)
                        unseen.discard(other)
                        current_group.append(other)

            # Sort and add group
            current_group.sort(key=lambda x: x.id)
            groups.append(current_group)
            progress_bar.update(1)

    return groups


def process_event(
    circuit: Circuit,
    event_index: int,
    waiting_circuits: dict[Position, tuple[Circuit, int]],
) -> tuple[bool, bool, Callable | None,]:
    callback = None
    event = circuit.events[event_index]
    event_type = event.event_type

    if event_type == EventType.START:
        return False, False, None
    elif event_type == EventType.UNCONDITIONAL_CLEAR:
        return False, False, callback
    elif event_type == EventType.EXIT:
        return False, False, callback
    elif event_type == EventType.DISPLAY:
        return False, False, callback
    elif event_type == EventType.READ_BIT:
        return False, False, callback
    elif event_type == EventType.WRITE_ONE:
        return False, False, callback
    elif event_type == EventType.WRITE_ZERO:
        return False, False, callback
    elif event_type in (EventType.CONDITIONAL_CLEAR, EventType.CONTROL):
        assert event.neighbor is not None
        if event.neighbor in waiting_circuits:
            return False, True, callback
        else:
            return True, False, callback

    else:
        assert False


def analyze_group(group: list[Circuit], group_id: int) -> Group:
    # Prepare structures
    counter = iter(itertools.count(0))
    cycle_count: dict[int, int] = {}
    last_cycles: dict[Circuit, int] = {}
    sorted_circuits: dict[int, list[Circuit]] = {}
    waiting_circuits: dict[Position, tuple[Circuit, int]] = {}
    event_index_to_ticks: dict[Circuit, dict[int, int]] = {}
    tick_to_event_indexes: dict[Circuit, list[tuple[int, int, bool]]] = {}
    priority_queue: list[tuple[int, int, int, Circuit]] = []

    # Prepare structures
    for circuit in group:
        sorted_circuits.setdefault(circuit.max_span, []).append(circuit)
        event_index_to_ticks[circuit] = {}
        tick_to_event_indexes[circuit] = []
        priority_queue.append((0, 0, next(counter), circuit))

    # Sort circuit using min_x
    for lst in sorted_circuits.values():
        lst.sort()

    # Helper
    def push_item(circuit: Circuit, event_index: int, current_tick: int):
        next_event_index = (event_index + 1) % len(circuit.events)
        delta_tick = (
            circuit.events[next_event_index].index - circuit.events[event_index].index
        )
        delta_tick = delta_tick % circuit.length if delta_tick != 0 else circuit.length
        item = (current_tick + delta_tick, next_event_index, next(counter), circuit)
        heapq.heappush(priority_queue, item)

    # Loop over events in queue
    with tqdm.tqdm(desc=f"Analyzing group {group_id}", unit=" tick") as progress_bar:
        while True:

            # Deadlock detection
            if not priority_queue:
                raise RuntimeError("Deadlock!")
            current_tick, event_index, _, current_circuit = heapq.heappop(
                priority_queue
            )

            # Update progress bar
            progress_bar.n = current_tick
            progress_bar.update(0)

            # Save tick info
            current_event_index_to_ticks = event_index_to_ticks[current_circuit]
            last_tick = current_event_index_to_ticks.get(event_index)
            current_event_index_to_ticks[event_index] = current_tick

            # Save cycle info
            if last_tick is not None:
                cycle = current_tick - last_tick
                last_cycle = last_cycles.get(current_circuit)
                last_cycles[current_circuit] = cycle

                # Update cycle count
                if cycle != last_cycle:
                    cycle_count.setdefault(cycle, 0)
                    cycle_count[cycle] += 1
                    if last_cycle is not None:
                        cycle_count[last_cycle] -= 1
                        if cycle_count[last_cycle] == 0:
                            del cycle_count[last_cycle]

                    # Check for global cycle
                    first_key, first_value = next(iter(cycle_count.items()))
                    if first_value == len(group):
                        assert len(cycle_count) == 1
                        cycle = first_key
                        start = current_tick - cycle
                        return Group(
                            group, sorted_circuits, start, cycle, tick_to_event_indexes
                        )

            # Process next_event
            waiting, restore, callback = process_event(
                current_circuit, event_index, waiting_circuits
            )

            # Pause this circuit
            if waiting:
                assert not restore
                position = current_circuit.positions[
                    current_circuit.events[event_index].index
                ]
                waiting_circuits[position] = (current_circuit, event_index)
            # Reschedule this circuit
            else:
                push_item(current_circuit, event_index, current_tick)

            # Restore another circuit
            if restore:
                neighbor = current_circuit.events[event_index].neighbor
                assert neighbor is not None
                other_circuit, other_circuit_event_index = waiting_circuits.pop(
                    neighbor
                )
                push_item(other_circuit, other_circuit_event_index, current_tick)

                # Save event info
                current_tick_to_event_indexes = tick_to_event_indexes[other_circuit]
                current_tick_to_event_indexes.append(
                    (current_tick, other_circuit_event_index, False)
                )

            # Save event info
            current_tick_to_event_indexes = tick_to_event_indexes[current_circuit]
            current_tick_to_event_indexes.append((current_tick, event_index, waiting))


# Display info


@functools.lru_cache(maxsize=16)
def get_circuits_in_window(group: Group, min_x: int, max_x: int) -> list[Circuit]:
    result = []
    # Get circuits in frame
    for max_span, circuits in group.sorted_circuits.items():
        index = bisect.bisect_left(circuits, min_x - max_span)
        for circuit in itertools.islice(circuits, index, None, None):
            if circuit.min_x > max_x:
                break
            if min_x <= circuit.max_x:
                result.append(circuit)
    return result


def get_marbles_in_window(
    group: Group, tick: int, values: list[int], min_x: int, max_x: int
) -> set[tuple[Position, int]]:
    result = set()

    # Normalize tick
    if tick >= group.cycle_start:
        tick = ((tick - group.cycle_start) % group.cycle_length) + group.cycle_start

    # Loop over circuits
    for circuit in get_circuits_in_window(group, min_x, max_x):

        # Get event info
        ticks_to_event_indexes = group.ticks_to_event_indexes[circuit]
        index = bisect.bisect(ticks_to_event_indexes, (tick, float("inf"))) - 1
        previous_tick, event_index, waiting = ticks_to_event_indexes[index]
        event = circuit.events[event_index]

        # Marble is waiting
        if waiting:
            position = circuit.positions[event.index]
            value = values[circuit.id] ^ event.inverted

        # Marble is running
        else:
            position_index = event.index
            i_start = bisect.bisect_left(circuit.invertors, position_index)
            position_index += tick - previous_tick
            i_stop = bisect.bisect_left(circuit.invertors, position_index)
            position_index %= circuit.length
            position = circuit.positions[position_index]
            invert = (i_stop - i_start) % 2
            value = values[circuit.id] ^ invert

        # Add marble to the result
        result.add((position, value))

    return result


# Display


def get_char_from_stdin() -> bytes | None:
    fd = sys.stdin.fileno()
    read, _, _ = select.select([fd], [], [], 0)
    if fd not in read:
        return None
    char = sys.stdin.buffer.read(1)
    if not char:
        exit()
    return char


@contextlib.contextmanager
def drawing_context():
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


class ScreenDisplay:
    def __init__(self, grid: Grid, speed: float, fps: float):
        self.fps = fps
        self.grid = grid
        self.speed = speed

        self.x_offset = 0
        self.y_offset = 0
        self.changed = True
        self.termsize = os.terminal_size((0, 0))
        self.old_marbles: set[tuple[Position, int]] = set()
        self.old_displays: set[tuple[Position, str]] = set()

        self.frame: int
        self.start_time: float
        self.start_frame: int

    def run(
        self,
        groups: list[Group],
        values: list[int],
        cycle_start: int,
        cycle_length: int,
    ):
        self.start_frame = 0
        self.start_time = time.time()
        simulation_tick = start_simulation_tick = 0

        # Iterator over steps
        for self.frame in itertools.count(1):
            previous_time = time.time()
            previous_simulation_tick = simulation_tick

            # Control
            if self.check_stdin():
                self.start_time = time.time()
                self.start_frame = self.frame
                start_simulation_tick = simulation_tick
            step1 = time.time() - previous_time

            # Show simulation
            self.show(groups, simulation_tick, values)

            step2 = time.time() - previous_time - step1

            delta_frame = self.frame - self.start_frame
            simulation_tick = start_simulation_tick + int(
                delta_frame * self.speed / self.fps
            )
            deadline = self.start_time + delta_frame / self.fps
            # simulation.run_until(simulation_tick, deadline)

            step3 = time.time() - previous_time - step1 - step2

            # Wait for the next tick
            delta = deadline - time.time()
            if delta > 0:
                time.sleep(delta)

            step4 = time.time() - previous_time - step1 - step2 - step3

            actual_fps = 1 / (time.time() - previous_time)
            actual_speed_per_frame = simulation_tick - previous_simulation_tick
            steps = step1, step2, step3, step4
            cycle_count = (simulation_tick - cycle_start) // cycle_length
            self.print_status_bar(
                actual_fps, actual_speed_per_frame, steps, cycle_count, cycle_length
            )

    def print_status_bar(
        self,
        fps: float,
        speed_per_frame: int,
        steps: tuple[float, float, float, float],
        cycle_count: int,
        cycle_length: int,
    ):
        info = f"FPS: {fps:7.2f} "
        info += f"  | Tick per frame: {speed_per_frame:5d}"
        info += f"  | Speed: {int(speed_per_frame * fps):6d}"
        info += f"  | Frame: {self.frame: 8d}"
        sum_steps = sum(steps)
        stdin, show, run, sleep = [int(round(s * 100 / sum_steps)) for s in steps]
        info += f"  | Display: {show:3d} % CPU"
        info += f"  | Simulation: {run:3d} % CPU"
        info += f"  | Global cycle: {cycle_count:4d} ({cycle_length:5d} ticks)"
        i = self.termsize.lines - 2
        j = 0
        string = f"\033[{i+1};{j+1}H"
        string += "━" * self.termsize.columns
        i += i
        string += f"\033[{i+1};{j+1}H"
        string += info[: self.termsize.columns].ljust(self.termsize.columns)
        print(string, end="", flush=True)

    def check_stdin(self) -> bool:
        if not sys.stdin.isatty():
            return False
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
                return True
            else:
                break
        return False

    def show(self, groups: list[Group], tick: int, values: list[int]) -> None:
        # Update termsize
        termsize = os.get_terminal_size()
        if termsize != self.termsize:
            self.termsize = termsize
            self.changed = True
        # Current window has changed
        if self.changed:
            self.changed = False
            print(self.draw_grid(), end="")
            self.old_marbles = set()
        # Get current window
        min_x = -self.x_offset
        max_x = min_x + self.termsize.lines - 2
        marbles = {
            item
            for group in groups
            for item in get_marbles_in_window(
                group, tick, values, min_x, max_x
            )
        }
        # Update marbles
        print(self.clear_marbles(self.old_marbles), end="")
        print(self.draw_marbles(marbles), end="", flush=True)
        # Save marbles
        self.old_marbles = marbles

        # simulation = self.simulation
        # termsize = os.get_terminal_size()
        # min_x = -self.x_offset
        # max_x = self.termsize.lines - 2 - 1 - self.x_offset
        # new_marbles = simulation.get_marbles(min_x, max_x)
        # unchanged = new_marbles & self.old_marbles
        # to_clear = self.old_marbles - unchanged
        # to_draw = new_marbles - unchanged
        # print(self.clear_marbles(to_clear), end="")
        # print(self.draw_marbles(to_draw), end="")
        # new_displays = simulation.get_display_chars(min_x, max_x)
        # print(
        #     self.draw_displays(new_displays - self.old_displays),
        #     end="",
        # )
        # print(end="", flush=True)
        # self.old_marbles = new_marbles
        # self.old_displays = new_displays

    def draw_char(self, i: int, j: int, char: str):
        i += self.x_offset
        j += self.y_offset
        if not 0 <= i < self.termsize.lines - 2:
            return ""
        if not 0 <= j < self.termsize.columns:
            return ""
        return f"\033[{i+1};{j+1}H{char or ' '}"

    def draw_displays(self, displays: set[tuple[Position, str]]) -> str:
        result = ""
        for position, char in displays:
            result += self.draw_char(position.x, position.y, char)
        return result

    def draw_grid(self) -> str:
        i = self.termsize.lines - 2 - 1
        j = self.termsize.columns - 1
        result = [f"\033[{i+1};{j+1}H\033[1J"]
        for i in range(self.termsize.lines):
            i -= self.x_offset
            if not 0 <= i < len(self.grid):
                continue
            row = self.grid[i]
            for j, char in row.items():
                result.append(self.draw_char(i, j, char))
        return "".join(result)

    def clear_marbles(self, marbles: set[tuple[Position, int]]) -> str:
        result = []
        for position, _ in marbles:
            char = get_character(self.grid, position)
            result.append(self.draw_char(position.x, position.y, char))
        return "".join(result)

    def draw_marbles(
        self,
        marbles: set[tuple[Position, int]],
        show_lower: bool = True,
    ) -> str:
        result = []
        for position, value in marbles:
            if value:
                result.append(self.draw_char(position.x, position.y, MARBLE_UPPER))
            elif show_lower:
                result.append(self.draw_char(position.x, position.y, MARBLE_LOWER))
        return "".join(result)


# Main routine

def get_sha256(path: pathlib.Path, _bufsize=2**18) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        buffer = bytearray(_bufsize)
        view = memoryview(buffer)
        while True:
            size = file.readinto(buffer)
            if size == 0:
                break  # EOF
            digest.update(view[:size])
    return digest.hexdigest()



def main(path: pathlib.Path, speed: float, fps: float, check_cache: bool):
    # Get cache name
    sha256 = get_sha256(path)[:16]
    cache_path = path.parent / (path.name + f".{sha256}.cache")

    # Load grid
    with open(path) as file:
        grid = create_grid(file)

    # Use cache
    if check_cache and cache_path.exists():
        with open(cache_path, "rb") as file:
            groups = pickle.load(file)

    # Perform analysis
    else:
        marbles = extract_marbles(grid)
        circuits = build_circuits(grid, marbles)
        raw_groups = build_groups(circuits)
        groups = [analyze_group(group, i) for i, group in enumerate(raw_groups)]
        with open(cache_path, "wb") as file:
            pickle.dump(groups, file)

    # Global cycle
    cycle_length = math.lcm(*(group.cycle_length for group in groups))
    cycle_start = max(group.cycle_start for group in groups)

    # Values
    values = [0] * sum(len(group.circuits) for group in groups)

    # Simulation
    with drawing_context():

        display = ScreenDisplay(grid, speed, fps)
        display.run(groups, values, cycle_start, cycle_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=pathlib.Path)
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--input", type=argparse.FileType("rb"), default=None)
    parser.add_argument("--output", type=argparse.FileType("wb"), default=None)
    parser.add_argument("--no-cache", action="store_true", default=False)
    namespace = parser.parse_args()
    main(
        namespace.file,
        namespace.speed,
        namespace.fps,
        not namespace.no_cache,
        # namespace.input,
        # namespace.output,
    )
