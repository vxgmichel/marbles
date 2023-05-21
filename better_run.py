from __future__ import annotations
import array
import bisect
import contextlib
import hashlib

import io
import enum
import heapq
import math
import os
import random
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
from typing import IO, Callable, Iterator, NamedTuple, TYPE_CHECKING, MutableSequence

import tqdm

# Compatibility

try:
    math.lcm
except AttributeError:

    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)

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


class Display:
    def __init__(
        self, positions: set[Position], off_char: str, on_char: str, initial_value: bool
    ):
        self.id: int | None = None
        self.initial_value = initial_value
        self.positions = positions
        self.off_char = off_char
        self.on_char = on_char

        # X positioning
        self.min_x = min(p.x for p in positions)
        self.max_x = max(p.x for p in positions)

    def extract_info(self):
        x_positions: list[int] = [position.x for position in self.positions]
        y_positions: list[int] = [position.y for position in self.positions]
        assert self.id is not None
        return DisplayInfo(
            self.id,
            self.min_x,
            self.max_x,
            x_positions,
            y_positions,
            self.off_char,
            self.on_char,
        )


class Circuit:
    def __init__(
        self,
        circuit_id: int,
        marble: Marble,
        positions: list[Position],
        events: list[Event],
        invertors: list[int],
        displays: dict[Position, Display],
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

    def extract_info(
        self,
        cycle_start: int,
        cycle_length: int,
        ticks_to_event_indexes: list[tuple[int, int, bool]],
    ):
        position_indexes: list[int] = [event.index for event in self.events]
        event_inverted: list[int] = [event.inverted for event in self.events]
        invertors: list[int] = self.invertors
        x_positions: list[int] = [position.x for position in self.positions]
        y_positions: list[int] = [position.y for position in self.positions]
        event_ticks: list[int] = [tick for tick, _, _ in ticks_to_event_indexes]
        event_indexes: list[int] = [
            event_index for _, event_index, _ in ticks_to_event_indexes
        ]
        event_waiting: list[int] = [waiting for _, _, waiting in ticks_to_event_indexes]
        return CircuitInfo(
            self.id,
            cycle_start,
            cycle_length,
            self.min_x,
            self.max_x,
            position_indexes,
            event_inverted,
            invertors,
            x_positions,
            y_positions,
            event_ticks,
            event_indexes,
            event_waiting,
        )


class Group:
    def __init__(
        self,
        circuits: list[Circuit],
        displays: list[Display],
        cycle_start: int,
        cycle_length: int,
        ticks_to_event_indexes: dict[Circuit, list[tuple[int, int, bool]]],
        actions: list[tuple[int, Action]],
    ):
        self.circuits = circuits
        self.displays = displays
        self.cycle_start = cycle_start
        self.cycle_length = cycle_length
        self.ticks_to_event_indexes = ticks_to_event_indexes
        self.actions = actions


class Action:
    def make_callback(
        self,
        values: MutableSequence[int],
        display_values: MutableSequence[int],
        read_bit: Callable[[], int],
        write_bit: Callable[[int], None],
    ) -> Callable | None:
        raise NotImplementedError


class StartAction(Action):
    def __init__(self, circuit: Circuit, event: Event):
        self.circuit_id = circuit.id
        self.inverted = event.inverted

    def make_callback(
        self,
        values: MutableSequence[int],
        display_values: MutableSequence[int],
        read_bit: Callable[[], int],
        write_bit: Callable[[int], None],
    ) -> Callable:
        circuit_id = self.circuit_id

        if self.inverted:

            def callback():
                values[circuit_id] ^= 1

        else:

            def callback():
                pass

        return callback


class UnconditionalClearAction(Action):
    def __init__(self, circuit: Circuit, event: Event):
        self.circuit_id = circuit.id
        self.inverted = event.inverted

    def make_callback(
        self,
        values: MutableSequence[int],
        display_values: MutableSequence[int],
        read_bit: Callable[[], int],
        write_bit: Callable[[int], None],
    ) -> Callable | None:
        circuit_id = self.circuit_id

        def callback():
            values[circuit_id] = 0

        return callback


class ExitAction(Action):
    def __init__(self, circuit: Circuit, event: Event):
        self.circuit_id = circuit.id
        self.inverted = event.inverted

    def make_callback(
        self,
        values: MutableSequence[int],
        display_values: MutableSequence[int],
        read_bit: Callable[[], int],
        write_bit: Callable[[int], None],
    ) -> Callable | None:
        circuit_id = self.circuit_id

        def callback():
            if values[circuit_id]:
                exit()

        return callback


class DisplayAction(Action):
    def __init__(self, circuit: Circuit, event: Event):
        self.circuit_id = circuit.id
        self.inverted = event.inverted

        assert event.neighbor is not None
        display_id = circuit.displays[event.neighbor].id
        assert display_id is not None
        self.display_id = display_id

    def make_callback(
        self,
        values: MutableSequence[int],
        display_values: MutableSequence[int],
        read_bit: Callable[[], int],
        write_bit: Callable[[int], None],
    ) -> Callable | None:
        circuit_id = self.circuit_id
        display_id = self.display_id

        if self.inverted:

            def callback():
                values[circuit_id] ^= 1
                display_values[display_id] = values[circuit_id]

        else:

            def callback():
                display_values[display_id] = values[circuit_id]

        return callback


class ReadBitAction(Action):
    def __init__(self, circuit: Circuit, event: Event):
        self.circuit_id = circuit.id
        self.inverted = event.inverted

    def make_callback(
        self,
        values: MutableSequence[int],
        display_values: MutableSequence[int],
        read_bit: Callable[[], int],
        write_bit: Callable[[int], None],
    ) -> Callable | None:
        circuit_id = self.circuit_id

        if self.inverted:

            def callback():
                values[circuit_id] ^= 1
                if values[circuit_id]:
                    values[circuit_id] = read_bit()

        else:

            def callback():
                if values[circuit_id]:
                    values[circuit_id] = read_bit()

        return callback


class WriteBitAction(Action):
    def __init__(self, circuit: Circuit, event: Event):
        self.circuit_id = circuit.id
        self.value = 1 if event.event_type == EventType.WRITE_ONE else 0
        self.inverted = event.inverted

    def make_callback(
        self,
        values: MutableSequence[int],
        display_values: MutableSequence[int],
        read_bit: Callable[[], int],
        write_bit: Callable[[int], None],
    ) -> Callable | None:
        circuit_id = self.circuit_id
        value = self.value

        if self.inverted:

            def callback():
                values[circuit_id] ^= 1
                if values[circuit_id]:
                    write_bit(value)

        else:

            def callback():
                if values[circuit_id]:
                    write_bit(value)

        return callback


class ConditionalClearAction(Action):
    def __init__(
        self,
        circuit: Circuit,
        event: Event,
        control_circuit: Circuit,
        control_event: Event,
    ):
        self.circuit_id = circuit.id
        self.control_circuit_id = control_circuit.id
        self.inverted = event.inverted
        self.control_circuit_inverted = control_event.inverted

    def make_callback(
        self,
        values: MutableSequence[int],
        display_values: MutableSequence[int],
        read_bit: Callable[[], int],
        write_bit: Callable[[int], None],
    ) -> Callable | None:
        circuit_id = self.circuit_id
        control_circuit_id = self.control_circuit_id

        if self.inverted and self.control_circuit_inverted:

            def callback():
                values[circuit_id] ^= 1
                values[control_circuit_id] ^= 1
                values[circuit_id] &= values[control_circuit_id]

        elif self.inverted and not self.control_circuit_inverted:

            def callback():
                values[circuit_id] ^= 1
                values[circuit_id] &= values[control_circuit_id]

        elif not self.inverted and self.control_circuit_inverted:

            def callback():
                values[control_circuit_id] ^= 1
                values[circuit_id] &= values[control_circuit_id]

        elif not self.inverted and not self.control_circuit_inverted:

            def callback():
                values[circuit_id] &= values[control_circuit_id]

        else:
            assert False

        return callback


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


def build_display_info(grid: list[dict[int, str]], position: Position) -> Display:
    c = get_character(grid, position)
    if c in DISPLAYS:
        initial_value = c == DISPLAY_ON
        return Display({position}, DISPLAY_OFF, DISPLAY_ON, initial_value)
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

    return Display(positions, GRID_OFF, GRID_ON, initial_value)


def build_circuit(
    grid: list[dict[int, str]], marble: Marble, circuit_id: int
) -> Circuit:
    current_p = marble.p
    current_d = marble.d

    positions: list[Position] = []
    invertors: list[int] = []
    events: list[Event] = [Event(EventType.START, 0, False)]
    displays: dict[Position, Display] = {}
    inverted: bool = False

    # Loop over steps
    for i in itertools.count():
        positions.append(current_p)

        # Check character
        c = get_character(grid, current_p)

        # Get new direction
        directions = get_directions(grid, current_p)
        if len(directions) == 2:
            d1, d2 = directions
            assert current_d.opposite in (d1, d2)
            new_d = d2 if current_d.opposite == d1 else d1
        elif len(directions) == 4 or c in GRIDS:
            new_d = current_d
        else:
            assert False

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
    random.shuffle(marbles)
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
) -> tuple[bool, bool, Action | None,]:
    event = circuit.events[event_index]
    event_type = event.event_type

    if event_type == EventType.START:
        return False, False, StartAction(circuit, event)
    elif event_type == EventType.UNCONDITIONAL_CLEAR:
        return False, False, UnconditionalClearAction(circuit, event)
    elif event_type == EventType.EXIT:
        return False, False, ExitAction(circuit, event)
    elif event_type == EventType.DISPLAY:
        return False, False, DisplayAction(circuit, event)
    elif event_type == EventType.READ_BIT:
        return False, False, ReadBitAction(circuit, event)
    elif event_type in (EventType.WRITE_ZERO, EventType.WRITE_ONE):
        return False, False, WriteBitAction(circuit, event)
    elif event_type in (EventType.CONDITIONAL_CLEAR, EventType.CONTROL):
        assert event.neighbor is not None
        if event.neighbor in waiting_circuits:
            other_circuit, other_event_index = waiting_circuits[event.neighbor]
            other_event = other_circuit.events[other_event_index]
            if event_type == EventType.CONTROL:
                circuit, other_circuit = other_circuit, circuit
                event, other_event = other_event, event
            return (
                False,
                True,
                ConditionalClearAction(circuit, event, other_circuit, other_event),
            )
        else:
            return True, False, None

    else:
        assert False


class TickInfo:
    def __init__(self, last_tick_info: TickInfo | None, tick_enter: int, waiting: bool):
        self.waiting = waiting
        self.tick_enter = tick_enter
        self.last_tick_info = last_tick_info
        self.tick_exit = tick_enter + 1 if not waiting else None

    def get_cycle(self) -> int | None:
        if self.tick_exit is None:
            return None
        if self.last_tick_info is None:
            return None
        if self.last_tick_info.tick_exit is None:
            return None
        return self.tick_exit - self.last_tick_info.tick_exit


def analyze_group(
    group: list[Circuit], group_id: int, display_counter: Iterator[int]
) -> Group:
    # Prepare structures
    counter = iter(itertools.count(0))
    cycle_count: dict[int, int] = {}
    waiting_infos: set[tuple[int, int, int]] = set()
    tick_infos: dict[Circuit, TickInfo] = {}
    waiting_circuits: dict[Position, tuple[Circuit, int]] = {}
    event_index_to_ticks: dict[Circuit, dict[int, TickInfo]] = {}
    tick_to_event_indexes: dict[Circuit, list[tuple[int, int, bool]]] = {}
    priority_queue: list[tuple[int, int, int, Circuit]] = []
    actions: list[tuple[int, Action]] = []
    displays: list[Display] = []

    # Prepare structures
    for circuit in group:
        event_index_to_ticks[circuit] = {}
        tick_to_event_indexes[circuit] = []
        priority_queue.append((0, 0, next(counter), circuit))
        for display in circuit.displays.values():
            display.id = next(display_counter)
            displays.append(display)

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

            # Process next_event
            waiting, restore, action = process_event(
                current_circuit, event_index, waiting_circuits
            )

            # Update progress bar
            progress_bar.n = current_tick
            progress_bar.update(0)

            # Save tick info
            current_event_index_to_ticks = event_index_to_ticks[current_circuit]
            last_tick_info = current_event_index_to_ticks.get(event_index)
            current_tick_info = TickInfo(last_tick_info, current_tick, waiting)
            current_event_index_to_ticks[event_index] = current_tick_info

            # Update data for restored circuit
            if restore:
                # Find restored circuit
                neighbor = current_circuit.events[event_index].neighbor
                assert neighbor is not None
                restored_circuit, _ = waiting_circuits[neighbor]
                # Update tick exit
                tick_infos[restored_circuit].tick_exit = current_tick + 1
                # Get last tick info
                restored_last_tick_info = tick_infos[restored_circuit].last_tick_info
                if restored_last_tick_info is not None:
                    # Remove from waiting dicts
                    assert restored_last_tick_info.tick_exit is not None
                    item = (
                        restored_circuit.id,
                        restored_last_tick_info.tick_enter,
                        restored_last_tick_info.tick_exit,
                    )
                    waiting_infos.remove(item)
                    # Add to cycle dict
                    cycle = tick_infos[restored_circuit].get_cycle()
                    assert cycle is not None
                    cycle_count.setdefault(cycle, 0)
                    cycle_count[cycle] += 1

            # Get last cycle
            if current_circuit in tick_infos:
                last_cycle = tick_infos[current_circuit].get_cycle()
                # Remove from cycles
                if last_cycle is not None:
                    cycle_count[last_cycle] -= 1
                    if cycle_count[last_cycle] == 0:
                        del cycle_count[last_cycle]
            # Set current tick info
            tick_infos[current_circuit] = current_tick_info
            # Update waiting dicts if waiting
            if last_tick_info is not None and waiting:
                assert last_tick_info.tick_exit is not None
                item = (
                    current_circuit.id,
                    last_tick_info.tick_enter,
                    last_tick_info.tick_exit,
                )
                waiting_infos.add(item)
            # Update cycle count if not waiting
            elif last_tick_info:
                cycle = current_tick_info.get_cycle()
                assert cycle is not None
                cycle_count.setdefault(cycle, 0)
                cycle_count[cycle] += 1

            # Check for global cycle
            if len(cycle_count) == 1:
                first_key, first_value = next(iter(cycle_count.items()))
                cycle = first_key
                reference = current_tick - cycle
                if len(waiting_infos) + first_value == len(group) and all(
                    tick_enter <= reference < tick_exit
                    for _, tick_enter, tick_exit in waiting_infos
                ):
                    start = current_tick - cycle
                    return Group(
                        group,
                        displays,
                        start,
                        cycle,
                        tick_to_event_indexes,
                        actions,
                    )

            # Save action
            if action is not None:
                actions.append((current_tick, action))

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


def analyze_groups(raw_groups: list[list[Circuit]]) -> list[Group]:
    display_counter = itertools.count(0)
    return [
        analyze_group(group, i, display_counter) for i, group in enumerate(raw_groups)
    ]


# Screen info


class CircuitInfo:
    def __init__(
        self,
        id: int,
        cycle_start: int,
        cycle_length: int,
        min_x: int,
        max_x: int,
        position_indexes: list[int],
        event_inverted: list[int],
        invertors: list[int],
        x_positions: list[int],
        y_positions: list[int],
        event_ticks: list[int],
        event_indexes: list[int],
        event_waiting: list[int],
    ):
        self.id = id
        self.cycle_start = cycle_start
        self.cycle_length = cycle_length
        self.min_x = min_x
        self.max_x = max_x
        self.position_indexes = position_indexes
        self.event_inverted = event_inverted
        self.invertors = invertors
        self.x_positions = x_positions
        self.y_positions = y_positions
        self.event_ticks = event_ticks
        self.event_indexes = event_indexes
        self.event_waiting = event_waiting

    @property
    def span(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def max_span(self) -> int:
        return 2 ** (self.span - 1).bit_length()

    def dump(self):
        return [
            self.id,
            self.cycle_start,
            self.cycle_length,
            self.min_x,
            self.max_x,
            self.position_indexes,
            self.event_inverted,
            self.invertors,
            self.x_positions,
            self.y_positions,
            self.event_ticks,
            self.event_indexes,
            self.event_waiting,
        ]

    @classmethod
    def load(cls, args: list):
        return cls(*args)


class DisplayInfo:
    def __init__(
        self,
        id: int,
        min_x: int,
        max_x: int,
        x_positions: list[int],
        y_positions: list[int],
        off_char: str,
        on_char: str,
    ):
        self.id = id
        self.min_x = min_x
        self.max_x = max_x
        self.x_positions = x_positions
        self.y_positions = y_positions
        self.off_char = off_char
        self.on_char = on_char

    @property
    def span(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def max_span(self) -> int:
        return 2 ** (self.span - 1).bit_length()

    def dump(self):
        return [
            self.id,
            self.min_x,
            self.max_x,
            self.x_positions,
            self.y_positions,
            self.off_char,
            self.on_char,
        ]

    @classmethod
    def load(cls, args: list):
        return cls(*args)


class ScreenInfo:
    def __init__(
        self, grid: Grid, circuits: list[CircuitInfo], displays: list[DisplayInfo]
    ):
        self.grid = grid
        self.sorted_circuits: dict[int, list[tuple[int, int, CircuitInfo]]] = {}
        self.sorted_displays: dict[int, list[tuple[int, int, DisplayInfo]]] = {}

        for circuit in circuits:
            self.sorted_circuits.setdefault(circuit.max_span, [])
            circuit_item = (circuit.min_x, circuit.id, circuit)
            self.sorted_circuits[circuit.max_span].append(circuit_item)

        for display in displays:
            self.sorted_displays.setdefault(display.max_span, [])
            display_item = (display.min_x, display.id, display)
            self.sorted_displays[display.max_span].append(display_item)

        for circuit_value in self.sorted_circuits.values():
            circuit_value.sort()
        for display_value in self.sorted_displays.values():
            display_value.sort()

    @functools.lru_cache(maxsize=16)
    def get_circuits_in_window(self, min_x: int, max_x: int) -> list[CircuitInfo]:
        result = []
        # Get circuits in frame
        for max_span, circuit_ids in self.sorted_circuits.items():
            index = bisect.bisect_left(circuit_ids, (min_x - max_span, -1, None))
            for _, _, circuit in itertools.islice(circuit_ids, index, None, None):
                if circuit.min_x > max_x:
                    break
                if min_x <= circuit.max_x:
                    result.append(circuit)
        return result

    @functools.lru_cache(maxsize=16)
    def get_displays_in_window(self, min_x: int, max_x: int) -> list[DisplayInfo]:
        result = []
        # Get circuits in frame
        for max_span, displays in self.sorted_displays.items():
            index = bisect.bisect_left(displays, (min_x - max_span, -1, None))
            for _, _, display in itertools.islice(displays, index, None, None):
                if display.min_x > max_x:
                    break
                if min_x <= display.max_x:
                    result.append(display)
        return result

    def get_marbles_in_window(
        self, tick: int, values: MutableSequence[int], min_x: int, max_x: int
    ) -> set[tuple[Position, str]]:
        result = set()

        # Loop over circuits
        for circuit in self.get_circuits_in_window(min_x, max_x):

            # Normalize tick
            if tick >= circuit.cycle_start:
                tick = (
                    (tick - circuit.cycle_start) % circuit.cycle_length
                ) + circuit.cycle_start

            # Get event info
            index = bisect.bisect(circuit.event_ticks, tick) - 1
            previous_tick = circuit.event_ticks[index]
            event_index = circuit.event_indexes[index]
            waiting = circuit.event_waiting[index]
            position_index = circuit.position_indexes[event_index]

            # Marble is waiting
            if waiting:
                inverted = circuit.event_inverted[event_index]
                position_x = circuit.x_positions[position_index]
                position_y = circuit.y_positions[position_index]
                position = Position(position_x, position_y)
                value = values[circuit.id] ^ inverted

            # Marble is running
            else:
                i_start = bisect.bisect_left(circuit.invertors, position_index)
                position_index += tick - previous_tick
                i_stop = bisect.bisect_left(circuit.invertors, position_index)
                position_index %= len(circuit.x_positions)
                position_x = circuit.x_positions[position_index]
                position_y = circuit.y_positions[position_index]
                position = Position(position_x, position_y)
                invert = (i_stop - i_start) % 2
                value = values[circuit.id] ^ invert

            # Add marble to the result
            char = MARBLE_UPPER if value else MARBLE_LOWER
            result.add((position, char))

        return result

    def get_display_values_in_window(
        self, tick: int, values: MutableSequence[int], min_x: int, max_x: int
    ) -> set[tuple[Position, str]]:
        result = set()
        for display in self.get_displays_in_window(min_x, max_x):
            char = display.on_char if values[display.id] else display.off_char
            for x, y in zip(display.x_positions, display.y_positions):
                position = Position(x, y)
                result.add((position, char))
        return result


# Simulation

if TYPE_CHECKING:
    CallbackItem = tuple[int, int, Callable[[], None]]


class GroupCallbacks:
    def __init__(self, simulation: Simulation, group: Group):
        self.cycle_start = group.cycle_start
        self.cycle_length = group.cycle_length

        self.init_callbacks = []
        self.cycle_callbacks = []

        # Make callbacks
        for i, (tick, action) in enumerate(group.actions):
            callback = action.make_callback(
                simulation.values,
                simulation.display_values,
                simulation.io.read_bit,
                simulation.io.write_bit,
            )
            if tick < group.cycle_start:
                item = tick, i, callback
                self.init_callbacks.append(item)
            else:
                item = tick - group.cycle_start, i, callback
                self.cycle_callbacks.append(item)

        # Add extra init and cycle callbacks
        (tick, i, callback) = self.cycle_callbacks[0]
        self.init_callbacks.append((tick + group.cycle_start, i, callback))
        self.cycle_callbacks.append((tick + group.cycle_length, i, callback))

    def run_until(self, start: int, stop: int, deadline: float) -> tuple[bool, int]:
        assert start < stop
        if stop <= self.cycle_start:
            return self.run_callbacks(
                self.init_callbacks, start, min(stop, self.cycle_start), deadline
            )
        if start < self.cycle_start:
            interrupted, next_tick = self.run_callbacks(
                self.init_callbacks, start, min(stop, self.cycle_start), deadline
            )
            if interrupted:
                return interrupted, next_tick
        interrupted, next_tick = self.run_cycle_callbacks(
            max(start - self.cycle_start, 0), stop - self.cycle_start, deadline
        )
        return interrupted, self.cycle_start + next_tick

    def run_cycle_callbacks(
        self, start: int, stop: int, deadline: float
    ) -> tuple[bool, int]:
        # Shift to first cycle
        if start > self.cycle_length:
            diff = (start // self.cycle_length) * self.cycle_length
            interrupted, next_tick = self.run_cycle_callbacks(
                start - diff, stop - diff, deadline
            )
            return interrupted, diff + next_tick
        # Stop before second cycle
        if stop <= self.cycle_length:
            return self.run_callbacks(self.cycle_callbacks, start, stop, deadline)
        # Complete current cycle
        if start != 0:
            interrupted, next_tick = self.run_callbacks(
                self.cycle_callbacks, start, self.cycle_length, deadline
            )
            if interrupted:
                return interrupted, next_tick
            interrupted, next_tick = self.run_cycle_callbacks(
                0, stop - self.cycle_length, deadline
            )
            return interrupted, self.cycle_length + next_tick
        # Run cycles
        cycles, stop = divmod(stop, self.cycle_length)
        for i in range(cycles):
            interrupted, next_tick = self.run_cycle(deadline)
            if interrupted:
                return interrupted, i * self.cycle_length + next_tick
        # Complete the last run
        interrupted, next_tick = self.run_callbacks(
            self.cycle_callbacks, 0, stop, deadline
        )
        return interrupted, cycles * self.cycle_length + next_tick

    def run_callbacks(
        self, callbacks: list[CallbackItem], start: int, stop: int, deadline: float
    ) -> tuple[bool, int]:
        if time.time() > deadline:
            return True, start
        assert callbacks
        start_index = bisect.bisect(callbacks, (start, -1))
        stop_index = bisect.bisect(callbacks, (stop, -1))
        # Run callbacks
        for index in range(start_index, stop_index):
            _, _, callback = callbacks[index]
            callback()
        return False, callbacks[stop_index][0]

    def run_cycle(self, deadline: float) -> tuple[bool, int]:
        cycle_callbacks = self.cycle_callbacks
        stop = len(self.cycle_callbacks) - 1
        interrupt = False
        previous_tick = None
        for i in range(0, stop):
            tick, _, callback = cycle_callbacks[i]
            if i % 100 == 0 and time.time() > deadline:
                interrupt = True
            if interrupt and tick != previous_tick:
                return True, tick
            callback()
            previous_tick = tick
        return False, self.cycle_length


class Simulation:
    def __init__(self, groups: list[Group], io: SimulationIO):
        self.groups = groups
        self.io = io

        # Ticks
        self.current_tick = 0

        # Global cycle
        self.cycle_length = math.lcm(*(group.cycle_length for group in groups))
        self.cycle_start = max(group.cycle_start for group in groups)

        # Initialize values
        self.values = array.array("B", [0]) * sum(
            len(group.circuits) for group in groups
        )
        self.display_values = array.array("B", [0]) * sum(
            len(group.displays) for group in groups
        )
        for group in groups:
            for circuit in group.circuits:
                self.values[circuit.id] = circuit.init_marble.upper
                self.values[circuit.id] ^= circuit.events[0].inverted
            for display in group.displays:
                assert display.id is not None
                self.display_values[display.id] = display.initial_value

        # Create callbacks
        self.callbacks = [GroupCallbacks(self, group) for group in groups]

    def run_until(self, target: int, deadline: float):
        # Not yet
        if target <= self.current_tick:
            return
        # Single group
        if len(self.callbacks) == 1:
            (group_callbacks,) = self.callbacks
            interrupted, next_tick = group_callbacks.run_until(
                self.current_tick, target, deadline
            )
            self.current_tick = min(next_tick, target)
            return
        # Prepare priority queue
        queue: list[tuple[int, int, GroupCallbacks]] = []
        for i, group_callbacks in enumerate(self.callbacks):
            item = (self.current_tick, i, group_callbacks)
            heapq.heappush(queue, item)
        # Run priority queue
        while queue[0][0] < target:
            group_starting, i, group_callbacks = heapq.heappop(queue)
            group_target = min(target, queue[0][0] + 1)
            interrupted, next_tick = group_callbacks.run_until(
                group_starting, group_target, deadline
            )
            item = (next_tick, i, group_callbacks)
            heapq.heappush(queue, item)
            if interrupted:
                break
        # Set current tick
        self.current_tick = min(queue[0][0], target)


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
        # Return
        print("\r", end="", flush=True)


class SimulationIO:
    def __init__(self, input_stream: IO[bytes], output_stream: IO[bytes]):
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.input_bits: list[int] = []
        self.output_bits: list[int] = []

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


class ScreenDisplay:
    def __init__(
        self, screen_info: ScreenInfo, speed: float, fps: float, simulation: Simulation
    ):
        self.fps = fps
        self.screen_info = screen_info
        self.speed = speed
        self.io = simulation.io
        self.simulation = simulation

        self.x_offset = 0
        self.y_offset = 0
        self.changed = True
        self.termsize = os.terminal_size((0, 0))
        self.old_chars: set[tuple[Position, str]] = set()

        self.frame: int
        self.start_time: float
        self.start_frame: int

    def run(self):
        # Complete tick 0
        self.simulation.run_until(1, time.time() + 1)

        # Save information
        self.start_frame = 0
        self.start_time = time.time()
        start_simulation_tick = self.simulation.current_tick

        # Iterator over steps
        for self.frame in itertools.count(1):
            previous_time = time.time()
            previous_simulation_tick = self.simulation.current_tick

            # Control
            if self.check_stdin():
                self.start_time = time.time()
                self.start_frame = self.frame
                start_simulation_tick = self.simulation.current_tick
            step1 = time.time() - previous_time

            # Show simulation
            self.show()
            step2 = time.time() - previous_time - step1

            # Run simulation
            delta_frame = self.frame - self.start_frame
            new_simulation_tick = start_simulation_tick + int(
                delta_frame * self.speed / self.fps
            )
            deadline = self.start_time + delta_frame / self.fps
            self.simulation.run_until(new_simulation_tick, deadline)
            step3 = time.time() - previous_time - step1 - step2

            # Wait for the next tick
            delta = deadline - time.time()
            if delta > 0:
                time.sleep(delta)

            step4 = time.time() - previous_time - step1 - step2 - step3

            actual_fps = 1 / (time.time() - previous_time)
            actual_speed_per_frame = (
                self.simulation.current_tick - previous_simulation_tick
            )
            steps = step1, step2, step3, step4
            cycle_count = (
                self.simulation.current_tick - self.simulation.cycle_start
            ) // self.simulation.cycle_length
            self.print_status_bar(
                actual_fps,
                actual_speed_per_frame,
                steps,
                cycle_count,
                self.simulation.cycle_length,
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
        info += f"  | Tick per frame: {speed_per_frame:8d}"
        info += f"  | Speed: {int(speed_per_frame * fps):8d}"
        info += f"  | Frame: {self.frame: 8d}"
        sum_steps = sum(steps, 0)
        stdin, show, run, sleep = [int(round(s * 100 / sum_steps)) for s in steps]
        info += f"  | Display: {show:3d} % CPU"
        info += f"  | Simulation: {run:3d} % CPU"
        info += f"  | Current cycle: {cycle_count:4d} ({cycle_length:5d} ticks)"
        info += f"  | Cycle speed: {speed_per_frame * fps / self.simulation.cycle_length:4.1f}"
        info += f"  | Tick: {self.simulation.current_tick:8d}"
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

    def show(self) -> None:
        tick = self.simulation.current_tick - 1
        # Update termsize
        termsize = os.get_terminal_size()
        if termsize != self.termsize:
            self.termsize = termsize
            self.changed = True
        # Current window has changed
        if self.changed:
            self.changed = False
            print(self.draw_grid(), end="")
            self.old_chars = set()
        # Get current window
        min_x = -self.x_offset
        max_x = min_x + self.termsize.lines - 2
        marbles = self.screen_info.get_marbles_in_window(
            tick, self.simulation.values, min_x, max_x
        )
        displays = self.screen_info.get_display_values_in_window(
            tick, self.simulation.display_values, min_x, max_x
        )
        # Remove marbles from ON grid and OFF grid from marbles
        marbles -= {
            (position, marble)
            for position, char in displays
            if char == GRID_ON
            for marble in MARBLES
        }
        displays -= {(position, GRID_OFF) for position, char in marbles}
        # Set operations
        new_chars = marbles | displays
        unchanged = new_chars & self.old_chars
        to_clear = self.old_chars - unchanged
        to_draw = new_chars - unchanged
        # Update marbles
        print(self.clear_chars(to_clear), end="")
        print(self.draw_chars(to_draw), end="", flush=True)
        # Save marbles
        self.old_chars = marbles

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
            if not 0 <= i < len(self.screen_info.grid):
                continue
            row = self.screen_info.grid[i]
            for j, char in row.items():
                result.append(self.draw_char(i, j, char))
        return "".join(result)

    def clear_chars(self, chars: set[tuple[Position, str]]) -> str:
        result = []
        for position, _ in chars:
            char = get_character(self.screen_info.grid, position)
            result.append(self.draw_char(position.x, position.y, char))
        return "".join(result)

    def draw_chars(
        self,
        marbles: set[tuple[Position, str]],
    ) -> str:
        result = []
        for position, value in marbles:
            result.append(self.draw_char(position.x, position.y, value))
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


def extract_info(
    path: pathlib.Path, groups: list[Group], write_cache: bool = True
) -> tuple[list[CircuitInfo], list[DisplayInfo]]:
    import msgpack

    circuit_mapping: dict[int, CircuitInfo] = {}
    for i, group in enumerate(groups):
        for circuit in tqdm.tqdm(
            group.circuits,
            desc=f"Extracting display info for group {i}",
            unit=" circuits",
            colour="green",
        ):
            ticks_to_event_indexes = group.ticks_to_event_indexes[circuit]
            circuit_mapping[circuit.id] = circuit.extract_info(
                group.cycle_start, group.cycle_length, ticks_to_event_indexes
            )

    display_mapping: dict[int, DisplayInfo] = {}
    for i, group in enumerate(groups):
        for display in tqdm.tqdm(
            group.displays,
            desc=f"Extracting display info for group {i}",
            unit=" displays",
            colour="green",
        ):
            assert display.id is not None
            display_mapping[display.id] = display.extract_info()

    circuit_infos = [circuit_mapping[i] for i in range(len(circuit_mapping))]
    display_infos = [display_mapping[i] for i in range(len(display_mapping))]

    if write_cache:
        packer = msgpack.Packer()
        with path.open("wb") as file:
            file.write(packer.pack((len(circuit_infos), len(display_infos))))
            for circuit_info in tqdm.tqdm(
                circuit_infos,
                desc="Dumping display info",
                unit=" circuits",
                colour="green",
            ):
                file.write(packer.pack(circuit_info.dump()))
            for display_info in tqdm.tqdm(
                display_infos,
                desc="Dumping display info",
                unit=" displays",
                colour="green",
            ):
                file.write(packer.pack(display_info.dump()))

    return circuit_infos, display_infos


def main(
    path: pathlib.Path,
    speed: float,
    fps: float,
    check_cache: bool,
    input_stream: IO[bytes] | None = None,
    output_stream: IO[bytes] | None = None,
):
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
        itertools.count(0)
        groups = analyze_groups(raw_groups)
        circuit_info, display_info = extract_info(cache_path, groups)

    # Input/Output
    if input_stream is None and not sys.stdin.isatty():
        input_stream = sys.stdin.buffer
    if output_stream is None and not sys.stderr.isatty():
        output_stream = sys.stderr.buffer
    if input_stream is None:
        input_stream = io.BytesIO()
    if output_stream is None:
        output_stream = io.BytesIO()

    # Simulation
    try:
        with drawing_context():

            # Run
            input_output = SimulationIO(input_stream, output_stream)
            simulation = Simulation(groups, input_output)
            screen_info = ScreenInfo(grid, circuit_info, display_info)
            display = ScreenDisplay(screen_info, speed, fps, simulation)
            display.run()

    # Ignore EOF
    except EOFError:
        pass
    # Output captured IO if necessary
    finally:
        if isinstance(output_stream, io.BytesIO):
            sys.stdout.buffer.write(output_stream.getvalue())
        print()


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
        False,  # not namespace.no_cache,
        namespace.input,
        namespace.output,
    )
