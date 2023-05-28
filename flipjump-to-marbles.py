#!/usr/bin/env python3
"""
usage: flipjump-to-marbles.py [-h] [--width WIDTH] [--size SIZE] [program]

Convert flip jump machine code (in version 1) to a marble program

positional arguments:
  program        path to a .fjm file in version 1

options:
  -h, --help     show this help message and exit
  --width WIDTH  word size in bits (when no program is provided)
  --size SIZE    program size in words (when no program is provided)
"""
from __future__ import annotations

import io
import array
import struct
import argparse

EMPTY = "   "
VERTICAL = "║  "
HORIZONTAL = "═══"
CROSSING = "╬══"
VERTICAL_INVERTED = "┃  "
ZERO_ADDRESS = VERTICAL_INVERTED
ONE_ADDRESS = VERTICAL
DECODER_0 = "╚═╗"
DECODER_1 = "╔╤╝"
DECODER_2 = "╬╛═"
DECODER_2_INVERTED = "╬╛━"
DECODER_OUT_0 = "╚═╗"
DECODER_OUT_1 = "╔╕╝"
DECODER_OUT_2 = "╬╧═"
DECODER_1_INVERTED = "╚━╗"
VERTICAL_WITH_CLEAR = "║○ "
EMPTY_WITH_CLEAR = " ○ "
HORIZONTAL_WITH_UPPER_MARBLE = "═●═"

WEST_SOUTH_CORNER = "╗  "
SOUTH_EAST_CORNER_WITH_MARBLE = "○╛━"
SOUTH_EAST_CORNER_WITHOUT_MARBLE = "╔══"
EAST_NORTH_CORNER_WITH_MARBLE = "○╛━"
EAST_NORTH_CORNER_WITHOUT_MARBLE = "╚══"
NORTH_WEST_CORNER = "╝  "
DISPLAY = "╟□ "
MARBLE = "○  "
VERTICAL_CLEAR = "╓○ "

MEMORY_0 = "╔══════════╤═╗ "
MEMORY_1 = "║╔━╒═══╗┼┼┼┼┼║ "
MEMORY_2 = "║║ ○╔━╗║┼┼┼┼┼║ "
MEMORY_3 = "║●╕━╬╗╚╬╗ ○XX║ "
MEMORY_4 = "╚═╧╕╝╚╤╝╚╤╛━╕○ "
MEMORY_5 = "═══╧═━╛═━╛══╧══"
MEMORY_WIDTH = 5

MEMORY_WRITE_ZERO_0 = "  ┏━━━━━━━━━┓  "
MEMORY_WRITE_ZERO_1 = "  ┃   IO    ┃  "
MEMORY_WRITE_ZERO_2 = "  ┃ Write 0 ┃  "
MEMORY_WRITE_ZERO_3 = "  ┗━━━━━━━━━┛  "
MEMORY_WRITE_ZERO_4 = "   ◇     ○   ○ "
MEMORY_WRITE_ZERO_5 = "═══╧═════╛═════"

MEMORY_WRITE_ONE_0 = "  ┏━━━━━━━━━┓  "
MEMORY_WRITE_ONE_1 = "  ┃   IO    ┃  "
MEMORY_WRITE_ONE_2 = "  ┃ Write 1 ┃  "
MEMORY_WRITE_ONE_3 = "  ┗━━━━━━━━━┛  "
MEMORY_WRITE_ONE_4 = "   ◆     ○   ○ "
MEMORY_WRITE_ONE_5 = "═══╧═════╛═════"

MEMORY_READ_0 = "  ┏━━━━━━━━━┓  "
MEMORY_READ_1 = "  ┃   IO    ┃  "
MEMORY_READ_2 = "  ┃  Read   ┃  "
MEMORY_READ_3 = "  ┗━━━━━━━━━┛  "
MEMORY_READ_4 = "         ○   ○ "
MEMORY_READ_5 = "═════════╛━════"

READ_0 = " ◇ "
READ_1 = "━╛━"

CMP_HEADER_0 = "╔━╒╗  "
CMP_HEADER_1 = "● ○║  "
CMP_LEFT_0 = "║○ "
CMP_LEFT_1 = "○╛━"
CMP_LEFT_2 = "╔╕━"
CMP_LEFT_3 = "║○ "
CMP_DECODER_0 = "╚╤╗"
CMP_DECODER_1 = "═╛╬"
CMP_DECODER_2 = "═╕╬"
CMP_DECODER_3 = "╔╧╝"
CMP_BLOCK_0 = "═════════╗  "
CMP_BLOCK_1 = "╔═══━╒╗  ║  "
CMP_BLOCK_2 = "●╕═╗ ○║  ║  "
CMP_BLOCK_3 = "═╧╕╬══╬━╤╝  "
CMP_BLOCK_4 = "══╧╬━╕╬━╛━═╗"
CMP_BLOCK_5 = "   ╚━╧╝    ╟"
CMP_BLOCK_6 = "═══════════╝"
CMP_FOOTER_0 = "║ ☒║  "
CMP_FOOTER_1 = "╚═╧╝  "
CMP_CHECK_LOOP = "╜  "
CMP_WIDTH = 4


def main(
    width: int | None = 8, size: int | None = None, program: io.IOBase | None = None
):

    words = None
    if program is None and width is None:
        width = 8
    if program is not None:
        header = program.read(0x40)
        program_string = program.read()
        (
            magic,
            word_size,
            version,
            segment_num,
            flags,
            reserved,
            segment_start,
            segment_length,
            data_start,
            data_length,
        ) = struct.unpack("<HHQQQIQQQQ", header)
        assert magic == 0x4A46
        assert version == 1
        assert segment_num == 1
        assert segment_start == 0
        assert data_start == 0
        assert data_length == segment_length
        assert width in (None, word_size)
        width = word_size
        assert size in (None, data_length)
        size = data_length

        words = array.array("_BHLQ"[word_size // 8], program_string)
        assert len(words) == size

    # Make sure width is a power of 2
    assert width is not None
    assert width >= 4
    assert bin(width).count("1") == 1
    word = width
    instruction = 2 * width
    max_memory = 2**width

    if size is None:
        size = max_memory // width
    assert size is not None
    assert size <= max_memory // width
    assert size % 2 == 0

    address_size = word
    low_address_size = (instruction - 1).bit_length()
    high_address_size = address_size - low_address_size

    j_size = high_address_size
    t_size = address_size
    r_size = t_size + j_size

    # Program name
    program_name = getattr(program, "name", "")
    title = f"{width}-bit FlipJump computer"
    if program_name:
        title += f" programmed with `{program_name}` ({size} words)"
    else:
        title += f" with {size} words of memory"
    print(title)
    print("━" * len(title))
    print()
    print()

    # Header for J
    for i in range(j_size):
        row = (
            EMPTY
            + VERTICAL * i
            + SOUTH_EAST_CORNER_WITHOUT_MARBLE
            + HORIZONTAL * (j_size - 1 - i)
            + HORIZONTAL
            + HORIZONTAL * t_size
            + HORIZONTAL * MEMORY_WIDTH
            + HORIZONTAL
            + HORIZONTAL * t_size
            + HORIZONTAL
            + HORIZONTAL * (j_size - 1 - i)
            + WEST_SOUTH_CORNER
            + VERTICAL * i
        )
        print(row)

    # Header for T
    for i in range(t_size):
        row = (
            EMPTY
            + VERTICAL * j_size
            + EMPTY
            + VERTICAL * i
            + SOUTH_EAST_CORNER_WITHOUT_MARBLE
            + HORIZONTAL * (t_size - 1 - i)
            + HORIZONTAL * MEMORY_WIDTH
            + HORIZONTAL
            + HORIZONTAL * (t_size - 1 - i)
            + WEST_SOUTH_CORNER
            + VERTICAL * i
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

    # Annotations
    row = (
        EMPTY
        + VERTICAL.replace("  ", "J ") * j_size
        + EMPTY
        + VERTICAL.replace("  ", "F ") * t_size
        + EMPTY * MEMORY_WIDTH
        + EMPTY
        + VERTICAL.replace("  ", "F ") * t_size
        + EMPTY
        + VERTICAL.replace("  ", "J ") * j_size
    )
    print(row)

    row = (
        EMPTY
        + "".join(
            VERTICAL.replace("  ", f"{d}".ljust(2)) for d in reversed(range(j_size))
        )
        + EMPTY
        + "".join(
            VERTICAL.replace("  ", f"{d}".ljust(2)) for d in reversed(range(t_size))
        )
        + EMPTY * MEMORY_WIDTH
        + EMPTY
        + "".join(VERTICAL.replace("  ", f"{d}".ljust(2)) for d in range(t_size))
        + EMPTY
        + "".join(VERTICAL.replace("  ", f"{d}".ljust(2)) for d in range(j_size))
    )
    print(row)

    # Display
    row = (
        EMPTY
        + DISPLAY * j_size
        + EMPTY
        + DISPLAY * t_size
        + EMPTY * MEMORY_WIDTH
        + EMPTY
        + DISPLAY * t_size
        + EMPTY
        + DISPLAY * j_size
    )
    print(row)

    # Initial marbles
    row = (
        EMPTY
        + MARBLE * j_size
        + EMPTY
        + MARBLE * t_size
        + EMPTY * MEMORY_WIDTH
        + EMPTY
        + ZERO_ADDRESS * t_size
        + EMPTY
        + ZERO_ADDRESS * j_size
    )
    print(row)

    # Comparator header
    for cmp_header in (CMP_HEADER_0, CMP_HEADER_1):
        row = (
            EMPTY
            + VERTICAL * j_size
            + EMPTY
            + VERTICAL * t_size
            + EMPTY * CMP_WIDTH
            + cmp_header
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

    # Iterate over comparator blocks
    for i in range(j_size):

        # Comparator block, row 0
        row = (
            SOUTH_EAST_CORNER_WITHOUT_MARBLE
            + CROSSING * j_size
            + HORIZONTAL
            + CROSSING * t_size
            + CMP_BLOCK_0
            + VERTICAL
            + VERTICAL
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

        # Comparator block, row 1
        row = (
            VERTICAL
            + VERTICAL * j_size
            + EMPTY
            + VERTICAL * t_size
            + CMP_BLOCK_1
            + VERTICAL
            + VERTICAL
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

        # Comparator block, row 2
        row = (
            CMP_LEFT_0
            + VERTICAL * i
            + CMP_DECODER_0
            + VERTICAL * (j_size - i - 1)
            + EMPTY
            + VERTICAL * t_size
            + CMP_BLOCK_2
            + VERTICAL
            + VERTICAL
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

        # Comparator block, row 3
        row = (
            CMP_LEFT_1
            + CROSSING * i
            + CMP_DECODER_1
            + CROSSING * (j_size - i - 1)
            + HORIZONTAL
            + CROSSING * t_size
            + CMP_BLOCK_3
            + VERTICAL
            + VERTICAL
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

        # Comparator block, row 4
        row = (
            CMP_LEFT_2
            + CROSSING * i
            + CMP_DECODER_2
            + CROSSING * (j_size - i - 1)
            + HORIZONTAL_WITH_UPPER_MARBLE
            + CROSSING * t_size
            + CMP_BLOCK_4
            + VERTICAL
            + VERTICAL
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

        # Comparator block, row 4
        row = (
            CMP_LEFT_3
            + VERTICAL * i
            + CMP_DECODER_3
            + VERTICAL * (j_size - i - 1)
            + EMPTY
            + VERTICAL * t_size
            + CMP_BLOCK_5
            + CMP_CHECK_LOOP
            + VERTICAL
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

        # Comparator block, row 6
        row = (
            EAST_NORTH_CORNER_WITHOUT_MARBLE
            + CROSSING * j_size
            + HORIZONTAL
            + CROSSING * t_size
            + CMP_BLOCK_6
            + VERTICAL
            + VERTICAL
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

    # Comparator footer
    for cmp_header in (CMP_FOOTER_0, CMP_FOOTER_1):
        row = (
            EMPTY
            + VERTICAL * j_size
            + EMPTY
            + VERTICAL * t_size
            + EMPTY * CMP_WIDTH
            + cmp_header
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

    # Iterate over blocks
    for block_number in range(size // 2):
        decoder = (
            format(block_number, f"0{j_size}b")
            .replace("0", ZERO_ADDRESS)
            .replace("1", ONE_ADDRESS)
        )
        if words is not None:
            word1 = words[block_number * 2]
            word2 = words[block_number * 2 + 1]
            str_value = (
                format(word1, f"0{width}b")[::-1] + format(word2, f"0{width}b")[::-1]
            )
            ignored = t_size - j_size
            assert str_value[t_size : t_size + ignored] == "0" * ignored, (
                str_value,
                ignored,
                block_number,
                word1,
                word2,
            )
            value = list(map(int, str_value))
        else:
            value = [0] * t_size * 2

        # Invert very first bit
        if block_number == 0:
            value[0] = int(not value[0])

        # Block row 1: Decoder value in
        info = "┏" + "━" * ((MEMORY_WIDTH * 3) - 1) + "┓  "
        row = (
            EMPTY
            + decoder
            + EMPTY
            + decoder
            + VERTICAL * low_address_size
            + info
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)
        # Block row 2: Decoder 0
        text_size = min(high_address_size // 4, (MEMORY_WIDTH * 3) - 1)
        text = format(block_number, f"0{text_size}X")
        info = "┃" + f"{text}".center((MEMORY_WIDTH * 3) - 1) + "┃  "
        row = (
            EMPTY
            + DECODER_0 * j_size
            + EMPTY
            + VERTICAL * t_size
            + info
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)
        # Block row 3: Decoder 1
        info = "┗" + "━" * ((MEMORY_WIDTH * 3) - 1) + "┛  "
        row = (
            EMPTY_WITH_CLEAR
            + DECODER_1 * j_size
            + EMPTY
            + VERTICAL * t_size
            + info
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)
        # Block row 4: Decoder 2
        row = (
            SOUTH_EAST_CORNER_WITH_MARBLE
            + DECODER_2 * j_size
            + HORIZONTAL
            + CROSSING * t_size
            + MEMORY_WIDTH * HORIZONTAL
            + WEST_SOUTH_CORNER
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

        # Loop over bits in blocks
        for i in range(r_size):
            bit = i if i < t_size else i + t_size - j_size

            # Bit row 0: Upper crossing
            row = (
                VERTICAL
                + VERTICAL * j_size
                + SOUTH_EAST_CORNER_WITHOUT_MARBLE
                + CROSSING * t_size
                + MEMORY_WIDTH * HORIZONTAL
                + CROSSING
                + CROSSING * t_size
                + HORIZONTAL
                + CROSSING * j_size
                + WEST_SOUTH_CORNER
            )
            print(row)

            # Detect memory type
            if block_number == 1 and i == 0:
                memory = [
                    MEMORY_WRITE_ZERO_0,
                    MEMORY_WRITE_ZERO_1,
                    MEMORY_WRITE_ZERO_2,
                    MEMORY_WRITE_ZERO_3,
                    MEMORY_WRITE_ZERO_4,
                    MEMORY_WRITE_ZERO_5,
                ]
            elif block_number == 1 and i == 1:
                memory = [
                    MEMORY_WRITE_ONE_0,
                    MEMORY_WRITE_ONE_1,
                    MEMORY_WRITE_ONE_2,
                    MEMORY_WRITE_ONE_3,
                    MEMORY_WRITE_ONE_4,
                    MEMORY_WRITE_ONE_5,
                ]
            elif block_number == 1 and i == t_size:
                memory = [
                    MEMORY_READ_0,
                    MEMORY_READ_1,
                    MEMORY_READ_2,
                    MEMORY_READ_3,
                    MEMORY_READ_4,
                    MEMORY_READ_5,
                ]
            else:
                memory = [
                    MEMORY_0,
                    MEMORY_1,
                    MEMORY_2,
                    MEMORY_3,
                    MEMORY_4,
                    MEMORY_5,
                ]

            # Bit row 1-3: Memory 0-2
            for memory_index in range(3):
                row = (
                    VERTICAL
                    + VERTICAL * j_size
                    + VERTICAL
                    + VERTICAL * t_size
                    + memory[memory_index]
                    + VERTICAL
                    + VERTICAL * t_size
                    + EMPTY
                    + VERTICAL * j_size
                    + VERTICAL
                )
                print(row)

            # Bit row 4: Memory 3
            previous_bit = i - 1 if i == t_size else bit - 1
            raw = format(
                previous_bit % (2**low_address_size) ^ bit, f"0{low_address_size}b"
            )
            low_decoder = raw.replace("0", DECODER_0).replace("1", DECODER_1_INVERTED)
            output = [VERTICAL] * (t_size + j_size)
            output[i] = DECODER_OUT_0
            output_t = "".join(output[:t_size])
            output_j = "".join(output[t_size:])
            row = (
                VERTICAL
                + VERTICAL * j_size
                + VERTICAL
                + DECODER_0 * high_address_size
                + low_decoder
                + memory[3].replace("XX", f"{bit:2X}")
                + DECODER_0
                + output_t
                + EMPTY
                + output_j
                + VERTICAL
            )
            print(row)

            # Bit row 5: Memory 4
            output = [VERTICAL] * (t_size + j_size)
            output[i] = DECODER_OUT_1
            output_t = "".join(output[:t_size])
            output_j = "".join(output[t_size:])
            row = (
                VERTICAL
                + VERTICAL * j_size
                + VERTICAL_WITH_CLEAR
                + DECODER_1 * t_size
                + (memory[4].replace("○", "●") if value[bit] else memory[4])
                + DECODER_1
                + output_t
                + (READ_0 if block_number == 1 and i == t_size else EMPTY)
                + output_j
                + VERTICAL
            )
            print(row)

            # Bit row 6: Memory 5
            output = [CROSSING] * (t_size + j_size)
            output[i] = DECODER_OUT_2
            output_t = "".join(output[:t_size])
            output_j = "".join(output[t_size:])
            row = (
                VERTICAL
                + VERTICAL * j_size
                + EAST_NORTH_CORNER_WITH_MARBLE
                + DECODER_2 * t_size
                + memory[5]
                + DECODER_2_INVERTED
                + output_t
                + (READ_1 if block_number == 1 and i == t_size else HORIZONTAL)
                + output_j
                + NORTH_WEST_CORNER
            )
            print(row)

        # Block row -2: lower crossing
        row = (
            EAST_NORTH_CORNER_WITHOUT_MARBLE
            + CROSSING * j_size
            + HORIZONTAL
            + CROSSING * t_size
            + MEMORY_WIDTH * HORIZONTAL
            + NORTH_WEST_CORNER
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

        # Block row 1: Decoder value out
        row = (
            EMPTY
            + decoder
            + EMPTY
            + decoder
            + VERTICAL * low_address_size
            + MEMORY_WIDTH * EMPTY
            + EMPTY
            + VERTICAL * t_size
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

    # Footer: inversion
    row = (
        EMPTY
        + VERTICAL * j_size
        + EMPTY
        + VERTICAL * t_size
        + MEMORY_WIDTH * EMPTY
        + EMPTY
        + VERTICAL_INVERTED * t_size
        + EMPTY
        + VERTICAL_INVERTED * j_size
    )
    print(row)

    # Footer: clear
    row = (
        EMPTY
        + VERTICAL * j_size
        + EMPTY
        + VERTICAL * t_size
        + MEMORY_WIDTH * EMPTY
        + EMPTY
        + VERTICAL_CLEAR * t_size
        + EMPTY
        + VERTICAL_CLEAR * j_size
    )
    print(row)

    # Footer for T
    for i in range(t_size):
        row = (
            EMPTY
            + VERTICAL * j_size
            + EMPTY
            + VERTICAL * (t_size - 1 - i)
            + EAST_NORTH_CORNER_WITHOUT_MARBLE
            + HORIZONTAL * i
            + HORIZONTAL * MEMORY_WIDTH
            + HORIZONTAL
            + HORIZONTAL * i
            + NORTH_WEST_CORNER
            + VERTICAL * (t_size - 1 - i)
            + EMPTY
            + VERTICAL * j_size
        )
        print(row)

    # Footer for J
    for i in range(j_size):
        row = (
            EMPTY
            + VERTICAL * (j_size - 1 - i)
            + EAST_NORTH_CORNER_WITHOUT_MARBLE
            + HORIZONTAL * i
            + HORIZONTAL
            + HORIZONTAL * t_size
            + HORIZONTAL * MEMORY_WIDTH
            + HORIZONTAL
            + HORIZONTAL * t_size
            + HORIZONTAL
            + HORIZONTAL * i
            + NORTH_WEST_CORNER
            + VERTICAL * (j_size - 1 - i)
        )
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert flip jump machine code (in version 1) to a marble program"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="word size in bits (when no program is provided)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="program size in words (when no program is provided)",
    )
    parser.add_argument(
        "program",
        type=argparse.FileType(mode="rb"),
        default=None,
        nargs="?",
        help="path to a .fjm file in version 1",
    )
    namespace = parser.parse_args()
    main(namespace.width, namespace.size, namespace.program)
