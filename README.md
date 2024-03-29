Marbles
=======

Marbles is an [esoteric programming language](https://esolangs.org/wiki/Marbles) based on marble circuitry.

<p align="center">
  <img src="./demo.gif" width="50%"/><br>
  <em><sub>An ASCII to-uppercase converter</sub></em>
</p>

Table of contents
-----------------

- [Definition](#definition)
- [Simulation](#simulation)
- [Programs](#programs)
- [FlipJump computers](#flipjump-computers)
- [Implementation and performance](#implementation-and-performance)
- [Further improvements](#further-improvements)


Definition
----------

### Circuitry

A marble program is composed of closed circuits.

Closed circuits are drawn using the unicode characters `╔`, `╗`, `╚`, `╝` for turns and `║`, `═` for straight lines:
```
  ╔═══╗
  ║   ║
  ╚═══╝
```

A circuit represents not one but two sets of tracks superposed on top of each other.

This way, a marble can either ride the lower track or the upper track.

A marble riding the lower track is represented using the character `○`:
```
  ╔═○═╗
  ║   ║
  ╚═══╝
```

And a marble riding the upper track is represented using the character `●`:
```
  ╔═●═╗
  ║   ║
  ╚═══╝
```

There can be at most one marble riding a given circuit.

Two circuits can cross using the character `╬`:
```
╔═══╗
║ ○═╬═╗
●═╬═╝ ║
  ╚═══╝
```

Note that marbles never collide.

The direction a marble starts running depends on the available directions, prioritizing first:
 - going right
 - otherwise, going down
 - and finally going up

Note that a marble never starts going left:
```
╔═══╗   → → ↓
║   ║   ↓   ↓
╚═══╝   → → ↑
```

At every simulation tick, all the marbles move in their current direction:
```
╔═○═╗     ╔══○╗     ╔═══○     ╔═══╗     ╔═══╗     ╔═══╗     ╔═══╗     ╔═══╗
║   ║  →  ║   ║  →  ║   ║  →  ║   ○  →  ║   ║  →  ║   ║  →  ║   ║  →  ║   ║  →  ...
╚═══╝     ╚═══╝     ╚═══╝     ╚═══╝     ╚═══○     ╚══○╝     ╚═○═╝     ╚○══╝
```

### Logic

The upper and lower tracks can be swapped using the characters `┃` and `━`:
```
○═━═╗     ╔○━═╗     ╔═●═╗     ╔═━●╗     ╔═━═●     ╔═━═╗     ╔═━═╗
┃   ┃  →  ┃   ┃  →  ┃   ┃  →  ┃   ┃  →  ┃   ┃  →  ┃   ○  →  ┃   ┃ →  ...
╚═━═╝     ╚═━═╝     ╚═━═╝     ╚═━═╝     ╚═━═╝     ╚═━═╝     ╚═━═○
```
In this example, the marble rides:
- the lower track in the top-left and bottom-right corners
- the upper track in the top-right and bottom-left corners

The only way two marbles can interact is by building an AND gate:
```
╚═○═╤═══╝
╔═○═╛═══╗

```

An AND gate is composed of two parts:
- A control part: `╤` (in any direction, i.e. one of `╟`, `╢`, `╤`, `╧`)
- A interrupted part: `╛` (in any direction, i.e. one of `╕`, `╜`, `╘`, `╓`, `╙`, `╒`, `╖`, `╛`)

The interrupted part plays the role of a conditional-clear.

More precisely, the marble riding the interrupted track part can only remain on the upper track if the control marble also rides the upper track.
```
╚═○═╤═══╝  →  ╚══○╤═══╝  →  ╚═══○═══╝  →  ╚═══╤○══╝  →  ╚═══╤═○═╝
╔═●═╛═══╗  →  ╔══●╛═══╗  →  ╔═══○═══╗  →  ╔═══╛○══╗  →  ╔═══╛═○═╗

╚═●═╤═══╝  →  ╚══●╤═══╝  →  ╚═══●═══╝  →  ╚═══╤●══╝  →  ╚═══╤═●═╝
╔═●═╛═══╗  →  ╔══●╛═══╗  →  ╔═══●═══╗  →  ╔═══╛●══╗  →  ╔═══╛═●═╗
```

You can see it as the affected marble falling on the lower track if the control marble is not there to keep it high.

The state of the marble riding on the control track is never affected by the interaction.

Similarly, the state of the marble riding the interrupted track won't change if it's riding the lower track.
```
╚═○═╤═══╝  →  ╚══○╤═══╝  →  ╚═══○═══╝  →  ╚═══╤○══╝  →  ╚═══╤═○═╝
╔═○═╛═══╗  →  ╔══○╛═══╗  →  ╔═══○═══╗  →  ╔═══╛○══╗  →  ╔═══╛═○═╗

╚═●═╤═══╝  →  ╚══●╤═══╝  →  ╚═══●═══╝  →  ╚═══╤●══╝  →  ╚═══╤═●═╝
╔═○═╛═══╗  →  ╔══○╛═══╗  →  ╔═══○═══╗  →  ╔═══╛○══╗  →  ╔═══╛═○═╗
```

Also, marbles on either side wait if the other marble is not there yet.

This is the affected marble waiting for the control marble:
```
╚═○═╤═══╝  →  ╚══○╤═══╝  →  ╚═══○═══╝  →  ╚═══╤○══╝  →  ╚═══╤═○═╝
╔══○╛═══╗  →  ╔═══○═══╗  →  ╔═══○═══╗  →  ╔═══╛○══╗  →  ╔═══╛═○═╗
```

And this is the opposite:
```
╚══○╤═══╝  →  ╚═══○═══╝  →  ╚═══○═══╝  →  ╚═══╤○══╝  →  ╚═══╤═○═╝
╔═○═╛═══╗  →  ╔══○╛═══╗  →  ╔═══○═══╗  →  ╔═══╛○══╗  →  ╔═══╛═○═╗
```

Finally, a marble can be static. This allows for a simple clear operation:
```
    ○      →      ○      →      ○      →      ○      →      ○
╔═●═╛═══╗  →  ╔══●╛═══╗  →  ╔═══○═══╗  →  ╔═══╛○══╗  →  ╔═══╛═○═╗
```

### Display

Control parts can also be connected to a display character, either `□` or `▣`.

This character is going to be updated every time the marble rides the control part.

Logically, `□` corresponds to a lower marble, and `▣` corresponds to an upper marble.

For instance, the display below will toggle at every cycle:

```
╔═●═╗      ╔═══╗      ╔═○═╗      ╔═══╗
┃   ╟□  →  ┃   ●▣  →  ┃   ╟▣  →  ┃   ○□
╚═══╝      ╚═══╝      ╚═══╝      ╚═══╝
```

It is also possible to build larger displays using the grid characters `┼` and `█`.

It works the same as the simple display except all connected grid characters are updated.

Here is a similar example:

```
╔═●═╗┼┼┼┼┼┼     ╔═══╗██████     ╔═○═╗██████     ╔═══╗┼┼┼┼┼┼
┃   ╟┼┼  ┼┼  →  ┃   ●██  ██  →  ┃   ╟██  ██  →  ┃   ○┼┼  ┼┼  →  ...
╚═══╝┼┼┼┼┼┼     ╚═══╝██████     ╚═══╝██████     ╚═══╝┼┼┼┼┼┼
```

Note that marbles can also ride on grid characters. In this case, the grid is treated as a straight track.

This allows for more packed display:
```
╔══┼┼═●╗     ╔══██══╗     ╔══██═○╗     ╔══┼┼══╗
┃┼┼┼┼┼┼╢  →  ┃██████●  →  ┃██████╢  →  ┃┼┼┼┼┼┼○  →  ...
╚══┼┼══╝     ╚══██══╝     ╚══██══╝     ╚══┼┼══╝
```

### Input / Output

The simulation reads from an input bit stream and writes to an output bit stream.

A read-bit contraption is built by combining an interrupted part and the IO character `◇`:
```
    ◇
╔═●═╛═══╗
```

When an upper marble reaches this part, a bit is read from the input stream. If the bit is zero, the marble drops on the lower track.

However, when a lower marble reaches the interrupted part, **no bit is read** from the input stream.

Similar contraptions are used to write to the output stream, using the control part this time.

The following combination writes a `0` to the output stream if the marble is riding the upper track:
```
    ◇
╔═●═╧═══╗
```

The other IO character `◆` is used to write a `1` to the output stream instead:
```
    ◆
╔═●═╧═══╗
```

Finally, it is possible to conditionally terminate the simulation using the exit character `☒`.

In this case, the simulation terminates if the marble is riding the upper track.

```
    ☒
╔═●═╧═══╗
```

### Cheat sheet

To sum it up, here is the full character set.

- Marbles: `○`, `●`
- Straight lines: `║`, `═`
- Turns: `╔`, `╗`, `╚`, `╝`
- Crossing: `╬`
- Control: `╟`, `╢`, `╤`, `╧`
- Inversion: `┃`, `━`
- Conditional clear: `╕`, `╜`, `╘`, `╓`, `╙`, `╒`, `╖`, `╛`
- Display: `□`, `▣`, `┼`, `█`
- IO/Exit: `◇`, `◆`, `☒`

Anything else is treated as empty and can be used for comments.

All this is summarized in the [cheat sheet](./cheatsheet.txt).


Simulation
----------

Marble programs can be executed using [marbles.py](./marbles.py).

It requires a python3 interpreter (version `>= 3.8`). It is compatible with pypy3 for faster simulation.

For instance, the [cheat sheet](./cheatsheet.txt) is a valid program and can be executed using:
```shell
./marbles.py cheatsheet.txt
```

The simulation is shown directly in the terminal at the rate of 10 simulation ticks per seconds.

A couple of key bindings are available:
- use the arrow keys to move around
- use page up / page down or the mouse wheel to scroll faster
- use `i` and `d` to increase or decrease the simulation speed (5 presses corresponds to a 10x factor)
- use `p` to pause the simulation
- use `q` to quit the simulation before it finishes

During the simulation, the input bit stream can come from different places:
- if the `--input` option is provided, the provided file is used as input stream
- if stdin is not a tty (e.g when it comes from a linux pipe `|`), stdin is used directly
- otherwise, the input stream is considered empty and the program will stop at the first attempt to read it.

Similarly, the output bit stream can go to different places:
- if the `--output` option is provided, the output stream is written at the given path
- if stdout is not a tty (e.g when it goes to a linux pipe `|`), stdout is used directly
- otherwise, the output stream is captured and written to stdout at the end of the simulation.

Note that when stdout is not a tty, the simulation is not displayed and it will run as fast as possible until it terminates.

For instance, try running the [to-uppercase.txt](./to-uppercase.txt) program using the following command:
```shell
$ echo Test! | ./marbles.py to-uppercase.txt | buffer
Loading group info: done (1 groups)
Generating callbacks for group 0: done (128 callbacks)
Running simulation: stopped with EOFError (7 cycles)
TEST!
```

Note that the analysis information can be silenced using the `--quiet` option:
```shell
$ echo Test! | ./marbles.py to-uppercase.txt --quiet | cat
TEST!
```

Other options are available, checkout the `--help` message more information:
```shell
$ ./marbles.py --help
usage: marbles.py [-h] [--speed SPEED] [--max-speed] [--fps FPS] [--input INPUT] [--output OUTPUT] [--ignore-cache] [--no-display] [--quiet] file

Run a marble simulation. Install `tqdm` for better progress bars and `msgpack` for caching the analysis results.

positional arguments:
  file             path to the marble program to run

options:
  -h, --help       show this help message and exit
  --speed SPEED    simulation speed in ticks/seconds (default is 10)
  --max-speed      run the simulation at maximum speed
  --fps FPS        the display refresh rate in frame/seconds (default is 60)
  --input INPUT    path to the file to use as input bit stream (default is stdin if it is not a tty)
  --output OUTPUT  path to the file to use as output bit stream (default is stdout if it is not a tty)
  --ignore-cache   ignore the cache even if it exists (enabled by default when msgpack is not installed)
  --no-display     run the simulation without the display at maximum speed (enabled by default when stdout is not a tty)
  --quiet          do not output any information about the analysis
```

Programs
--------

A couple of marble programs are provided in the repository, other than the [cheatsheet.txt](./cheatsheet.txt) program mentioned earlier.


### Counters with 7-segment display

Three different counter programs are provided:
- [4-bit-counter.txt](./4-bit-counter.txt)
- [8-bit-counter.txt](./8-bit-counter.txt)
- [16-bit-counter.txt](./16-bit-counter.txt)

It demonstrate the use of a single-digit 7-segment display design with a width of 31 characters that can be concatenated for larger displays.

<p align="center">
  <img src="./7-segment.gif" width="30%"/><br>
  <em><sub>A single-digit 7-segment display</sub></em>
</p>

It includes:
- The display itself, using the grid characters
- The synchronization of all the segment marbles in order to avoid display artifacts
- A 4-to-16-bit decoder to help with setting the right segment for the right digit
- The 4/8/16-bit counter itself, that loops forever

### ASCII to-uppercase converter

A [to-uppercase.txt](./to-uppercase.txt) program is provided to demonstrate that the marble simulation can run non-trivial computation.

This program converts every ascii lowercase letter from the input stream to an uppercase letter, leaving the other characters untouched. It stops when the input stream is empty.

Each block in the program is commented to demonstrate how such programs can be structured in a readable way. It also shows that those blocks can be easily copied, pasted and moved around.

Another version of the same program is provided as [to-uppercase-with-flipjump.txt.gz](./to-uppercase-with-flipjump.txt.gz).

This version is compressed in order to keep the file size small, as the original text file is about 140 MB.

The way this file is generated is explained in the next section where flip jump computers are presented, starting with [tiny-flipjump.txt](./tiny-flipjump.txt).

FlipJump computers
------------------

[FlipJump](https://esolangs.org/wiki/FlipJump) is a 1-instruction language. As the wiki entry on [esoloangs](https://esolangs.org) says:

> FlipJump is intending to be the simplest / most-primitive programming language. Yet, it can do any modern computation.
> As the name implies - It Flips a bit, then Jumps (unconditionally).


### A tiny computer

A FlipJump computer is defined by its word size in bits (also called width). The smallest FlipJump computer has a width of 4 bits.

A marble implementation of this tiny computer can be found in [tiny-flipjump.txt](./tiny-flipjump.txt). It is programmed with two instructions:
```
 9;8 // Write 1 to the output stream and go to the second instruction
 8;0 // Write 0 to the output stream and read a bit from the input stream
     // Go to the first instruction if it is 0, stop the program otherwise
 ```

This program outputs `1` then `0` for each `0` in the input stream and stops at the first `1`. For instance, giving `\x00\x10` as input produces `\x55\x55\x55`

It is 16 bits long but in practice, only 7 of those bits are actually programmed in the computer. That's because the jump addresses are considered to be instruction-aligned, meaning that bits 4, 5, 6 for each instruction are expected to be 0. Also, bits 0, 1 and 7 of the second instruction have a special meaning and are dedicated to input/output operations.

In short, only the following bits are programmed in the computer:
```
0b1001 0b1...
0b10.. 0b....
```

Those bit values appear in the bottom-right corner of each memory cell, using a lower marble `○` for `0` and and upper marble `●` for `1`.

Here is the program running with the `\x00\x10` input mentioned earlier:
```shell
± echo -n '\x00\x10' | ./marbles.py tiny-flipjump.txt --quiet | xxd
00000000: 5555 55                                  UUU
```

### Larger computers

This 4-bit computer is quite limited but the same design can be generalized to larger width.

The script [flipjump-to-marbles.py](./flipjump-to-marbles.py) can be used to generate larger computers:
```shell
$ ./flipjump-to-marbles.py --width 8
8-bit FlipJump computer with 32 words of memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[...]
```

Those computers grows in size very rapidly so make sure to specify the size in words:
```shell
$ ./flipjump-to-marbles.py --width 16
[...] # About 163 MB

$ ./flipjump-to-marbles.py --width 32
[...] # About 21 TB

$ ./flipjump-to-marbles.py --width 16 --size 128
[...] # About 5 MB

$ ./flipjump-to-marbles.py --width 32 --size 128
[...] # About 21 MB
```

More conveniently, those computers can be generated from a FlipJump memory file (`.fjm`). Width and size are taken directly from the file header, e.g:
```shell
$ ./flipjump-to-marbles.py hello_world.fjm
16-bit FlipJump computer programmed with `hello_world.fjm` (262 words)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[...]
```

See below the complete toolchain from a FlipJump assembly file to executing the program:
```shell
# Install flipjump package
$ pip install flipjump

# Write a hello world program
$ echo '''
stl.startup
stl.output "Hello, World!"
stl.loop
''' > hello_world.fj

# Assemble a hello world program from the flip-jump repository (width 16)
$ fj --asm hello_world.fj -o hello_world.fjm -w 16

# Convert it to a marble program
$ ./flipjump-to-marbles.py hello_world.fjm > hello_world.txt

# Run it with mypy for faster execution
$ pypy3 ./marbles.py hello_world.txt --quiet | buffer
Hello, World!
```

### Turing-completeness

FlipJump is considered a [Bounded-storage machine](https://esolangs.org/wiki/Bounded-storage_machine):

> A bounded-storage machine, abbreviated BSM, is in all respects similar to a Turing machine, save that there is an arbitrary bound on the size of its data storage unit (i.e. the length of its tape.)

The fact that [flipjump-to-marbles.py](./flipjump-to-marbles.py) is able to generate arbitrarily large FlipJump computers should be enough to prove that this marble model is itself a bounded-storage machine.

It is however **not** [Turing-complete](https://esolangs.org/wiki/Turing-complete), since marble programs do not have access to an arbitrarily large memory at runtime. For instance, it's not possible to write a marble program that would reverse an arbitrarily large null-terminated string coming from the input stream.

Turing-completeness could possibly be achieved by adding stacks to the language as discussed in the [further improvements](#implementing-stacks) section.


### ASCII to-uppercase converter, using FlipJump

As an example, let's implement the ascii `to-uppercase` program from earlier using FlipJump.

First, write the program in FlipJump assembly, as in [to-uppercase.fj](./to-uppercase.fj):

```asm
  startup
main:
  bit.input current_char
  bit.cmp 8, current_char, a_char, skip, continue1, continue1
continue1:
  bit.cmp 8, current_char, z_char, continue2, continue2, skip
continue2:
  bit.sub 8, current_char, to_upper
skip:
  bit.print current_char
  ;main

current_char: bit.vec 8, 0x00
to_upper: bit.vec 8, 0x20
a_char: bit.vec 8, 0x61
z_char: bit.vec 8, 0x7a
```

Then assemble it with the lowest width possible
```shell
$ fj --asm to-uppercase.fj -o to-uppercase.fjm -w 16
```

Convert it to a marble program. The file is likely to be quite big (about 140 MB in this case), so compress it with gzip:
```shell
$ ./flipjump-to-marbles.py to-uppercase.fjm | gzip --best > to-uppercase-with-flipjump.txt.gz
```

Compressed files can be provided directly to the [marbles.py](./marbles.py) simulator:
```shell
$ echo a | pypy3 ./marbles.py to-uppercase-with-flipjump.txt.gz --quiet | cat
A
```

It's going to take about a minute for the simulator to run the analysis on a file this big, but the results will be cached if `msgpack` is installed. On the next execution, the simulation should start in a couple of seconds.

Remove the `--quiet` option to show a couple of interesting metrics:
```shell
$ echo a | pypy3 ./marbles.py to-uppercase-with-flipjump.txt.gz | buffer
Loading group info: 100%|██████████████████████████████████████| 1/1 [00:02<00:00,  2.48s/ groups]
Generating callbacks for group 0: 100%|████| 2570584/2570584 [00:02<00:00, 1121743.30 callbacks/s]
Running simulation: 1588 cycles [01:25, 18.61 cycles/s]
A
```

On the second execution, it took only 4 seconds for the simulation to start. However, the simulation itself took 1 minute and 25 seconds to process the 2 characters in the input string `a\n`. This is because the program took 1588 instructions to complete and the simulator barely reaches 20 instructions per second.

This might sound very slow (and it is), but keep in mind that FlipJump only flip bits so it takes many instructions to perform the most basic arithmetic operations. Also note that for every instruction, many marble interactions happen in all the memory cells even if all of them but one are left untouched by the computer. Actually, it took quite a lot of effort to optimize the simulator to the point where such large marble programs can realistically run.

More implementation details are discussed below.


Implementation and performance
------------------------------

### General idea

A naive implementation of the simulator would list all the marbles, initialize them with the proper directions and loop over simulation ticks. At each tick, the new state for a marble is computed from the information on its cell and its neighbors, similar to a cellular automaton.

This would be really slow. Instead, the current implementation focus on interactions. It first performs an analysis of the program, separating the positional information (i.e where the marbles are specifically on the grid) from the state information (i.e whether the marble is in upper or lower state).

One key insight is that after some time, positions on a given circuit become periodic and that all connected circuits share the same period. In the context of the simulator, connected circuits are called a group. For instance, the [cheat sheet](./cheatsheet.txt) has 17 different groups.

That means that for a given group, events can be recorded from the first tick to the point when the group becomes periodic. Then, events can be recorded for a full cycle. All this information allows for quickly computing the position of any marble at any arbitrary tick in the future.

Similarly, the order of interactions between the marbles can be determined in advance for both the initialization phase and the cycle phase. Callbacks are generated for each interaction so that the actual simulation is only a matter of running and cycling those callbacks.

### Analysis

More precisely, the analysis is composed of the following steps:

- Creating the grid: read the program file and initialize the grid

- Extracting marbles from the grid: list the marbles and replace them with the right piece of tracks

- Building circuits: for each marble, follow the circuit and list all positions and interactions in order

- Detecting groups: group the circuits based on their connections

- Analyzing groups: for each group, find its period and the tick when the first cycle start

- Extracting info: gather all information necessary to run the simulation

- Dumping info: write this information to a cache file using `msgpack`

- Generating callbacks: generate initialization and cycle callbacks for each group


### Display

The display logic has also been optimized for large programs. In practice, showing the simulation on screen should only take a small part of the time in each frame, even when rendering at 60 FPS. This information is available in the status bar in the bottom row of the simulation screen.

This is mostly done by using a binary search when looking for marbles (and displays) in the current window. This search however is only performed along the X-axis (vertically), which means that the display logic is optimized for long vertical programs rather than long horizontal programs.

The simulator also uses a binary search to find the position for each marble using the information computed by the analysis. It also avoids refreshing part of the screen that doesn't need to be refreshed.


Further improvements
--------------------

### Faster simulation

There are a couple of ways to improve the simulation speed.

Using a faster language would be one them, but I wouldn't expect more than a ~5x speed up compare to the current pypy speed, as it's already able to compile callback execution at runtime.

Another technique would be to gather the full boolean expressions for all marbles during a complete cycle, and then perform another layer of analysis to factorize them in a way that would reduce the amount of computation, using both short-circuits and intermediate results.

Taking the example of the FlipJump computers, the analysis should theoretically be able to realize that:
- Memory marbles don't change state unless a specific combination of input is present, and that those marbles share the same inputs.
- Some marbles are the product of a large OR computation where each branch is mutually exclusive depending on a specific combination of inputs.

In both cases, indexing and indirections might be used to essentially reduce the computation to its theoretical limit, i.e. what a standard FlipJump interpreter would do.


### Other computer architectures

It would be interesting to see other computer architectures implemented with the marbles model. In particular, [Subleq](https://esolangs.org/wiki/Subleq) might be a nice place to start.


### Implementing stacks

As [explained here](#turing-completeness), it should be possible to make the marble model Turing-complete by adding infinite bit stacks to its definition.

Here's a possible implementation based on a new stack character `▷`/`▶`:

```
POP     ═○════╗
PUSH    ═○══╗ ║
DATA_IN ═○══╬╤╬════ ...
            ╟▷╜
            ║ ╚════ DATA_OUT
            ╚══════ ...
```
- `DATA_IN` prepares the value to push on the stack (the stack character is either `▷` or `▶` depending on the last prepared value)
- `PUSH` decides whether the prepared value should be pushed on the stack
- `POP` decides whether a value should be popped from the stack
- `DATA_OUT` is the value popped from the stack, or `○` if `POP` was low.
