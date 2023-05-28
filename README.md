Marbles
=======

Marbles is a computational model and an esoteric programming language based marble circuitry.


Table of contents
-----------------

- [Definition](#definition)
- [Simulation](#simulation)
- [Programs](#programs)
- [Flip-jump computers](#flip-jump-computers)
- [Implementation](#implementation)
- [Future improvements](#future-improvements)


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

There can be at most one marble riding a given track.

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

The upper and lower track can be swapped using the characters `┃` and `━`:
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
Compiling callbacks for group 0: done (128 callbacks)
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

It includes:
- The display itself, using the grid characters
- The synchronization of all the segment marbles in order to avoid display artifacts
- A 4-to-16-bit decoder to help with setting the right segment for the right digit
- The 4/8/16-bit counter itself, that loops forever

### ASCII to-uppercase converter

A [to-uppercase.txt](./to-uppercase.txt) program is provided to demonstrate that the marble simulation can run non-trivial computation.

This program converts every ascii lowercase letter from the input stream to an uppercase letter, leaving the other characters untouched.

It stops when the input stream is empty.

Each block in the program is commented to demonstrate how such programs can be structured in a readable way.

It also shows that those blocks can be easily copied, pasted and moved around.

Another version of the same program is provided as [to-uppercase-with-flipjump.txt.gz](./to-uppercase-with-flipjump.txt.gz).

This version is compressed in order to keep the file size small, as the original text file is about 140 MB.

The way this file is generated is explained in the next section where flip jump computers are presented, starting with [tiny-flip-jump.txt](./tiny-flip-jump.txt).

Flip-Jump computers
-------------------

TODO


Implementation
--------------

TODO


Further improvements
-------------------

TODO