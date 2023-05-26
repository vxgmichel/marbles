Marbles
=======

Marbles is a computational model and an esoteric programming language based marble circuitry.


Table of contents
-----------------

- [Definition](#definition)
- [Simulation](#simulation)
- [Programs](#programs)
- [Flip-jump computer](#flip-jump-computer)
- [Implementation][#implementation]
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

TODO


Programs
--------

TODO


Flip-Jump computer
------------------

TODO


Implementation
--------------

TODO


Further improvements
-------------------

TODO