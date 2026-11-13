"""The one clock every device's samples are put on.

``time.time()`` can be stepped or slewed by the operating system's time
synchronisation in the middle of a recording; ``perf_counter`` cannot.
So brAIn stream's clock is ``perf_counter``, moved once, when brAIn stream
starts, to read seconds since 1970: steady like ``perf_counter``, and as
readable as ``time.time()`` (within whatever the computer's clock was off
by then). Every time stamp in a recording is on it.
"""
from __future__ import annotations

import time

#: perf_counter + _OFFSET reads seconds since 1970, as time.time() did
#: when brAIn stream started.
_OFFSET = time.time() - time.perf_counter()


def now() -> float:
    """Seconds since 1970, on a clock that never jumps."""
    return time.perf_counter() + _OFFSET


def from_perf(t: float) -> float:
    """A ``time.perf_counter()`` reading, on this clock."""
    return t + _OFFSET


def from_wall(t: float) -> float:
    """A ``time.time()`` reading (taken moments ago), on this clock: the
    two clocks' difference is measured now, so a step of the wall clock
    since brAIn stream started does not carry over."""
    return t - (time.time() - now())


def from_stamp(t: float) -> float:
    """A reading of g.Pype's stamp clock (``channels.stamp_clock()``), on
    this clock."""
    from gpype.common._private import channels
    return t + (now() - channels.stamp_clock())
