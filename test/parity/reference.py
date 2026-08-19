"""Arbitrary-precision references for the four cancellation-prone kernels.

Why this module exists
----------------------
Four scalar-mediator elastic cross sections evaluate

.. code-block:: text

    P * atan(ms / width_s)  -  P * atan((ms**2 - 4 mx**2 + s) / (ms * width_s))

with the *same* prefactor ``P`` on both terms. Whenever the two ``atan``
arguments are close (near ``e_cm = 2 mx``, where they are equal) or both
saturate at ``pi/2`` (``width_s -> 0``), that subtraction is a difference
of two doubles that agree to the last bit, and what survives is the
platform's ``atan`` rounding rather than the physics. Dividing by the
small ``4 mx**2 - s`` in the denominator then amplifies the residue into
the answer.

The corpus pinned whatever macOS/arm64 produced there, and that is not a
number any other implementation reproduces. To decide *which* pinned
values are like that, something has to know the right answer. That is
this module: the same closed forms, evaluated at 60 decimal digits, where
the cancellation costs 20 digits out of 60 instead of all 16 out of 16.

`stability.py` uses it once, offline, to build the unpinnable-point mask
that `test_parity` honours; nothing in the test path imports it at
runtime, so `mpmath` stays a regeneration-only dependency.

How faithful this is
--------------------
The four bodies below were extracted **verbatim** from
``hazma/scalar_mediator/_c_scalar_mediator_cross_sections.pyx`` at kernel
digest ``f5e6e269be47`` (the digest the corpus manifest records), lines
265-292, 293-391, 392-489 and 490-515. A ``cdef double`` body in that
file is pure Python-syntax arithmetic, so the copy needed no
transcription: only the ``cdef`` declarations were dropped and the
arguments promoted to ``mpmath.mpf``, after which `black` rewrapped the
lines — whitespace only, and the mask regenerates identically either side
of that run. Keeping them verbatim is the point
— a re-derived expression would answer a different question, namely
whether the published formula is right, rather than whether the
*evaluation* of it is.

Three deliberate differences, none of which moves a value by more than
~1e-16 relative:

- **Constants stay doubles.** ``vh``, ``b0``, ``alpha_em`` and the meson
  masses below are the same literals the ``.pyx`` declares; promoting a
  double to ``mpf`` is exact, so the reference evaluates *this* formula
  with *these* constants and only the arithmetic becomes exact.
- **``M_PI`` becomes true pi.** ``libc``'s ``M_PI`` is the double nearest
  pi; the reference uses ``mpmath.pi``. The question being asked is what
  the formula is worth, so the mathematical constant is the right one.
- **The threshold guards run in double.** ``e_cm < mx + ml`` is evaluated
  on the incoming floats, before promotion, because ``mx + ml`` rounds in
  double and the corpus grid places an anchor exactly there. Comparing in
  exact arithmetic would put that anchor on the other side of the branch
  and report a spurious disagreement — measured, not hypothesised: it
  flagged four points as "wrong" that are only on the far side of a
  rounded comparison.

Phases 05 deletes the ``.pyx`` these came from. This module is a standing
copy, not a view, and the header above is the provenance that survives
the deletion.
"""

from __future__ import annotations

import mpmath as mp

# The four signatures below mirror the `.pyx` kernels argument for
# argument, which is the whole point of the module: a caller passes the
# corpus block's `params["args"]` straight through. Grouping them into a
# dataclass would be a transcription, and a transcription is exactly what
# this file exists to avoid.
# ruff: noqa: PLR0913, PLR0917


#: Digits carried by every evaluation here. The worst cancellation the
#: corpus samples is `closed_resonance`, where `atan(u) - atan(v)` loses
#: about 33 decimal digits (the true difference is ~3e-25 against terms
#: of order 1). 60 leaves ~27 digits standing, which is 18 more than the
#: 1e-9 the mask is thresholded at.
DPS = 60

# The constants below are the `cdef double` module-level declarations of
# `_c_scalar_mediator_cross_sections.pyx`, copied verbatim. `me` and `mmu`
# are unused by these four kernels but kept so the block matches its
# source; the fermion mass arrives as the `ml` argument instead.
vh = 246.22795e3
alpha_em = 1.0 / 137.04
me = 0.510998928
mmu = 105.6583715
mpi0 = 134.9766
mpi = 139.57018
b0 = 2654.082197477761
muq = 2.3
mdq = 4.8

M_PI = mp.pi
atan = mp.atan
log = mp.log
sqrt = mp.sqrt
atanh = mp.atanh


class _NumpyShim:
    """Resolves the one ``np.log`` call `sigma_xl_to_xl`'s body makes."""

    log = staticmethod(mp.log)


np = _NumpyShim()


def _to_mpf(*values: float) -> tuple[mp.mpf, ...]:
    """Promote every argument to `mpmath.mpf`. Exact for a float input."""
    return tuple(mp.mpf(value) for value in values)


# ===========================================================================
# ---- Verbatim kernel bodies -----------------------------------------------
# ===========================================================================


def sigma_xl_to_xl(
    e_cm: float,
    mx: float,
    ms: float,
    gsxx: float,
    gsff: float,
    gsGG: float,
    gsFF: float,
    lam: float,
    width_s: float,
    vs: float,
    ml: float,
) -> mp.mpf | float:
    """Verbatim ``__sigma_xl_to_xl`` body, evaluated in mpmath."""
    if e_cm < mx + ml:
        return 0.0

    e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs, ml = _to_mpf(
        e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs, ml
    )

    s = e_cm**2

    return 2.0 * (
        (
            (gsff * ml) ** 2
            * gsxx**2
            * (
                (
                    -4 * ml**2 * (ms**2 - 4 * mx**2)
                    + ms**2 * (ms**2 - 4 * mx**2 - width_s**2)
                )
                * atan(ms / width_s)
                + (
                    4 * ml**2 * (ms**2 - 4 * mx**2)
                    + ms**2 * (-(ms**2) + 4 * mx**2 + width_s**2)
                )
                * atan((ms**2 - 4 * mx**2 + s) / (ms * width_s))
                + ms
                * width_s
                * (
                    4 * mx**2
                    - s
                    + ms**2 * np.log(4)
                    - ml**2 * log(16)
                    - mx**2 * log(16)
                    + (2 * ml**2 - ms**2 + 2 * mx**2)
                    * log(4 * ms**2 * (ms**2 + width_s**2))
                    + (-2 * ml**2 + ms**2 - 2 * mx**2)
                    * log(
                        ms**4
                        + (-4 * mx**2 + s) ** 2
                        + ms**2 * (-8 * mx**2 + 2 * s + width_s**2)
                    )
                )
            )
        )
        / (32.0 * ms * M_PI * e_cm**2 * (4 * mx**2 - s) * width_s)
    )


def sigma_xpi_to_xpi(
    e_cm: float,
    mx: float,
    ms: float,
    gsxx: float,
    gsff: float,
    gsGG: float,
    gsFF: float,
    lam: float,
    width_s: float,
    vs: float,
) -> mp.mpf | float:
    """Verbatim ``__sigma_xpi_to_xpi`` body, evaluated in mpmath."""
    if e_cm < mx + mpi:
        return 0.0

    e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs = _to_mpf(
        e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs
    )

    return 2.0 * (
        (
            gsxx**2
            * (
                2
                * (
                    b0**2
                    * (mdq + muq) ** 2
                    * (ms**2 - 4 * mx**2)
                    * (9 * lam + 4 * gsGG * vs) ** 2
                    * (
                        27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                        - 2
                        * gsGG
                        * vh**2
                        * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                        + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                    )
                    ** 2
                    + 324
                    * b0
                    * gsGG
                    * lam**3
                    * (mdq + muq)
                    * vh**2
                    * (9 * lam + 4 * gsGG * vs)
                    * (
                        27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                        - 2
                        * gsGG
                        * vh**2
                        * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                        + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                    )
                    * (
                        2 * mpi**2 * (ms**2 - 4 * mx**2)
                        + ms**2 * (-(ms**2) + 4 * mx**2 + width_s**2)
                    )
                    + 26244
                    * gsGG**2
                    * lam**6
                    * vh**4
                    * (
                        ms**6
                        + 4 * mpi**4 * (ms**2 - 4 * mx**2)
                        + 4 * ms**2 * mx**2 * width_s**2
                        - 4 * mpi**2 * ms**2 * (ms**2 - 4 * mx**2 - width_s**2)
                        - ms**4 * (4 * mx**2 + 3 * width_s**2)
                    )
                )
                * atan(ms / width_s)
                - 2
                * (
                    b0**2
                    * (mdq + muq) ** 2
                    * (ms**2 - 4 * mx**2)
                    * (9 * lam + 4 * gsGG * vs) ** 2
                    * (
                        27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                        - 2
                        * gsGG
                        * vh**2
                        * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                        + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                    )
                    ** 2
                    + 324
                    * b0
                    * gsGG
                    * lam**3
                    * (mdq + muq)
                    * vh**2
                    * (9 * lam + 4 * gsGG * vs)
                    * (
                        27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                        - 2
                        * gsGG
                        * vh**2
                        * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                        + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                    )
                    * (
                        2 * mpi**2 * (ms**2 - 4 * mx**2)
                        + ms**2 * (-(ms**2) + 4 * mx**2 + width_s**2)
                    )
                    + 26244
                    * gsGG**2
                    * lam**6
                    * vh**4
                    * (
                        ms**6
                        + 4 * mpi**4 * (ms**2 - 4 * mx**2)
                        + 4 * ms**2 * mx**2 * width_s**2
                        - 4 * mpi**2 * ms**2 * (ms**2 - 4 * mx**2 - width_s**2)
                        - ms**4 * (4 * mx**2 + 3 * width_s**2)
                    )
                )
                * atan((ms**2 - 4 * mx**2 + e_cm**2) / (ms * width_s))
                + ms
                * width_s
                * (
                    -324
                    * gsGG
                    * lam**3
                    * (4 * mx**2 - e_cm**2)
                    * vh**2
                    * (
                        81
                        * gsGG
                        * lam**3
                        * (8 * mpi**2 - 4 * ms**2 + 4 * mx**2 + e_cm**2)
                        * vh**2
                        + 2
                        * b0
                        * (mdq + muq)
                        * (9 * lam + 4 * gsGG * vs)
                        * (
                            27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                            - 2
                            * gsGG
                            * vh**2
                            * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                            + gsff
                            * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                        )
                    )
                    - (
                        648
                        * b0
                        * gsGG
                        * lam**3
                        * (mdq + muq)
                        * (mpi**2 - ms**2 + 2 * mx**2)
                        * vh**2
                        * (9 * lam + 4 * gsGG * vs)
                        * (
                            27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                            - 2
                            * gsGG
                            * vh**2
                            * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                            + gsff
                            * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                        )
                        + b0**2
                        * (mdq + muq) ** 2
                        * (9 * lam + 4 * gsGG * vs) ** 2
                        * (
                            27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                            - 2
                            * gsGG
                            * vh**2
                            * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                            + gsff
                            * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                        )
                        ** 2
                        + 26244
                        * gsGG**2
                        * lam**6
                        * vh**4
                        * (
                            4 * mpi**4
                            - 8 * mpi**2 * (ms**2 - 2 * mx**2)
                            + ms**2 * (3 * ms**2 - 8 * mx**2 - width_s**2)
                        )
                    )
                    * log(ms**2 * (ms**2 + width_s**2))
                    + (
                        648
                        * b0
                        * gsGG
                        * lam**3
                        * (mdq + muq)
                        * (mpi**2 - ms**2 + 2 * mx**2)
                        * vh**2
                        * (9 * lam + 4 * gsGG * vs)
                        * (
                            27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                            - 2
                            * gsGG
                            * vh**2
                            * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                            + gsff
                            * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                        )
                        + b0**2
                        * (mdq + muq) ** 2
                        * (9 * lam + 4 * gsGG * vs) ** 2
                        * (
                            27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                            - 2
                            * gsGG
                            * vh**2
                            * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                            + gsff
                            * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                        )
                        ** 2
                        + 26244
                        * gsGG**2
                        * lam**6
                        * vh**4
                        * (
                            4 * mpi**4
                            - 8 * mpi**2 * (ms**2 - 2 * mx**2)
                            + ms**2 * (3 * ms**2 - 8 * mx**2 - width_s**2)
                        )
                    )
                    * log(
                        ms**4
                        + (-4 * mx**2 + e_cm**2) ** 2
                        + ms**2 * (-8 * mx**2 + 2 * e_cm**2 + width_s**2)
                    )
                )
            )
        )
        / (
            419904.0
            * lam**6
            * ms
            * M_PI
            * e_cm**2
            * (-4 * mx**2 + e_cm**2)
            * vh**4
            * (9 * lam + 4 * gsGG * vs) ** 2
            * width_s
        )
    )


def sigma_xpi0_to_xpi0(
    e_cm: float,
    mx: float,
    ms: float,
    gsxx: float,
    gsff: float,
    gsGG: float,
    gsFF: float,
    lam: float,
    width_s: float,
    vs: float,
) -> mp.mpf | float:
    """Verbatim ``__sigma_xpi0_to_xpi0`` body, evaluated in mpmath."""
    if e_cm < mx + mpi0:
        return 0.0

    e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs = _to_mpf(
        e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs
    )

    return (
        gsxx**2
        * (
            2
            * (
                b0**2
                * (mdq + muq) ** 2
                * (ms**2 - 4 * mx**2)
                * (9 * lam + 4 * gsGG * vs) ** 2
                * (
                    27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                    - 2
                    * gsGG
                    * vh**2
                    * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                    + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                )
                ** 2
                + 324
                * b0
                * gsGG
                * lam**3
                * (mdq + muq)
                * vh**2
                * (9 * lam + 4 * gsGG * vs)
                * (
                    27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                    - 2
                    * gsGG
                    * vh**2
                    * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                    + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                )
                * (
                    2 * mpi0**2 * (ms**2 - 4 * mx**2)
                    + ms**2 * (-(ms**2) + 4 * mx**2 + width_s**2)
                )
                + 26244
                * gsGG**2
                * lam**6
                * vh**4
                * (
                    ms**6
                    + 4 * mpi0**4 * (ms**2 - 4 * mx**2)
                    + 4 * ms**2 * mx**2 * width_s**2
                    - 4 * mpi0**2 * ms**2 * (ms**2 - 4 * mx**2 - width_s**2)
                    - ms**4 * (4 * mx**2 + 3 * width_s**2)
                )
            )
            * atan(ms / width_s)
            - 2
            * (
                b0**2
                * (mdq + muq) ** 2
                * (ms**2 - 4 * mx**2)
                * (9 * lam + 4 * gsGG * vs) ** 2
                * (
                    27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                    - 2
                    * gsGG
                    * vh**2
                    * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                    + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                )
                ** 2
                + 324
                * b0
                * gsGG
                * lam**3
                * (mdq + muq)
                * vh**2
                * (9 * lam + 4 * gsGG * vs)
                * (
                    27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                    - 2
                    * gsGG
                    * vh**2
                    * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                    + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                )
                * (
                    2 * mpi0**2 * (ms**2 - 4 * mx**2)
                    + ms**2 * (-(ms**2) + 4 * mx**2 + width_s**2)
                )
                + 26244
                * gsGG**2
                * lam**6
                * vh**4
                * (
                    ms**6
                    + 4 * mpi0**4 * (ms**2 - 4 * mx**2)
                    + 4 * ms**2 * mx**2 * width_s**2
                    - 4 * mpi0**2 * ms**2 * (ms**2 - 4 * mx**2 - width_s**2)
                    - ms**4 * (4 * mx**2 + 3 * width_s**2)
                )
            )
            * atan((ms**2 - 4 * mx**2 + e_cm**2) / (ms * width_s))
            + ms
            * width_s
            * (
                -324
                * gsGG
                * lam**3
                * (4 * mx**2 - e_cm**2)
                * vh**2
                * (
                    81
                    * gsGG
                    * lam**3
                    * (8 * mpi0**2 - 4 * ms**2 + 4 * mx**2 + e_cm**2)
                    * vh**2
                    + 2
                    * b0
                    * (mdq + muq)
                    * (9 * lam + 4 * gsGG * vs)
                    * (
                        27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                        - 2
                        * gsGG
                        * vh**2
                        * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                        + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                    )
                )
                - (
                    648
                    * b0
                    * gsGG
                    * lam**3
                    * (mdq + muq)
                    * (mpi0**2 - ms**2 + 2 * mx**2)
                    * vh**2
                    * (9 * lam + 4 * gsGG * vs)
                    * (
                        27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                        - 2
                        * gsGG
                        * vh**2
                        * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                        + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                    )
                    + b0**2
                    * (mdq + muq) ** 2
                    * (9 * lam + 4 * gsGG * vs) ** 2
                    * (
                        27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                        - 2
                        * gsGG
                        * vh**2
                        * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                        + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                    )
                    ** 2
                    + 26244
                    * gsGG**2
                    * lam**6
                    * vh**4
                    * (
                        4 * mpi0**4
                        - 8 * mpi0**2 * (ms**2 - 2 * mx**2)
                        + ms**2 * (3 * ms**2 - 8 * mx**2 - width_s**2)
                    )
                )
                * log(ms**2 * (ms**2 + width_s**2))
                + (
                    648
                    * b0
                    * gsGG
                    * lam**3
                    * (mdq + muq)
                    * (mpi0**2 - ms**2 + 2 * mx**2)
                    * vh**2
                    * (9 * lam + 4 * gsGG * vs)
                    * (
                        27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                        - 2
                        * gsGG
                        * vh**2
                        * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                        + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                    )
                    + b0**2
                    * (mdq + muq) ** 2
                    * (9 * lam + 4 * gsGG * vs) ** 2
                    * (
                        27 * gsff**2 * lam**2 * vs * (3 * lam + 4 * gsGG * vs)
                        - 2
                        * gsGG
                        * vh**2
                        * (27 * lam**2 - 30 * gsGG * lam * vs + 8 * gsGG**2 * vs**2)
                        + gsff * (-81 * lam**3 * vh + 48 * gsGG**2 * lam * vh * vs**2)
                    )
                    ** 2
                    + 26244
                    * gsGG**2
                    * lam**6
                    * vh**4
                    * (
                        4 * mpi0**4
                        - 8 * mpi0**2 * (ms**2 - 2 * mx**2)
                        + ms**2 * (3 * ms**2 - 8 * mx**2 - width_s**2)
                    )
                )
                * log(
                    ms**4
                    + (-4 * mx**2 + e_cm**2) ** 2
                    + ms**2 * (-8 * mx**2 + 2 * e_cm**2 + width_s**2)
                )
            )
        )
    ) / (
        419904.0
        * lam**6
        * ms
        * M_PI
        * e_cm**2
        * (-4 * mx**2 + e_cm**2)
        * vh**4
        * (9 * lam + 4 * gsGG * vs) ** 2
        * width_s
    )


def sigma_xg_to_xg(
    e_cm: float,
    mx: float,
    ms: float,
    gsxx: float,
    gsff: float,
    gsGG: float,
    gsFF: float,
    lam: float,
    width_s: float,
    vs: float,
) -> mp.mpf | float:
    """Verbatim ``__sigma_xg_to_xg`` body, evaluated in mpmath."""
    # for e_cm = 2mx there is complete destructive interference
    if e_cm < mx or e_cm == 2.0 * mx:
        return 0.0

    e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs = _to_mpf(
        e_cm, mx, ms, gsxx, gsff, gsGG, gsFF, lam, width_s, vs
    )

    s = e_cm**2

    return (
        alpha_em**2
        * gsFF**2
        * gsxx**2
        * (
            -2
            * (
                ms**5
                + 4 * ms * mx**2 * width_s**2
                - ms**3 * (4 * mx**2 + 3 * width_s**2)
            )
            * atan(ms / width_s)
            + 2
            * (
                ms**5
                + 4 * ms * mx**2 * width_s**2
                - ms**3 * (4 * mx**2 + 3 * width_s**2)
            )
            * atan((ms**2 - 4 * mx**2 + s) / (ms * width_s))
            - width_s
            * (
                (4 * ms**2 - 4 * mx**2 - s) * (4 * mx**2 - s)
                + ms**2
                * (-3 * ms**2 + 8 * mx**2 + width_s**2)
                * log(ms**2 * (ms**2 + width_s**2))
                + ms**2
                * (3 * ms**2 - 8 * mx**2 - width_s**2)
                * log(
                    ms**4
                    + (-4 * mx**2 + s) ** 2
                    + ms**2 * (-8 * mx**2 + 2 * s + width_s**2)
                )
            )
        )
    ) / (128.0 * lam**2 * M_PI**3 * (4 * mx**2 - s) * s * width_s)
