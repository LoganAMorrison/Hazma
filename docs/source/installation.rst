Installation
============

``hazma`` requires python 3.10 or newer. The simplest way to install it is
from PyPI:

.. code-block:: bash

    pip install hazma

On manylinux x86_64 and macOS arm64 that installs a prebuilt wheel, which
carries the compiled extension and needs no compiler on your system. The
scientific packages ``hazma`` depends on at runtime — ``numpy``,
``scipy``, ``scikit-image`` and ``matplotlib`` — are installed alongside
it.

Building from source
--------------------

To build from a checkout instead, download the package from the Hazma
repo_, navigate to the package directory and run:

.. code-block:: bash

    pip install .

``hazma`` computes its spectra in a compiled Rust extension, ``hazma._core``,
so a source build needs a Rust toolchain: ``cargo`` on your ``PATH`` and
``rustc`` 1.85 or newer. Install one from rustup_. ``pip`` cannot supply it,
and without it the build fails before any python code runs.

A source build also needs your platform's native linker, because ``rustc``
links through the system C toolchain driver rather than shipping its own:
the Xcode command line tools on macOS (``xcode-select --install``), a C
toolchain such as ``build-essential`` on Linux, or the MSVC build tools on
Windows. This is a Rust requirement, not a hazma one — ``hazma`` itself no
longer contains any C or C++ source to compile.

.. _repo: https://github.com/LoganAMorrison/Hazma.git
.. _rustup: https://rustup.rs
