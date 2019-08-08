Installation
============

``hazma`` was developed for python3. Before installing ``hazma``, the user
needs to install several well-established python packages: ``cython``,
``scipy``, ``numpy``, and ``scikit-image``. Theses are easily installed by using
PyPI. If the user has PyPI installed on their system, then these packages
can be installed using:

.. code-block:: bash

    pip install cython, scipy, numpy, scikit-image, matplotlib

``hazma`` can be installed in the same way, using:

.. code-block:: bash

    pip install hazma

This will download a tarball from the PyPI repository, compile all the
c-code and install ``hazma`` on the system. Alternatively, the user can
install ``hazma`` by downloading the package from Hazma repo_. Once
downloaded, navigate to the package directory using the command line and
run either:

.. code-block:: bash

    pip install .

or:

.. code-block:: bash

    python setup.py install

Note that since ``hazma`` makes extensive usage of the package
``cython``, the user will need to have a ``c`` and ``c++`` compiler installed on
their system (for example ``gcc`` and ``g++`` on unix-like systems or
Microsoft Visual Studios 2015 or later on Windows). For more information,
see the Cython_ installation guide.


.. _repo: https://github.com/LoganAMorrison/Hazma.git
.. _Cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
