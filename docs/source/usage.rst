.. _usage:

Usage
=====

The user has several options for tapping into the resources provided by
``hazma``. The easiest is to use one of the built-in simplified models,
where a user only needs to specify the parameters of the model. If the
user is working with a model which specializes on one of the simplified
models, they can define their own class and inherit from one of the
simplified models, obtaining all of the functionality of the built in
models (such as final state radiation (FSR) spectra, cross sections,
mediator decay widths, etc.) while supplying the user with a simpler,
more specialized interface to the underlying models. For a detailed
explanations of how these two options are done, see the `basic usage`
section below. Another option is for the user to define their own model.
To do this, they need to define a class which contains functions for the
gamma-ray and positron spectra, as well as the annihilation cross sections
and branching fractions. In the `advanced usage` section, we provide a
detailed example for this case.

.. _basic_usage:

Basic Usage
-----------

.. _using_simplified_models:

Using Simplified Models
^^^^^^^^^^^^^^^^^^^^^^^

Here we give a compact overview of how to use the built in simplified models
in ``hazma``. All the models built into ``hazma`` have identical interfaces.
The only difference in the interfaces is the parameters which need to be
specified for the particular model being used. Thus, we will only show the
usage of one of the models. The others can be used in an identical fashion.
The example we will use is the ``KineticMixing`` model.
To create a ``KineticMixing`` model, we use:

.. code-block:: python

    # Import the model
    >>> from hazma.vector_mediator import KineticMixing
    # Specify the parameters
    >>> params = {'mx': 250.0, 'mv': 1e6, 'gvxx': 1.0, 'eps': 1e-3}
    # Create KineticMixing object
    >>> km = KineticMixing(**params)

Here we have created a model with a dark matter fermion of mass
250 MeV, a vector mediator which mixes with the standard model with a
mass of 1 TeV. We set the coupling of the vector mediator to the dark
matter, :math:`g_{V\chi} = 1` and set the kinetic mixing parameter
:math:`\epsilon = 10^{-3}`. To list all of the available final states for
which the dark matter can annihilate into, we use:

.. code-block:: python

    >>> km.list_annihilation_final_states()
    ['mu mu', 'e e', 'pi pi', 'pi0 g', 'pi0 v', 'v v']

This tells us that we can potentially annihilate through:
:math:`\bar{\chi}\chi\to\mu^{+}\mu^{-},e^{+}e^{-},
\pi^{+}\pi^{-},\pi^{0}\gamma,\pi^0{0}V` or :math:`V V` (the two-mediator
final state). However, which of these final states is actually available
depends on the center of mass energy. We can see this fact by looking at
the annihilation cross sections or branching fractions, which can be
computed using:

.. code-block:: python

    >>> cme = 2.0 * km.mx * (1.0 + 0.5 * 1e-6)
    >>> km.annihilation_cross_sections(cme)
    {'mu mu': 8.94839775021393e-25,
     'e e': 9.064036692829845e-25,
     'pi pi': 1.2940469635262499e-25,
     'pi0 g': 5.206158864833925e-29,
     'pi0 v': 0.0,
     'v v': 0.0,
     'total': 1.9307002022456507e-24}
    >>> km.annihilation_branching_fractions(cme)
    {'mu mu': 0.46347940191883763,
     'e e': 0.4694688839980031,
     'pi pi': 0.06702474894968717,
     'pi0 g': 2.6965133472190545e-05,
     'pi0 v': 0.0,
     'v v': 0.0}

Here we have chosen a realistic center of mass energy for dark matter in
our galaxy, which as a velocity dispersion of :math:`\sigma_v \sim 10^{-3}`.
We can that the :math:`V V` final state is unavailable, as it
should be since the vector mediator mass is too heavy. In this theory, the
vector mediator can decay. If we would like to know the decay width and the
partial widths, we can use:

.. code-block:: python

    >>> km.partial_widths()
    {'pi pi': 0.0018242846671063036,
     'pi0 g': 2.1037425397685694,
     'x x': 79577.47154594581,
     'e e': 0.007297139521307648,
     'mu mu': 0.007297139521307642,
     'total': 79579.5917070493}

If we would like to know the gamma-ray spectrum from dark matter
annihilations, we can use:

.. code-block:: python

    >>> photon_energies = np.array([cme/4])
    >>> km.spectra(photon_energies, cme)
    {'mu mu': array([2.94759389e-05]),
     'e e': array([0.00013171]),
     'pi pi': array([2.20142244e-06]),
     'pi0 g': array([2.29931655e-07]),
     'pi0 v': array([0.]),
     'v v': array([0.]),
     'total': array([0.00016362])}

Note that we only used a single photon energy because of display purposes,
but in general the user can specify any number of photon energies. If the
user would like access to the underlying spectrum functions so they can call
them repeatedly, they can use:

.. code-block:: python

    >>> spec_funs = km.spectrum_functions()
    >>> spec_funs['mu mu'](photon_energies, cme)
    [6.35970849e-05]
    >>> mumu_bf = km.annihilation_branching_fractions(cme)['mu mu']
    >>> mumu_bf * spec_funs['mu mu'](photon_energies, cme)
    [2.94759389e-05]

Notice that the direct call to the spectrum function for
:math:`\bar{\chi}\chi\to\mu^{+}\mu^{-}` doesn't given the same result as
``km.spectra(photon_energies, cme)['mu mu']``. This is because the
branching fractions are not applied for the
``spec_funs = km.spectrum_funcs()``. If the user doesn't care about
the underlying components of the gamma-ray spectra, the can simply call:

.. code-block:: python

    >>> km.total_spectrum(photon_energies, cme)
    array([0.00016362])

to get the total gamma-ray spectrum. The reader may have caught the fact
that there is a gamma-ray line in the spectrum for
:math:`\bar{\chi}\chi\to\pi^{0}\gamma`. To get the location of this
monochromatic gamma-ray line, the user can run:

.. code-block:: python

    >>> km.gamma_ray_lines(cme)
    {'pi0 g': {'energy': 231.78145156177675, 'bf': 2.6965133472190545e-05}}

This tells us the process which produces the line, the location of the
line and the branching fraction for the process. We don't include the
line in the total spectrum since the line produces a Dirac-delta function.
In order to get a realistic spectrum including the line, we need to
convolve the gamma-ray spectrum with an energy resolution. This can be
achieved using:

.. code-block:: python

    >>> min_photon_energy = 1e-3
    >>> max_photon_energy = cme
    >>> energy_resolution = lambda photon_energy : 1.0
    >>> number_points = 1000
    >>> spec = km.total_conv_spectrum_fn(min_photon_energy, max_photon_energy,
    ...                                  cme, energy_resolution, number_points)
    >>> spec(cme / 4)  # compute the spectrum at a photon energy of `cme/4`
    array(0.001718)

The ``km.total_conv_spectrum_fn`` computes and returns an
interpolating function of the convolved function. An important thing to
note here is that the ``km.total_conv_spectrum_fn`` takes in a function
for the energy resolution. This allows the user to define the energy
resolution to depend on the specific photon energy. Such a dependence is
common for gamma-ray telescopes. Next we present the positron spectra.
These have an identical interface to the gamma-ray spectra, so we only
show how to call the functions and we suppress the output

.. code-block:: python

    >>> from hazma.parameters import electron_mass as me
    >>> positron_energies = np.logspace(np.log10(me), np.log10(cme), num=100)
    >>> km.positron_spectra(positron_energies, cme)
    >>> km.positron_lines(cme)
    >>> km.total_positron_spectrum(positron_energies, cme)
    >>> dnde_pos = km.total_conv_positron_spectrum_fn(min(positron_energies),
    ...                                               max(positron_energies),
    ...                                               cme,
    ...                                               energy_resolution,
    ...                                               number_points)

The last thing that we would like to demonstrate is how to compute
limits. In order to compute the limits on the annihilation cross section
of a model from a gamma-ray telescope, say EGRET, we can use:

.. code-block:: python

    >>> from hazma.gamma_ray_parameters import egret_diffuse
    # Choose DM masses from half the electron mass to 250 MeV
    >>> mxs = np.linspace(me/2., 250., num=10)
    # Compute limits from e-ASTROGAM
    >>> limits = np.zeros(len(mxs), dtype=float)
    >>> for i, mx in enumerate(mxs):
    ...     km.mx = mx
    ...     limits[i] = km.binned_limit(egret_diffuse)

Similarly, if we would like to set constraints using e-ASTROGAM, one can
use:

.. code-block:: python

    # Import target and background model for the e-ASTROGAM telescope
    >>> from hazma.gamma_ray_parameters import gc_target, gc_bg_model
    # Choose DM masses from half the electron mass to 250 MeV
    >>> mxs = np.linspace(me/2., 250., num=10)
    # Compute limits from e-ASTROGAM
    >>> limits = np.zeros(len(mxs), dtype=float)
    >>> for i, mx in enumerate(mxs):
    ...     km.mx = mx
    ...     limits[i] = km.unbinned_limit(target_params=gc_target,
    ...                                   bg_model=gc_bg_model)

.. _subclassing_the_simplified_models:

Subclassing the Simplified Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The user might not be
interested in the generic simplified models built into ``hazma``,
but instead a more specialized model. In this case, it makes sense for
the user to subclass one of the simplified models (i.e. create a class
which inherits from one of the simplified models.) As and example, we
illustrate how to do this with the Higgs-portal model (of course this
model is already built into ``hazma``, but it works nicely as an
example.) Recall that the full set of parameters for the scalar mediator
model are:

1. :math:`m_{\chi}`: dark matter mass,
2. :math:`m_{S}`: scalar mediator mass,
3. :math:`g_{S\chi}`: coupling of scalar mediator to dark matter,
4. :math:`g_{Sf}`: coupling of scalar mediator to standard model fermions,
5. :math:`g_{SG}`: effective coupling of scalar mediator to gluons,
6. :math:`g_{SF}`: effective coupling of scalar mediator to photons and
7. :math:`\Lambda`: cut-off scale for the effective interactions.

In the case of the Higgs-portal model, the scalar mediator talks to the
standard model only through the Higgs boson, i.e. it mixes with the Higgs.
Therefore, the scalar mediator inherits its interactions with the standard
model fermions, gluons and photon through the Higgs. In the Higgs-portal
model, the relevant parameters are:

1. :math:`m_{\chi}`: dark matter mass,
2. :math:`m_{S}`: scalar mediator mass,
3. :math:`g_{S\chi}`: coupling of scalar mediator to dark matter,
4. :math:`\sin\theta`: the mixing angle between the scalar mediator and
   the Higgs,

The remaining parameters can be deduced from these using:

.. math::

    g_{Sf} = \sin\theta,  g_{SG} = 3\sin\theta, g_{SF} = -\frac{5}{6}\sin\theta,  \Lambda = v_{h}.

Below, we construct a class which subclasses the scalar mediator class
to implement the Higgs-portal model.

.. code-block:: python

    from hazma.scalar_mediator import ScalarMediator
    from hazma.parameters import vh

    class HiggsPortal(ScalarMediator):
        def __init__(self, mx, ms, gsxx, stheta):
            self._lam = vh
            self._stheta = stheta
            super(HiggsPortal, self).__init__(mx, ms, gsxx, stheta, 3.*stheta,
                                              -5.*stheta/6., vh)

        @property
        def stheta(self):
            return self._stheta

        @stheta.setter
        def stheta(self, stheta):
            self._stheta = stheta
            self.gsff = stheta
            self.gsGG = 3. * stheta
            self.gsFF = - 5. * stheta / 6.

        # Hide underlying properties' setters
        @ScalarMediator.gsff.setter
        def gsff(self, gsff):
            raise AttributeError("Cannot set gsff")

        @ScalarMediator.gsGG.setter
        def gsGG(self, gsGG):
            raise AttributeError("Cannot set gsGG")

        @ScalarMediator.gsFF.setter
        def gsFF(self, gsFF):
            raise AttributeError("Cannot set gsFF")

There are a couple things to note about our above implementation. First,
our model only takes in :math:`m_{\chi}`, :math:`m_{S}`, :math:`g_{S\chi}`
and :math:`\sin\theta`, as desired. But the underlying model, i.e. the
``ScalarMediator`` model only knows about :math:`m_{\chi}`, :math:`m_{S}`,
:math:`g_{S\chi}`, :math:`g_{Sf}`, :math:`g_{SG}`, :math:`g_{SF}` and
:math:`\Lambda`. So if we update :math:`\sin\theta`, we additionally need
to update the underlying parameters, :math:`g_{Sf}`, :math:`g_{SG}`,
:math:`g_{SF}` and :math:`\Lambda`. The easiest way to do this is using
getters and setters by defining :math:`\sin\theta` to be a ``property``
through the ``@property`` decorator. Then every time we update
:math:`\sin\theta`, we can also update the underlying parameters. The
second thing to note is that we want to make sure we don't accidentally
change the underlying parameters directly, since in this model, they are
only defined through :math:`\sin\theta`. We an ensure that we cannot
change the underlying parameters directly by overriding the getters and
setters for ``gsff``, ``gsGG`` and ``gsGG`` and raising an error if we
try to change them. This isn't strictly necessary (as long as the user is
careful), but can help avoid confusing behavior.

.. _advanced_usage:

Advanced Usage
--------------

.. _adding_new_gamma_ray_experiments:

Adding New Gamma-Ray Experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently ``hazma`` only includes information for producing projected
unbinned limits with e-ASTROGAM, using the dwarf Draco or inner
:math:`10^\circ\times10^\circ` region of the Milky Way as a target.
Adding new detectors and target regions is straightforward. A detector is
characterized by the effective area :math:`A_{\mathrm{eff}}(E)`, the
energy resolution :math:`\epsilon(E)` and observation time
:math:`T_{\mathrm{obs}}`. In ``hazma``, the first two can be any callables
(functions) and the third must be a float. The region of interest is
defined by a ``TargetParams`` object, which can be instantiated
with:

.. code-block:: python

    >>> from hazma.gamma_ray_parameters import TargetParams
    >>> tp = TargetParams(J=1e29, dOmega=0.1)

The background model should be packaged in an object of type
``BackgroundModel``. This light-weight class has a function
``dPhi_dEdOmega()`` for computing the differential photon flux per solid
angle (in :math:`\mathrm{MeV}^{-1}\mathrm{sr}`) and an attribute
``e_range`` specifying the energy range over which the model is valid
(in MeV). New background models are defined by passing these two the

.. code-block:: python

    >>> from hazma.background_model import BackgroundModel
    >>> bg = BackgroundModel(e_range=[0.5, 1e4],
    ...                      dPhi_dEdOmega=lambda e: 2.7e-3 / e**2)


Gamma-ray observation information from Fermi-LAT, EGRET and COMPTEL is
included with ``hazma``, and other observations can be added using the
container class ``FluxMeasurement``. The initializer requires:

#. The name of a CSV file containing gamma-ray observations. The file's
   columns must contain:

   #. Lower bin edge (MeV)
   #. Upper bin edge (MeV)
   #. :math:`E^n d^2\Phi / dE\, d\Omega` (in :math:`\mathrm{MeV}^{n-1}
      \mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{sr}^{-1}`)
   #. Upper error bar (in :math:`\mathrm{MeV}^{n-1} \mathrm{cm}^{-2}
      \mathrm{s}^{-1} \mathrm{sr}^{-1}`)
   #. Lower error bar (in :math:`\mathrm{MeV}^{n-1} \mathrm{cm}^{-2}
      \mathrm{s}^{-1} \mathrm{sr}^{-1}`)

   Note that the error bar values are their :math:`y`-coordinates, not
   their relative distances from the central flux.
#. The detector's energy resolution function.
#. A ``TargetParams`` object for the target region.

For example, a CSV file ``obs.csv`` containing observations

+-----------+-----------+------------------------------------+-------------+-------------+
| lower bin | upper bin | :math:`E^n d^2\Phi / dE\, d\Omega` | upper error | lower error |
+===========+===========+====================================+=============+=============+
| 150.      | 275.0     | 0.0040                             | 0.0043      | 0.0038      |
+-----------+-----------+------------------------------------+-------------+-------------+
| 650.      | 900.0     | 0.0035                             | 0.0043      | 0.003       |
+-----------+-----------+------------------------------------+-------------+-------------+

with :math:`n=2` for an instrument with energy resolution
:math:`\epsilon(E) = 0.05` observing the target region ``tp`` defined
above can be loaded using [1]_:

.. code-block:: python

    >>> from hazma.flux_measurement import FluxMeasurement
    >>> obs = FluxMeasurement("obs.dat", lambda e: 0.05, tp)

The attributes of the ``FluxMeasurement`` store all of the provide
information, with the :math:`E^n` prefactor removed from the flux and
error bars, and the errors converted from the positions of the error bars
to their sizes. These are used internally by the ``Theory.binned_limit()``
method, and can be accessed as follows:

.. code-block:: python

    >>> obs.e_lows, obs.e_highs
    (array([150., 650.]), array([275., 900.]))
    >>> obs.target
    <hazma.gamma_ray_parameters.TargetParams at 0x1c1bbbafd0>
    >>> obs.fluxes
    array([8.85813149e-08, 5.82726327e-09])
    >>> obs.upper_errors
    array([6.64359862e-09, 1.33194589e-09])
    >>> obs.lower_errors
    array([4.42906574e-09, 8.32466181e-10])
    >>> obs.energy_res(10.)
    0.05

.. _user_defined_models:

User-Defined Models
^^^^^^^^^^^^^^^^^^^

In this subsection, we demonstrate how to implement new models in Hazma. A notebook containing all th.. code in this appendix can be downloaded from GitHub HazmaExample_. The model we will consider is an effective field theory with a Dirac fermion DM particle which talks to neutral and charged pions through gauge-invariant dimension-5 operators. The Lagrangian for this model is:

.. math::

    \mathcal{L} \supset \frac{c_1}{\Lambda}\overline{\chi}\chi\pi^{+}\pi^{-}+\frac{c_2}{\Lambda}\overline{\chi}\chi\pi^{0}\pi^{0}

where :math:`c_{1}, c_{2}` are dimensionless Wilson coefficients and :math:`\Lambda` is the cut-off scale of the theory. In order to implement this model in Hazma, we need to compute the annihilation cross sections and the FSR spectra. The annihilation channels for this model are simply :math:`\bar{\chi}\chi\to\pi^{0}\pi^{0}` and :math:`\bar{\chi}\chi\to\pi^{+}\pi^{-}`. The computations for the cross sections are straight forward and yield:

.. math::

    \sigma(\bar{\chi}\chi\to\pi^{+}\pi^{-}) = \frac{c_1^2 \sqrt{1-4 \mu _{\pi }^2} \sqrt{1-4 \mu _{\chi }^2}}{32 \pi \Lambda^2}\\
    \sigma(\bar{\chi}\chi\to\pi^{0}\pi^{0}) = \frac{c_2^2 \sqrt{1-4 \mu_{\pi^{0}}^2} \sqrt{1-4 \mu_{\chi}^2}}{8 \pi \Lambda^2}

where :math:`Q` is the center of mass energy, :math:`\mu_{\chi} = m_{\chi}/Q`, :math:`\mu_{\pi} = m_{\pi^{\pm}}/Q` and :math:`\mu_{\pi^{0}} = m_{\pi^{0}}/Q`. In addition to the cross sections, we need the FSR spectrum for :math:`\overline{\chi}\chi\to\pi^{+}\pi^{-}\gamma`. This is:

.. math::

    \frac{dN(\bar{\chi}\chi\to\pi^{+}\pi^{-}\gamma)}{dE_{\gamma}} = \frac{\alpha  \left(2 f(x)-2\left(1-x-2 \mu_{\pi} ^2\right)
   \log \left(\frac{1-x-f(x)}{1-x+f(x)}\right)\right)}{\pi\sqrt{1-4 \mu_{\pi} ^2} x}

where

.. math::

    f(x) = \sqrt{1-x} \sqrt{1-x-4 \mu_{\pi} ^2}

We are now ready to set up the Hazma model. For ``hazma`` to work properly, we will need to define the following functions in our model:

#. ``annihilation_cross_section_funcs()``: A function returning a ``dict`` of the annihilation cross sections functions, each of which take a center of mass energy.
#. ``spectrum_funcs()``: A function returning a ``dict`` of functions which take photon energies and a center of mass energy and return the gamma-ray spectrum contribution from each final state.
#. ``gamma_ray_lines(e_cm)``: A function returning a ``dict`` of the gamma-ray lines for a given center of mass energy.
#. ``positron_spectrum_funcs()``: Like ``spectrum_funcs()``, but for positron spectra.
#. ``positron_lines(e_cm)``: A function returning a ``dict`` of the electron/positron lines for a center of mass energy.

We find it easiest to place all of these components is modular classes and then combine all the individual classes into a master class representing our model. Before we begin writing the classes, we will need a few helper functions and constants from ``hazma``:

.. code-block:: python

    import numpy as np # NumPy is heavily used
    import matplotlib.pyplot as plt # Plotting utilities
    # neutral and charged pion masses
    from hazma.parameters import neutral_pion_mass as mpi0
    from hazma.parameters import charged_pion_mass as mpi
    from hazma.parameters import qe # Electric charge
    # Positron spectra for neutral and charged pions
    from hazma.positron_spectra import charged_pion as pspec_charged_pion
    # Deay spectra for neutral and charged pions
    from hazma.decay import neutral_pion, charged_pion
    # The `Theory` class which we will ultimately inherit from
    from hazma.theory import Theory

Now, we implement a cross section class:

.. code-block:: python

    class HazmaExampleCrossSection:
        def sigma_xx_to_pipi(self, Q):
            mupi = mpi / Q
            mux = self.mx / Q

            if Q > 2 * self.mx and Q > 2 * mpi:
                sigma = (self.c1**2 * np.sqrt(1 - 4 * mupi**2) *
                         np.sqrt(1 - 4 * mux**2)**2 /
                         (32.0 * self.lam**2 * np.pi))
            else:
                sigma = 0.0

            return sigma

        def sigma_xx_to_pi0pi0(self, Q):
            mupi0 = mpi0 / Q
            mux = self.mx / Q

            if Q > 2 * self.mx and Q > 2 * mpi0:
                sigma = (self.c2**2 * np.sqrt(1 - 4 * mux**2) *
                         np.sqrt(1 - 4 * mupi0**2) /
                         (8.0 * self.lam**2 * np.pi))
            else:
                sigma = 0.0

            return sigma

        def annihilation_cross_section_funcs(self):
            return {'pi0 pi0': self.sigma_xx_to_pi0pi0,
                    'pi pi': self.sigma_xx_to_pipi}

The key function is ``annihilation_cross_sections``, which is required to be implemented by ``hazma``. Next, we implement the spectrum functions which will produce the FSR and decay spectra:

.. code-block:: python

    class HazmaExampleSpectra:
        def dnde_pi0pi0(self, e_gams, e_cm):
            return 2.0 * neutral_pion(e_gams, e_cm / 2.0)

        def __dnde_xx_to_pipig(self, e_gam, Q):
            # Unvectorized function for computing FSR spectrum
            mupi = mpi / Q
            mux = self.mx / Q
            x = 2.0 * e_gam / Q
            if 0.0 < x and x < 1. - 4. * mupi**2:
                dnde = ((qe**2 * (2 * np.sqrt(1 - x) * np.sqrt(1 - 4*mupi**2 - x) +
                              (-1 + 2 * mupi**2 + x) *
                              np.log((-1 + np.sqrt(1 - x) * np.sqrt(1 - 4*mupi**2 - x) + x)**2/
                                     (1 + np.sqrt(1 - x)*np.sqrt(1 - 4*mupi**2 - x) - x)**2)))/
                    (Q * 2.0 * np.sqrt(1 - 4 * mupi**2) * np.pi**2 * x))
            else:
                dnde = 0

            return dnde

        def dnde_pipi(self, e_gams, e_cm):
            return (np.vectorize(self.__dnde_xx_to_pipig)(e_gams, e_cm) +
                    2. * charged_pion(e_gams, e_cm / 2.0))

        def spectrum_funcs(self):
            return {'pi0 pi0':  self.dnde_pi0pi0,
                    'pi pi':  self.dnde_pipi}

        def gamma_ray_lines(self, e_cm):
            return {}

Note the the second ``__dnde_xx_to_pipig`` is an unvectorized helper function, which is not to be used directly. Next we implement the positron spectra:

.. code-block:: python

    class HazmaExamplePositronSpectra:
        def dnde_pos_pipi(self, e_ps, e_cm):
            return pspec_charged_pion(e_ps, e_cm / 2.)

        def positron_spectrum_funcs(self):
            return {"pi pi": self.dnde_pos_pipi}

        def positron_lines(self, e_cm):
            return {}

Lastly, we group all of these classes into a master class and we're done:

.. code-block:: python

    class HazmaExample(HazmaExampleCrossSection,
                       HazmaExamplePositronSpectra,
                       HazmaExampleSpectra,
                       Theory):
        # Model parameters are DM mass: mx,
        # Wilson coefficients: c1, c2 and
        # cutoff scale: lam
        def __init__(self, mx, c1, c2, lam):
            self.mx = mx
            self.c1 = c1
            self.c2 = c2
            self.lam = lam

        @staticmethod
        def list_annihilation_final_states():
            return ['pi pi', 'pi0 pi0']

Now we can easily compute gamma-ray spectra, positron spectra and limit on our new model from gamma-ray telescopes. To implement our new model with :math:`m_{\chi} = 200~\mathrm{MeV}, c_{1} = c_{2} = 1` and :math:`\Lambda = 100~\mathrm{GeV}`, we can use:

.. code-block:: python

    >>> model = HazmaExample(200.0, 1.0, 1.0, 100e3)

To compute a gamma-ray spectrum:

.. code-block:: python

    # Photon energies from 1 keV to 1 GeV
    >>> egams = np.logspace(-3.0, 3.0, num=150)
    # Assume the DM is moving with a velocity of 10^-3
    >>> vdm = 1e-3
    # Compute CM energy assuming the above velocity
    >>> Q = 2.0 * model.mx * (1 + 0.5 * vdm**2)
    # Compute spectra
    >>> spectra = model.spectra(egams, Q)

Then we can plot the spectra using:

.. code-block:: python

    >>> plt.figure(dpi=100)
    >>> for key, val in spectra.items():
    ...     plt.plot(egams, val, label=key)
    >>> plt.xlabel(r'$E_{\gamma} (\mathrm{MeV})$', fontsize=16)
    >>> plt.ylabel(r'$\frac{dN}{dE_{\gamma}} (\mathrm{MeV}^{-1})$', fontsize=16)
    >>> plt.xscale('log')
    >>> plt.yscale('log')
    >>> plt.legend()

Additionally, we can compute limits on the thermally-averaged annihilation cross section of our model for various DM masses using

.. code-block:: python

    # Import target and background model for the E-Astrogam telescope
    >>> from hazma.gamma_ray_parameters import gc_target, gc_bg_model
    # Choose DM masses from half the pion mass to 250 MeV
    >>> mxs = np.linspace(mpi/2., 250., num=100)
    # Compute limits from E-Astrogam
    >>> limits = np.zeros(len(mxs), dtype=float)
    >>> for i, mx in enumerate(mxs):
    ...     model.mx = mx
    ...     limits[i] = model.unbinned_limit(target_params=gc_target,
    ...                                      bg_model=gc_bg_model)

.. _HazmaExample: https://github.com/LoganAMorrison/Hazma/blob/master/notebooks/hazma_paper/hazma_example.ipynb

.. [1] If the CSV containing the observations uses a different power of
       :math:`E` than :math:`n=2`, this can be specified using the
       ``power`` keyword argument to the initializer for ``FluxMeasurement``
