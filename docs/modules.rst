*******
Modules
*******

The main modules of Hazma are the particle modules: electron, muon, charged pion, neutral pion, charged kaon and the neutral kaons. Each of these particle modules has two functions ``decay_spectra`` and ``fsr`` which produce the gamma ray spectra from radiative decays and final state radiation, respectively.

Particles (``hazma.particles``)
===============================

Muon (``hazma.particles.muon``)
-------------------------------

.. py:function:: muon.decay_spectra(eng_gam, eng_mu)

    Returns the photon spectrum :math:`dN/dE` from a muon decay. The dominant contribution is from :math:`\mu\to e\nu_{e}\nu_{\mu}\gamma`.

    :param eng_gam: A float or ``numpy.ndarray`` containing gamma ray energy(ies) to evaluate the spectrum.
    :type eng_gam: float or numpy.ndarray
    :param float eng_mu: Energy of the muon.
    :return: Returns the spectrum evaluated at (eng_gam, eng_mu).
    :rtype: float or numpy.ndarray

.. py:function:: muon.fsr(eng_gam, eng_mu, mediator)

    Returns the photon spectrum :math:`dN/dE` from muon final state radiation given a mediator :math:`M`: :math:`M^{*}\to\mu^{+}\mu^{-}\gamma`.


    :param eng_gam: A float or ``numpy.ndarray`` containing gamma ray energy(ies) to evaluate the spectrum.
    :type eng_gam: float or numpy.ndarray
    :param float eng_mu: Energy of the muon.
    :param str mediator: Mediator type: 'scalar', 'pseudo-scalar', 'vector', 'axial-vector'
    :return: Returns the spectrum evaluated at (eng_gam, eng_mu).
    :rtype: float or numpy.ndarray


Charged Pion (``hazma.particles.charged_pion``)
-----------------------------------------------

.. py:function:: charged_pion.decay_spectra(eng_gam, eng_pi)

    Returns the photon spectrum :math:`dN/dE` from a charged pion decay. The dominant contribution is from :math:`\pi\to\mu\nu_{\mu}\to e\nu_{\mu}\nu_{\mu}\nu_{e}\gamma`.

    :param eng_gam: A float or ``numpy.ndarray`` containing gamma ray energy(ies) to evaluate the spectrum.
    :type eng_gam: float or numpy.ndarray
    :param float eng_pi: Energy of the charged pion.
    :return: Returns the spectrum evaluated at (eng_gam, eng_pi).
    :rtype: float or numpy.ndarray

.. py:function:: charged_pion.fsr(eng_gam, eng_pi, mediator)

    Returns the photon spectrum :math:`dN/dE` from charged pion final state radiation given a mediator :math:`M`: :math:`M^{*}\to\pi^{+}\pi^{-}\gamma`.


    :param eng_gam: A float or ``numpy.ndarray`` containing gamma ray energy(ies) to evaluate the spectrum.
    :type eng_gam: float or numpy.ndarray
    :param float eng_mu: Energy of the charged pion.
    :param str mediator: Mediator type: 'scalar', 'pseudo-scalar', 'vector', 'axial-vector'
    :return: Returns the spectrum evaluated at (eng_gam, eng_pi).
    :rtype: float or numpy.ndarray


Neutral Pion (``hazma.particles.neutral_pion``)
-----------------------------------------------

.. py:function:: neutral_pion.decay_spectra(eng_gam, eng_pi)

    Returns the photon spectrum :math:`dN/dE` from a neutral pion decay. The dominant contribution is from :math:`\pi\to\gamma\gamma`.

    :param eng_gam: A float or ``numpy.ndarray`` containing gamma ray energy(ies) to evaluate the spectrum.
    :type eng_gam: float or numpy.ndarray
    :param float eng_pi: Energy of the neutral pion.
    :return: Returns the spectrum evaluated at (eng_gam, eng_pi).
    :rtype: float or numpy.ndarray

.. py:function:: charged_pion.fsr(eng_gam, eng_pi, mediator)

    Returns zero.

Electron (``hazma.particles.electron``)
---------------------------------------

.. py:function:: electron.decay_spectra(eng_gam, eng_e)

    Returns zero. Electron is stable.

.. py:function:: electron.fsr(eng_gam, eng_pi, mediator)

    Returns the photon spectrum :math:`dN/dE` from electron final state radiation given a mediator :math:`M`: :math:`M^{*}\to e^{+}e^{-}\gamma`.


    :param eng_gam: A float or ``numpy.ndarray`` containing gamma ray energy(ies) to evaluate the spectrum.
    :type eng_gam: float or numpy.ndarray
    :param float eng_mu: Energy of the electron.
    :param str mediator: Mediator type: 'scalar', 'pseudo-scalar', 'vector', 'axial-vector'
    :return: Returns the spectrum evaluated at (eng_gam, eng_e).
    :rtype: float or numpy.ndarray
