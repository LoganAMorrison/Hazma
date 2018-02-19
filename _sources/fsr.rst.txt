Final State Radiation
=====================

Along with computing decay spectra, hazma is able to compute final state radiation spectra from decays of off-shell mediators (scalar, psuedo-scalar, vector or axial-vector.) The relavent diagrams for such processes are

+-------------------------------------+------------------------------------+
| .. figure:: _images/scalar-fsr.png  | .. figure:: _images/vector-fsr.png |
|                                     |                                    |
| (a) Scalar mediator                 | (b) Vector mediator                |
+-------------------------------------+------------------------------------+

Computing the matrix elements squared of these diagrams (including diagrams with the photon attached to the other fermion leg) and integrating over all variables except the photon energy yields :math:`d\sigma(M^*\to\mu^{+}\mu^{-}\gamma)/dE`. To compute :math:`dN/dE`, we divide :math:`d\sigma(M^*\to\mu^{+}\mu^{-}\gamma)/dE` by :math:`\sigma(M^*\to\mu^{+}\mu^{-})`.

.. image:: _images/muon_fsr.png
   :alt: Gamma ray spectrum from radiative muon decay
   :align: center
   :width: 800px
   :height: 800px
