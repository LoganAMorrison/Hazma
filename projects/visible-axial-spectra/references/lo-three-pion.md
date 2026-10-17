# LO neutral isovector three-pion current

This note defines the tree-level SU(2), exact-isospin provider implemented
in `hazma/form_factors/axial/_three_pi.py`. The common pion mass defaults
to Hazma's neutral pion mass, and the decay constant is 92.2 MeV. This
choice does not mix charged and neutral physical masses within one LO
amplitude. Zero common mass is also supported for chiral-limit tests.
The declared support ends at 600 MeV and can be narrowed. This endpoint
is a restricted LO implementation domain, not a precision prediction
for the truncation error of chiral perturbation theory.

## Action, source and field coordinates

Use metric (+---), dimensionless flavor source
\(G_A=g_3\operatorname{diag}(1,-1,0)\), and the SU(2) O(p²) action

\[
\mathcal L_2=\frac{F^2}{4}\operatorname{Tr}
(D_\mu U D^\mu U^\dagger)
+\frac{F^2m^2}{4}\operatorname{Tr}(U+U^\dagger),
\qquad
D_\mu U=\partial_\mu U-i a_\mu U-i U a_\mu.
\]

The embedded source uses Pauli matrices, not half-Pauli matrices. Thus
its one-pion coupling is twice the conventional current with generator
\(\tau_3/2\). Choose the common external-state phase so the tree-level
matrix elements below are real. An overall common phase changes no
observable or Ward-identity magnitude.

In sigma coordinates,
\(U=(\sigma+i\tau\cdot\phi)/F\),
\(\sigma=\sqrt{F^2-\phi^2}\), the Noether current with this phase is

\[
J_A^3=2g_3(\sigma\partial\phi_3-\phi_3\partial\sigma)
=2g_3\left[F\partial\phi_3+
\frac{\phi_3\phi\cdot\partial\phi
-\phi^2\partial\phi_3/2}{F}\right]+O(\phi^5).
\]

The same action contains the four-pion terms

\[
\mathcal L_{4\pi}=
\frac{(\phi\cdot\partial\phi)^2}{2F^2}
-\frac{m^2(\phi^2)^2}{8F^2}.
\]

Take all three final pions on shell at the same mass, let their total
momentum be \(q\), and define \(s=(p_++p_-)^2\). The cubic contact
term for the charged channel is \(2g_3(q-2p_0)/F\). Attaching the
linear source-pion vertex to the off-shell four-pion vertex gives
\(-2g_3q(s-m^2)/[F(q^2-m^2)]\). Their sum is

\[
H^\mu_{+-0}=\frac{2g_3}{F}
\left[-2p_0^\mu+q^\mu\frac{q^2-s}{q^2-m^2}\right].
\]

For three neutral pions, the contact is \(2g_3q/F\), while the
four-pion vertex is \(q^2/F^2\). Thus

\[
H^\mu_{000}=-\frac{2g_3m^2q^\mu}{F(q^2-m^2)}.
\]

These are the full currents, including their components parallel to
\(q\). Only after contraction with a physical on-shell polarization
can the charged amplitude be reduced to \(-4g_3\epsilon\cdot p_0/F\).
That reduced expression cannot compute the source-longitudinal density.

An independent field-coordinate check uses
\(U=\exp(i\tau\cdot\pi/F)\). In these coordinates the charged contact
is \(4g_3(q-3p_0)/(3F)\), and the off-shell four-pion vertex is
\((3s-q^2-2m^2)/(3F^2)\). Attaching the same linear source and propagator
reproduces the charged expression above. The neutral contact vanishes
in exponential coordinates; its four-pion vertex is \(m^2/F^2\),
reproducing the neutral current entirely through the pion pole.

## Ward identity and chiral limit

Contraction gives

\[
q\cdot H_{+-0}=-\frac{2g_3m^2(s-m^2)}{F(q^2-m^2)},
\qquad
q\cdot H_{000}=-\frac{2g_3m^2q^2}{F(q^2-m^2)}.
\]

This is the tree-level pseudoscalar Ward identity: in sigma coordinates
the pseudoscalar density is linear in \(\phi_3\), and its three-pion
matrix element is obtained by attaching the same four-pion vertex and
pion propagator. The explicit breaking coefficient supplies \(m^2\).
There is no flavor singlet in this current and no gluonic anomaly term.
An octet, singlet, or generic diagonal source is rejected, not projected
silently onto this direction.

Both divergences vanish as the common pion mass goes to zero. In the
charged channel a soft neutral pion gives \(s=q^2\), so the full
current vanishes (Adler zero). This does not claim an Adler zero when a
charged pion is soft; that limit has a different current-algebra term.
At positive mass, the pion pole \(q^2=m^2\) lies below the three-pion
threshold and is never evaluated by the integrated provider.

## Spectral normalization and identical particles

The pointwise spin-averaged transverse and source-longitudinal weights
are

\[
T=\frac{-H\cdot H^*+|q\cdot H|^2/q^2}{3},
\qquad L=\frac{|q\cdot H|^2}{q^2}.
\]

For the charged channel this gives

\[
T_{+-0}=\frac{4g_3^2\lambda(q^2,s,m^2)}{3F^2q^2},
\qquad
L_{+-0}=\frac{4g_3^2m^4(s-m^2)^2}
{F^2q^2(q^2-m^2)^2}.
\]

For the neutral channel, \(T_{000}=0\) exactly and
\(L_{000}=4g_3^2m^4q^2/[F^2(q^2-m^2)^2]\). All expressions are
dimensionless before integration. The invariant phase measure follows
by factorizing three-body phase space into two two-body phase spaces:

\[
\frac{d\Phi_3}{ds}=
\frac{\sqrt{\lambda(q^2,s,m^2)}\sqrt{1-4m^2/s}}
{128\pi^3q^2},\qquad 4m^2\leq s\leq(\sqrt{q^2}-m)^2.
\]

The density above is dimensionless, while \(ds\) and the integrated
phase volume have units MeV². The stored spectra are
\(A_T=\int d\Phi_3 T\), \(A_L=\int d\Phi_3 L\), also in MeV².
The neutral integral includes one factor \(1/3!\); the charged channel
has no identical-particle factor. Neither has a Gram-matrix factor
of one half. The physical spin-1 width is \(A_T/(2\sqrt{q^2})\), in
MeV. Consequently the neutral width vanishes even when its source
\(A_L\) is nonzero. Physical helicity zero belongs to \(A_T\).

Writing \(Q=\sqrt{q^2}\), the analytic massless limits are

\[
\Phi_3=\frac{Q^2}{256\pi^3},\qquad
A_{T,+-0}=\frac{g_3^2Q^4}{384\pi^3F^2},\qquad A_L=0.
\]

The massive implementation integrates one invariant with deterministic
Gauss-Legendre quadrature and an endpoint-smoothing cosine substitution.
Tests compare it with Hazma's maintained adaptive two-dimensional
`ThreeBody` phase-space integrator, using the independently expanded
exponential-coordinate current. The massless expression is evaluated
analytically, also avoiding underflow-induced division by zero at tiny
positive energies. Exact and subthreshold energies return zero.

## Supplemental research oracle

The pinned test values in
`test/form_factors/axial/test_three_pi.py` were generated read-only on
2026-10-12 from the separate research checkout's
`version3/integrated/ax_xsec.py`. This engine assembles symbolic vertices
from `version3/integrated/ax_feynman_rules.json` into tree amplitudes.
Production Hazma imports none of the research code or SymPy.

The SHA-256 values at capture were:

| Research file under `version3/integrated/` | SHA-256 |
| --- | --- |
| `ax_xsec.py` | `0dd4f7e47ec50e8fbda6c7621eb8c1e9c0ce276592039b7069ba0c43ae9f7020` |
| `ax_feynman_rules.json` | `f96115bed43f2ff4dc2448e2f9a7b6328077a8a166444d36a25b42abe189e840` |
| `rule_selection.py` | `0879261ddde5a4f9cf5a595b773b52c1f230ced6b90fd52743fcfc112eb20110` |

The setup used `AxParams(model="chpt", gau=1, gad=-1, gas=0, f=92.2)`
and set each of `MASS["pi0"]`, `MASS["pip"]`, and `MASS["pim"]` to
135 MeV. The neutral pion width was set to zero in the propagator,
matching the tree-level real pole. Values come from
`Model.spectral("Zp", final, q, n=48)` with final states
`("pip", "pim", "pi0")` and `("pi0", "pi0", "pi0")`. Orders 32 and
48 agreed to less than 6e-15 relative for the nonzero entries.

| Q (MeV) | Charged A_T (MeV²) | Charged A_L (MeV²) | Neutral A_L (MeV²) |
| --- | --- | --- | --- |
| 450 | 1.5906418165653142 | 0.06123401653855351 | 0.09098608979168443 |
| 500 | 15.231055501000954 | 0.2071752550990431 | 0.30090781143351225 |
| 600 | 136.87334329168735 | 0.5610413513667469 | 0.7726880053721584 |

The oracle's neutral transverse entries were below 4e-34 MeV² and are
analytically zero. The regression tolerance is 1e-11 relative, allowing
platform roundoff without absorbing a normalization discrepancy. This
oracle supplements the independent action, Ward, phase-measure, and
massless checks; it is not their sole justification.
