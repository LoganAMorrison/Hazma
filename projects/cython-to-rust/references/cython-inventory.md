# Reference: Cython inventory — dead-code map and live surface

**Audience:** Phase 00 (purge), Phases 04–06 (ports), and any task that
needs to know what exists, what is reachable, and what calls what.
**Nature:** Grounded facts.

Verified against Hazma 2.1.0 (August 2026) by reading every `.pyx`/`.pxd`
in full, cross-referenced with `setup.py`, the Python import graph, and
`test/conftest.py` collection rules. Re-verify line numbers before citing
in code review — this file records a snapshot.

## Headline numbers

- 44 `.pyx`/`.pxd` files, 11,613 lines total.
- `setup.py` builds **32 extension modules** in 11 groups; 3 groups
  (`_gamma_ray`, `_phase_space`, `field_theory_helper_functions`) compile
  as C++ (`-std=c++11`), the rest as C.
- **12 `.pyx` are never compiled** (all of `_decay/`,
  `_neutrino/neutrino.pyx`, `rh_neutrino/_rh_neutrino_fsr_four_body.pyx`,
  `spectra/_positron/_kaon.pyx`).
- Live surface: **~19 modules, ~5,000 live lines, 43 public entry points,
  behind 11 Python import statements.**
- Churn: 3 commits touched `.pyx`/`.pxd` in the 3 years to Aug 2026; two
  were toolchain-forced (NumPy 2.0 migration, build-system upgrade).

## Dead-code map (Phase 00 targets)

<!-- markdownlint-disable MD013 -- grounded-fact table; width is the content -->
| Target | Lines | Why it is safe to delete |
| --- | --- | --- |
| `hazma/_decay/*.pyx` + backups | ~1,850 | Not in `setup.py`; sole importer `hazma/__decay.py` is itself imported by nothing (the `hazma/__init__.py` block referencing it is commented out). `decay_charged_kaon.pyx` uses Cython-2 implicit relative cimports and cannot compile under Cython 3. **Keep** `parameters.pxd` until its includes are repointed (below). |
| `hazma/_positron/` (3 built mods) | 443 | `positron_decay.pyx:11-13` does `from hazma import rambo` — deleted module → ImportError at import today. Superseded by `hazma/spectra/_positron/`. Importer `hazma/__positron_spectra.py` is legacy; verify zero importers at delete time. |
| `hazma/_neutrino/` (2 built mods) | 492 | `muon.pyx` and `charged_pion.pyx` cimport plain cdef functions from `hazma._neutrino.neutrino`, which is **not built** → `__Pyx_ImportFunction` fails at import. Superseded by `hazma/spectra/_neutrino/`. |
| `hazma/_phase_space/` (3 built mods, C++) | 626 | Sole consumer is `hazma/deprecated/rambo.py` (imports at lines 22-24), which nothing imports. Live RAMBO is pure NumPy: `hazma/phase_space/` (`np.random.default_rng`, no compiled deps). |
| `hazma/_gamma_ray/` (2 built mods, C++) | 618 | `gamma_ray_generator.pyx:11-13` imports deleted `hazma.rambo` → broken at import; `gamma_ray_fsr` has zero Python importers and its C-function-pointer half exists solely for the never-built rh_neutrino file. |
| `hazma/rh_neutrino/_rh_neutrino_fsr_four_body.pyx` (+ stale `.pyi`) | 681 | Never built; the three references in `rh_neutrino/__init__.py:78-80` are commented out; live pure-Python replacements exist in `_rh_neutrino_fsr.py` (Altarelli–Parisi). |
| `hazma/spectra/_positron/_kaon.pyx` | 126 | Not in `setup.py`; references undefined names (`short_data`, `eta_data_*`) — would not compile. Live implementation is pure Python (`_kaon.py` + `_utils.py`). |
| `hazma/field_theory_helper_functions/three_body_phase_space.pyx` | 315 | Built, zero importers repo-wide. Pure scalar arithmetic. |
| `hazma/field_theory_helper_functions/common_functions.pyx` | 84 | 2 live functions: `cross_section_prefactor` (imported by broken `hazma/gamma_ray.py`; an algebraically identical pure-Python copy already exists at `hazma/utils.py:81`) and `minkowski_dot` (imported by `hazma/experimental/axial_vector_mediator/avm_msqrd.py:8` and the dead `_decay` data-regen script). Replace with pure Python, then delete. |
| `hazma/_decay/interpolation_data/` | ~1.05 MB | 21 `.dat` files + 3 stale top-level `.dat` + regen script, all serving the unbuilt `_decay` package. Shipped as package-data for nothing. |
| `hazma/deprecated/rambo.py` | — | Unimported (no `hazma/deprecated/__init__.py` exports; grep finds zero importers). NOTE: `hazma/deprecated/` removal policy is `major` per versioning.md — deleting `rambo.py` specifically needs the versioning call made explicitly in the purge PR. |
| `hazma/__decay.py`, `hazma/__positron_spectra.py`, `hazma/__neutrino_spectra.py` | — | Double-underscore legacy API shims; the `__init__.py` references are commented out. Verify zero external importers at delete time. |
| `hazma/_utils/boost.pyx` internal dead half | ~165 | `boost_integrate_linear_interp_massive`, `integrate_linear_interp_edge`, `integration_bounds`: not in `.pxd`, no `def` wrapper, and contain real index-pairing bugs (`boost.pyx:427,447,456`). Delete, never port. |
| `hazma/_positron/parameters.pxd` | 67 | Orphan — nothing includes it. |
<!-- markdownlint-enable MD013 -->

### Constants-header entanglement (must precede `_decay/` deletion)

Textual `include "../_decay/parameters.pxd"` (a filesystem-relative paste,
no compiled module involved) appears in **four live built modules**:

- `hazma/scalar_mediator/scalar_mediator_decay_spectrum.pyx:14`
- `hazma/scalar_mediator/scalar_mediator_positron_spec.pyx:11`
- `hazma/vector_mediator/vector_mediator_decay_spectrum.pyx:14`
- `hazma/vector_mediator/vector_mediator_positron_spec.pyx:12`

(plus the doomed `_gamma_ray/gamma_ray_generator.pyx:24`). Relocate the
file **verbatim** (see the constants-divergence rule in `../rules.md`)
to `hazma/_utils/legacy_parameters.pxd`, repoint the four includes, then
delete `_decay/`. `_decay/common.pxd` is included only by `_positron/`
modules and dies with them.

### `hazma.gamma_ray` decision (Phase 00, Task 0.5)

`hazma/gamma_ray.py` is a public-named module that cannot be imported
today (transitively imports deleted `hazma.rambo`). Options:
(a) rebuild its `gamma`/`gamma_point` over `hazma.phase_space` /
`hazma.spectra` machinery (keeps `version_bump: minor`);
(b) delete it (a `major` event per versioning.md, even though it is
already broken). Decide in Task 0.5; record as ADR-0003 if (b).

## Live surface (the port targets)

### Entry points by module

<!-- markdownlint-disable MD013 -- grounded-fact table; width is the content -->
| Module | Entry points (all scalar-or-1D-array in/out, float64) |
| --- | --- |
| `spectra/_photon/_muon` | `dnde_photon(egam, emu)` |
| `spectra/_photon/_pion` | `dnde_photon_charged_pion`, `dnde_photon_neutral_pion` |
| `spectra/_photon/_rho` | `dnde_photon_charged_rho`, `dnde_photon_neutral_rho` |
| `spectra/_photon/_kaon` | `dnde_photon_charged_kaon`, `dnde_photon_long_kaon`, `dnde_photon_short_kaon` |
| `spectra/_photon/_eta`, `_omega`, `_eta_prime`, `_phi` | `dnde_photon_eta`, `dnde_photon_omega`, `dnde_photon_eta_prime`, `dnde_photon_phi` |
| `spectra/_positron/_muon`, `_pion` | `dnde_positron_muon`, `dnde_positron_charged_pion` |
| `spectra/_neutrino/_muon`, `_pion` | `dnde_neutrino_muon`, `dnde_neutrino_charged_pion` — scalar → 3-tuple, array → `(3, N)` array |
| `scalar_mediator/_c_scalar_mediator_cross_sections` | 13 `def`s: `sigma_xx_to_s_to_{ff,gg,pi0pi0,pipi}`, `sigma_xx_to_ss`, `sigma_ss_to_xx`, `sigma_x{l,pi,pi0,g,s}_to_x{l,pi,pi0,g,s}`, `sigma_xx_to_all`(unused), `thermal_cross_section` |
| `vector_mediator/_c_vector_mediator_cross_sections` | 7 `def`s: `sigma_xx_to_v_to_{ff,pipi,pi0g,pi0v}`, `sigma_xx_to_vv`, `sigma_xx_to_all`(unused), `thermal_cross_section` |
| `scalar_mediator/scalar_mediator_decay_spectrum` | `scalar_mediator_decay_spectrum` |
| `scalar_mediator/scalar_mediator_positron_spec` | `dnde_decay_s`, `dnde_decay_s_pt` |
| `vector_mediator/vector_mediator_decay_spectrum` | `dnde_decay_v`, `dnde_decay_v_pt` |
| `vector_mediator/vector_mediator_positron_spec` | `dnde_decay_v`, `dnde_decay_v_pt` |
<!-- markdownlint-enable MD013 -->

Python-side import sites (the wrapper layer to repoint during ports, one
file each): `spectra/_photon/__init__.py:12`,
`spectra/_neutrino/__init__.py:11-12`, `spectra/_positron/__init__.py:12`,
`scalar_mediator/_scalar_mediator_cross_sections.py:1`,
`scalar_mediator/_scalar_mediator_spectra.py:7`,
`scalar_mediator/_scalar_mediator_positron_spectra.py:4`,
`vector_mediator/_vector_mediator_cross_sections.py:8-24`,
`vector_mediator/_vector_mediator_spectra.py:7`,
`vector_mediator/_vector_mediator_positron_spectra.py:4`.

### C-level dependency graph (dictates port order)

```text
_utils/constants.pxd  — textual include of 151 compile-time DEFs → all live kernels
_utils/kinematics.pxd — 3 cdef inline fns → _photon/_rho, _neutrino/_pion
_utils/boost.pxd      — boost_beta, boost_gamma (inline);
                        boost_delta_function, boost_integrate_linear_interp (linked)
                        → 10 spectra kernels

_photon/_muon ──► _photon/_pion ──► _photon/_rho          (cimport chains)
_positron/_muon ──► _positron/_pion
_neutrino/_muon ──► _neutrino/_pion   (+ both use the _neutrino struct module)

mediator decay/positron spectra ──cimport──► 8 cdef symbols:
  dnde_photon_muon_{point,array}, dnde_photon_charged_pion_{point,array},
  dnde_photon_neutral_pion_{point,array},
  dnde_positron_muon_array, dnde_positron_charged_pion_array
```

The mediator spectrum modules link against other extensions'
`__pyx_capi__` capsules — a mechanism Rust cannot join. They therefore
port **after** the spectra kernels (Phase 06 after Phase 04). The two
cross-section modules cimport nothing from hazma and can port any time
after the foundation (Phase 05 needs only `k1`/`kn` + `qags`).

`_utils/boost.pyx` (the compiled half: `boost_delta_function`,
`boost_integrate_linear_interp`) stays compiled until the last Cython
cimporter dies — that is the end of Phase 04.

### Data files read by live compiled modules

All in `hazma/spectra/_photon/data/`, loaded at module import via
`np.loadtxt(..., delimiter=",")`, path via `spectra/_photon/path.py`:

<!-- markdownlint-disable MD013 -- grounded-fact table; width is the content -->
| File | Shape | Consumer |
| --- | --- | --- |
| `charged_kaon_photon.csv` | 501×8 | `_kaon.pyx` |
| `long_kaon_photon.csv` | 501×7 | `_kaon.pyx` |
| `short_kaon_photon.csv` | 501×3 | `_kaon.pyx` |
| `eta_photon.csv` | 101×6 | `_eta.pyx` |
| `eta_prime_photon.csv` | 501×7 | `_eta_prime.pyx` |
| `omega_photon.csv` | 501×7 | `_omega.pyx` |
| `phi_photon.csv` | 501×11 | `_phi.pyx` |
<!-- markdownlint-enable MD013 -->

Pattern: transpose, column 0 = energies, dnde = sum of remaining
channel columns, `emin`/`emax` from the grid ends. The
positron/neutrino CSVs are read by pure Python and are out of scope.

## Structural facts that shrink the port

- `_eta`/`_omega`/`_eta_prime`/`_phi` are one ~70-line template (table +
  boost + delta terms); `_kaon` is the same template ×3 with the interp
  factored out. `_kaon.pyx` also carries ~120 lines of commented-out dead
  code; `_eta`-family files carry ~50 each.
- The two cross-section modules are three-tier: 19 `cdef` scalar kernels
  (Mathematica `CForm` dumps — magic rationals like `419904.0` = 2⁵·3⁸,
  90-line single expressions, one subexpression repeated verbatim 8×),
  then per-kernel `__vec_*` loops, then per-kernel `def` dispatchers.
  Tiers 2+3 (~1,200 lines) collapse into one generic dispatch helper.
- The four mediator spectrum modules are two designs cloned scalar↔vector
  (the positron pair differs by `ms`→`mv` renaming).
- Distinct logic across the whole live surface: **~2,500–3,000 lines.**
- Entry-point polymorphism is uniform: dispatch on
  `hasattr(x, "__len__")`, `float` in → `float` out, 1-D array in →
  1-D array out (neutrino: 3-tuple / `(3, N)`).

## Bugs found during the audit (fix or avoid during ports)

Live code:

1. **Dead memo-cache** — `scalar_mediator_positron_spec.pyx:21-22,49-55`
   and `vector_mediator_positron_spec.pyx:22-23,50-56`: `cache_ms`/
   `cache_pws` are read but never assigned, so `__set_spectra` (two
   500-point tables, each point a `quad`) reruns on **every call**, and
   per point in the `_pt` variants. Numbers correct; performance awful.
   Fix lands with the Phase 06 redesign (identical numbers, declared as
   performance-only).
2. **`np.log(4)`** at `_c_scalar_mediator_cross_sections.pyx:283` inside
   `cdef double __sigma_xl_to_xl` — a Python round-trip amid `libc.math`
   calls. Value is correct; port as the constant.
3. **Constants divergence** — `MASS_E` = 0.510998928 in
   `_decay/common.pxd` / `_decay/parameters.pxd` / `_positron/parameters.pxd`
   vs 0.5109989461 in `_utils/constants.pxd`; `BR_PI_TO_ENU` has three
   spellings (0.000123 / 1.2e-4 / 1.230e-4); `WIDTH_K = 3.3406**-13.`
   and `WIDTH_PI = 2.528511206475808**-14.` are `**` exponentiation
   where `e-` notation was meant (~10⁶ off; both currently unused).
   Governed by the bit-parity rule in `../rules.md`.
4. Unused cimports in mediator decay modules (safe to drop at port time):
   `dnde_photon_charged_pion_point` + `dnde_photon_neutral_pion_array`
   in the scalar module, `dnde_photon_neutral_pion_array` in the vector
   module, `exp` in both.

Dead code (do not resurrect): out-of-bounds `std::vector` write in
`gamma_ray_fsr.pyx` Python-callback path (`momenta[nfsp][0]` on an empty
inner vector); uninitialized `prefactor` read at
`gamma_ray_generator.pyx:196,234` plus `is`-vs-`==` string compares;
off-by-one index pairing in `boost_integrate_linear_interp_massive`.
