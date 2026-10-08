# the GeV vector mediator's multi-body channels use the wrong kernels

- **Added:** 2026-10-07
- **Source:** numerics review of PR #123
- **Scope:** cross-cutting (public spectrum values)
- **Status:** open

## Why

`dnde_positron_*` and `dnde_neutrino_*` in
`hazma/vector_mediator/_gev/` build each multi-body channel from a list
of decay kernels, one per final-state particle, in the order of the form
factor's `fsp_masses`. Three entries of that table do not match the
particles they decay.

1. **`pi k k0` gives the pion a kaon kernel.** The form factor's
   `fsp_masses` is `(MK0, MK, MPI)`
   (`hazma/form_factors/vector/_pi_k_k.py:672`), but the kernel list is
   `[zero, charged_kaon, charged_kaon]` at
   `hazma/vector_mediator/_gev/positron.py:743-748` and at
   `neutrino.py:837-840`. The neutral kaon gets no leptons and the
   charged pion gets a charged kaon's.
2. **`pi pi pi pi` and `pi pi pi0 pi0` give charged pions the kaon
   kernel.** All four final-state particles of `pi pi pi pi` are charged
   pions (`_pi_pi_pi_pi.py:161-163`), yet `positron.py:793-798` and
   `neutrino.py:889-893` list `[charged_kaon, charged_kaon,
   charged_pion, charged_pion]`. For `pi pi pi0 pi0`, `positron.py:838-843`
   and `neutrino.py:937-941` list `[charged_kaon, charged_kaon, zero,
   zero]`, so the two charged pions decay as kaons.
3. **The `phi` neutrino kernel is far below the `phi` positron kernel.**
   At a `phi` energy of 2000 MeV, `spectra.dnde_positron_phi`
   integrates to 2.412 e± per `phi`, while `spectra.dnde_neutrino_phi`
   integrates to 0.305 `ν_e` and 0.608 `ν_μ` (and zero `ν_τ`). A charged
   kaon, which `phi` decays to about half the time, gives 1.111 and 1.486
   on the same grid, so the neutrino kernel is missing roughly a factor
   of two to three.

## What

Replace each list with the kernel of the particle at that index, using
the short-lived and long-lived kaon kernels for neutral kaons as
`pi0 k0 k0` already does, and compare the `phi` neutrino kernel with the
branching fractions of `phi` to `K⁺K⁻`, `K_L K_S` and `3π`. Each repaired
channel needs a test that pins its integral against the sum of its
particles' kernels, in the style of
`test/vector_mediator/test_gev_neutrino_channels.py`. These move
published spectra, so the fix needs a `CHANGELOG.md` entry with
before-and-after counts.

## Entry points

- `hazma/vector_mediator/_gev/positron.py:743-748`, `793-798`, `838-843`
- `hazma/vector_mediator/_gev/neutrino.py:837-840`, `889-893`, `937-941`
- `hazma/form_factors/vector/_pi_k_k.py:672`
- `hazma/form_factors/vector/_pi_pi_pi_pi.py:161-163`
- `hazma/spectra/_neutrino/__init__.py` (`dnde_neutrino_phi`)

## Risks / open questions

- The `phi` neutrino shortfall may come from the kernel's tabulated data
  rather than a wiring error, which would put the repair in
  `hazma/spectra`, not in the GeV model.
