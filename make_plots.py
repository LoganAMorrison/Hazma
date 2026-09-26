import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
from hazma.relic_density._thermal_functions import thermal_cross_section as tcs

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
C_OLD, C_NEW, C_REF = "#d62728", "#1f77b4", "0.25"

# 1. <sigma v>(x) at one HiggsPortal point via the generic fallback.
hp = HiggsPortal(mx=100.0, ms=300.0, gsxx=1.0, stheta=1e-1)
wrapped = NoThermalCrossSection(hp)
xs = np.geomspace(1, 300, 120)
new = np.array([tcs(x, wrapped) for x in xs])
with old_limit(): old = np.array([tcs(x, wrapped) for x in xs])
ref = np.array([hp.thermal_cross_section(x) for x in xs])
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(xs, ref, color=C_REF, lw=3, alpha=0.4, label="Rust scalar kernel (reference)")
ax.loglog(xs, np.where(old > 0, old, np.nan), color=C_OLD, ls="--", label=r"fallback, before: $z \in [2,\ 50/x]$")
ax.loglog(xs, new, color=C_NEW, label=r"fallback, after: $z \in [2,\ 2 + 50/x]$")
ax.axvline(25, color=C_OLD, lw=0.8, ls=":"); ax.text(26, ax.get_ylim()[0]*3, "x = 25: old interval closes\n(returns 0 above)", color=C_OLD, fontsize=9)
ax.axvspan(20, 30, color="0.85", alpha=0.5, zorder=0, label="freeze-out, $x \\sim$ 20–30")
ax.set_xlabel(r"$x = m_\chi / T$"); ax.set_ylabel(r"$\langle\sigma v\rangle$ [MeV$^{-2}$]")
ax.set_title(r"Generic thermal average, HiggsPortal($m_\chi$=100, $m_S$=300 MeV, $g_{S\chi}$=1, $s_\theta$=0.1)", fontsize=10)
ax.legend(fontsize=9, loc="lower left"); fig.tight_layout(); fig.savefig("sigmav_vs_x.png", dpi=130)

# 2. Omega h^2 vs mx, HiggsPortal through the generic fallback.
mxs = np.geomspace(20, 280, 45)
rows = []
for mx in mxs:
    m = HiggsPortal(mx=mx, ms=300.0, gsxx=1.0, stheta=1e-1)
    w = NoThermalCrossSection(m)
    n = relic_density(w, semi_analytic=True)
    with old_limit(): o = relic_density(w, semi_analytic=True)
    r = relic_density(m, semi_analytic=True)
    rows.append((o, n, r)); print("hp", mx, o, n, r, flush=True)
o, n, r = map(np.array, zip(*rows))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
a1.loglog(mxs, r, color=C_REF, lw=3, alpha=0.4, label="Rust scalar kernel (reference)")
a1.loglog(mxs, o, color=C_OLD, ls="--", marker=".", label="generic fallback, before")
a1.loglog(mxs, n, color=C_NEW, marker=".", label="generic fallback, after")
a1.axhline(0.12, color="0.5", lw=0.8, ls=":"); a1.text(mxs[0], 0.14, r"$\Omega h^2 = 0.12$", fontsize=9, color="0.4")
a1.set_ylabel(r"$\Omega_\chi h^2$ (semi-analytic)")
a1.set_title(r"HiggsPortal, $m_S$=300 MeV, $g_{S\chi}$=1, $s_\theta$=0.1", fontsize=10); a1.legend(fontsize=9)
a2.semilogx(mxs, np.abs(o / r - 1), color=C_OLD, ls="--", marker=".", label="before")
a2.semilogx(mxs, np.abs(n / r - 1), color=C_NEW, marker=".", label="after")
a2.set_yscale("log"); a2.set_ylabel(r"$|$fallback / kernel $- 1|$"); a2.set_xlabel(r"$m_\chi$ [MeV]"); a2.legend(fontsize=9)
fig.tight_layout(); fig.savefig("higgs_portal_omega_vs_mx.png", dpi=130)

# 3. Omega h^2 vs mx, VectorMediatorGeV.relic_density.
mxs = np.geomspace(1.1e3, 1e4, 16)
rows = []
for mx in mxs:
    g = gev(mx)
    kw = dict(semi_analytic=True, three_body=False, four_body=False)
    n = g.relic_density(**kw)
    with old_limit(): o = g.relic_density(**kw)
    rows.append((o, n)); print("gev", mx, o, n, flush=True)
o, n = map(np.array, zip(*rows))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(mxs, n, color=C_NEW, marker="o", label="after")
fin = np.isfinite(o)
if fin.any(): ax.loglog(mxs[fin], o[fin], color=C_OLD, ls="--", marker="x", label="before")
for mx in mxs[~fin]: ax.axvline(mx, color=C_OLD, alpha=0.15, lw=4)
ax.plot([], [], color=C_OLD, alpha=0.3, lw=4, label=f"before: NaN ({(~fin).sum()}/{len(mxs)} points)")
ax.set_xlabel(r"$m_\chi$ [MeV]"); ax.set_ylabel(r"$\Omega_\chi h^2$ (semi-analytic)")
ax.set_title(r"VectorMediatorGeV, $m_V$=2 GeV, $g_{V\chi}$=1, $(g_u, g_d, g_s)$=(3, 1, −1), two-body channels", fontsize=10)
ax.legend(fontsize=9); fig.tight_layout(); fig.savefig("vector_gev_omega_vs_mx.png", dpi=130)
