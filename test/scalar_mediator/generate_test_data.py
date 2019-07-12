from hazma.scalar_mediator import ScalarMediator
from hazma.parameters import vh
from hazma.parameters import electron_mass as me
import numpy as np


mx1, mx2 = 250.0, 250.0
ms1, ms2 = 550.0, 200.0
gsxx = 1.0
stheta = 1e-3
gsff = stheta
gsGG = 3.0 * stheta
gsFF = -5.0 * stheta / 6.0
lam = vh

vrel = 1e-3
cme1 = 2.0 * mx1 * (1.0 + 0.5 * vrel ** 2)
cme2 = 2.0 * mx2 * (1.0 + 0.5 * vrel ** 2)

params1 = {
    "mx": mx1,
    "ms": ms1,
    "gsxx": gsxx,
    "gsff": gsff,
    "gsGG": gsGG,
    "gsFF": gsFF,
    "lam": lam,
}
params2 = {
    "mx": mx2,
    "ms": ms2,
    "gsxx": gsxx,
    "gsff": gsff,
    "gsGG": gsGG,
    "gsFF": gsFF,
    "lam": lam,
}

SM1 = ScalarMediator(**params1)
SM2 = ScalarMediator(**params2)


np.save("sm1_data/params.npy", params1)
np.save("sm2_data/params.npy", params2)
np.save("sm1_data/cme.npy", cme1)
np.save("sm2_data/cme.npy", cme2)


np.save("sm1_data/ann_cross_sections.npy", SM1.annihilation_cross_sections(cme1))
np.save("sm2_data/ann_cross_sections.npy", SM2.annihilation_cross_sections(cme2))


np.save(
    "sm1_data/ann_branching_fractions.npy", SM1.annihilation_branching_fractions(cme1)
)
np.save(
    "sm2_data/ann_branching_fractions.npy", SM2.annihilation_branching_fractions(cme2)
)


np.save("sm1_data/vs.npy", SM1.compute_vs())
np.save("sm2_data/vs.npy", SM2.compute_vs())


egams1 = np.logspace(0.0, np.log10(cme1), num=10)
egams2 = np.logspace(0.0, np.log10(cme2), num=10)

np.save("sm1_data/spectra.npy", SM1.spectra(egams1, cme1))
np.save("sm2_data/spectra.npy", SM2.spectra(egams2, cme2))
np.save("sm1_data/spectra_egams.npy", egams1)
np.save("sm2_data/spectra_egams.npy", egams2)


np.save("sm1_data/partial_widths.npy", SM1.partial_widths())
np.save("sm2_data/partial_widths.npy", SM2.partial_widths())


eng_ps1 = np.logspace(me, np.log10(cme1), num=10)
eng_ps2 = np.logspace(me, np.log10(cme2), num=10)

np.save("sm1_data/positron_spectra.npy", SM1.positron_spectra(eng_ps1, cme1))
np.save("sm2_data/positron_spectra.npy", SM2.positron_spectra(eng_ps2, cme2))
np.save("sm1_data/eng_ps.npy", eng_ps1)
np.save("sm2_data/eng_ps.npy", eng_ps2)


np.save("sm1_data/ps_lines.npy", SM1.positron_lines(cme1))
np.save("sm2_data/ps_lines.npy", SM2.positron_lines(cme2))
