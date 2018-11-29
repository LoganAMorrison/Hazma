from hazma.vector_mediator import VectorMediator, KineticMixing
from hazma.parameters import electron_mass as me
import numpy as np


mx = 250.
gvxx = 1.
eps = 0.1

mv1, mv2 = 125., 550.

gvuu = 1.
gvdd1, gvdd2 = 1., -1.
gvss, gvee, gvmumu = 0., 0., 0.

vrel = 1e-3
cme = 2. * mx * (1. + 0.5 * vrel**2)
egams = np.logspace(0., np.log10(cme), num=10)
eng_ps = np.logspace(me, np.log10(cme), num=10)

params1 = {'mx': mx, 'mv': mv1, 'gvxx': gvxx, 'eps': eps}
params2 = {'mx': mx, 'mv': mv2, 'gvxx': gvxx, 'eps': eps}
params3 = {'mx': mx, 'mv': mv1, 'gvxx': gvxx, 'gvuu': gvuu, 'gvdd': gvdd1,
           'gvss': gvss, 'gvee': gvee, 'gvmumu': gvmumu}
params4 = {'mx': mx, 'mv': mv2, 'gvxx': gvxx, 'gvuu': gvuu, 'gvdd': gvdd1,
           'gvss': gvss, 'gvee': gvee, 'gvmumu': gvmumu}
params5 = {'mx': mx, 'mv': mv1, 'gvxx': gvxx, 'gvuu': gvuu, 'gvdd': gvdd2,
           'gvss': gvss, 'gvee': gvee, 'gvmumu': gvmumu}
params6 = {'mx': mx, 'mv': mv2, 'gvxx': gvxx, 'gvuu': gvuu, 'gvdd': gvdd2,
           'gvss': gvss, 'gvee': gvee, 'gvmumu': gvmumu}

vm1 = KineticMixing(**params1)
vm2 = KineticMixing(**params2)
vm3 = VectorMediator(**params3)
vm4 = VectorMediator(**params4)
vm5 = VectorMediator(**params5)
vm6 = VectorMediator(**params6)

# Save the shared data
np.save('shared_data/cme.npy', cme)
np.save('shared_data/spectra_egams.npy', egams)
np.save('shared_data/eng_ps.npy', eng_ps)

# Save the parameters
np.save('vm1_data/params.npy', params1)
np.save('vm2_data/params.npy', params2)
np.save('vm3_data/params.npy', params3)
np.save('vm4_data/params.npy', params4)
np.save('vm5_data/params.npy', params5)
np.save('vm6_data/params.npy', params6)

# Save the annihilation cross sections
np.save('vm1_data/ann_cross_sections.npy',
        vm1.annihilation_cross_sections(cme))
np.save('vm2_data/ann_cross_sections.npy',
        vm2.annihilation_cross_sections(cme))
np.save('vm3_data/ann_cross_sections.npy',
        vm3.annihilation_cross_sections(cme))
np.save('vm4_data/ann_cross_sections.npy',
        vm4.annihilation_cross_sections(cme))
np.save('vm5_data/ann_cross_sections.npy',
        vm5.annihilation_cross_sections(cme))
np.save('vm6_data/ann_cross_sections.npy',
        vm6.annihilation_cross_sections(cme))

# Save the annihilation branching fractions
np.save('vm1_data/ann_branching_fractions.npy',
        vm1.annihilation_branching_fractions(cme))
np.save('vm2_data/ann_branching_fractions.npy',
        vm2.annihilation_branching_fractions(cme))
np.save('vm3_data/ann_branching_fractions.npy',
        vm3.annihilation_branching_fractions(cme))
np.save('vm3_data/ann_branching_fractions.npy',
        vm3.annihilation_branching_fractions(cme))
np.save('vm4_data/ann_branching_fractions.npy',
        vm4.annihilation_branching_fractions(cme))
np.save('vm5_data/ann_branching_fractions.npy',
        vm5.annihilation_branching_fractions(cme))
np.save('vm6_data/ann_branching_fractions.npy',
        vm6.annihilation_branching_fractions(cme))

# Save the spectra
np.save('vm1_data/spectra.npy', vm1.spectra(egams, cme))
np.save('vm2_data/spectra.npy', vm2.spectra(egams, cme))
np.save('vm3_data/spectra.npy', vm3.spectra(egams, cme))
np.save('vm4_data/spectra.npy', vm4.spectra(egams, cme))
np.save('vm5_data/spectra.npy', vm5.spectra(egams, cme))
np.save('vm6_data/spectra.npy', vm6.spectra(egams, cme))

# Save the partial widths
np.save('vm1_data/partial_widths.npy', vm1.partial_widths())
np.save('vm2_data/partial_widths.npy', vm2.partial_widths())
np.save('vm3_data/partial_widths.npy', vm3.partial_widths())
np.save('vm4_data/partial_widths.npy', vm4.partial_widths())
np.save('vm5_data/partial_widths.npy', vm5.partial_widths())
np.save('vm6_data/partial_widths.npy', vm6.partial_widths())

# Save the positron spectra
np.save('vm1_data/positron_spectra.npy', vm1.positron_spectra(eng_ps, cme))
np.save('vm2_data/positron_spectra.npy', vm2.positron_spectra(eng_ps, cme))
np.save('vm3_data/positron_spectra.npy', vm3.positron_spectra(eng_ps, cme))
np.save('vm4_data/positron_spectra.npy', vm4.positron_spectra(eng_ps, cme))
np.save('vm5_data/positron_spectra.npy', vm5.positron_spectra(eng_ps, cme))
np.save('vm6_data/positron_spectra.npy', vm6.positron_spectra(eng_ps, cme))

# Save the positron lines
np.save('vm1_data/ps_lines.npy', vm1.positron_lines(cme))
np.save('vm2_data/ps_lines.npy', vm2.positron_lines(cme))
np.save('vm3_data/ps_lines.npy', vm3.positron_lines(cme))
np.save('vm4_data/ps_lines.npy', vm4.positron_lines(cme))
np.save('vm5_data/ps_lines.npy', vm5.positron_lines(cme))
np.save('vm6_data/ps_lines.npy', vm6.positron_lines(cme))
