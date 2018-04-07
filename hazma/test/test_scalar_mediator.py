from hazma.scalar_mediator import ScalarMediator


def list_final_states(mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.list_final_states()


def cross_sections(cme, mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.cross_sections(cme)


def branching_fractions(cme, mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.branching_fractions(cme)


def spectra(eng_gams, cme, mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.spectra(eng_gams, cme)


def spectrum_functions(mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.spectrum_functions()


def partial_widths(mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.partial_widths()
