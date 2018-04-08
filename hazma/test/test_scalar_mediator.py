from hazma.scalar_mediator import ScalarMediator


def test_list_final_states(mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.list_final_states()


def test_cross_sections(cme, mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.cross_sections(cme)


def test_branching_fractions(cme, mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.branching_fractions(cme)


def test_spectra(eng_gams, cme, mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.spectra(eng_gams, cme)


def test_spectrum_functions(mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.spectrum_functions()


def test_partial_widths(mx, ms, gsxx, gsff, gsGG, gsFF):
    SM = ScalarMediator(mx, ms, gsxx, gsff, gsGG, gsFF)
    SM.partial_widths()
