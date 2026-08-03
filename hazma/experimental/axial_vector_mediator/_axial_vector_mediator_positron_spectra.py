import numpy as np


class VectorMediatorPositronSpectra:
    def positron_spectra(self, eng_ps, cme):
        """Computes total continuum positron spectrum."""

        total = np.zeros(len(eng_ps))

        return {"total": total}

    def positron_lines(self, cme):
        return {}
