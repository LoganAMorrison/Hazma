from hazma import gamma_ray_parameters as grp

from ._abstract import AbstractLimit


class ExistingTelescopeLimit(AbstractLimit):
    """
    Class for computing the limits on the dark-matter models from an
    existing telescope.
    """

    def __init__(self, measurement, sigma=2.0, method="chi2"):
        super().__init__()
        self.measurement = measurement
        self.sigma = sigma
        self.method = method

    def _constrain(self, model):
        assert hasattr(model, "binned_limit"), (
            "Model does not have 'binned_limit'. "
            + "Make sure the model inherets from 'Theory'."
        )
        return model.binned_limit(self.measurement, self.sigma, self.method)


class ComptelLimit(ExistingTelescopeLimit):
    def __init__(self, sigma=2.0, method="chi2"):
        super().__init__(grp.comptel_diffuse, sigma, method)

    @property
    def description(self):
        return "[deep_sky_blue2] COMPTEL"

    @property
    def name(self):
        return "comptel"


class EgretLimit(ExistingTelescopeLimit):
    def __init__(self, sigma=2.0, method="chi2"):
        super().__init__(grp.egret_diffuse, sigma, method)

    @property
    def description(self):
        return "[deep_sky_blue1] EGRET"

    @property
    def name(self):
        return "egret"


class FermiLimit(ExistingTelescopeLimit):
    def __init__(self, sigma=2.0, method="chi2"):
        super().__init__(grp.fermi_diffuse, sigma, method)

    @property
    def description(self):
        return "[light_sea_green] Fermi"

    @property
    def name(self):
        return "fermi"


class IntegralLimit(ExistingTelescopeLimit):
    def __init__(self, sigma=2.0, method="chi2"):
        super().__init__(grp.integral_diffuse, sigma, method)

    @property
    def description(self):
        return "[dark_cyan] INTEGRAL"

    @property
    def name(self):
        return "integral"
