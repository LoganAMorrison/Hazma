# Hazma errors


class HazmaError(Exception):
    """Base class for exceptions in Hazma."""

    pass


class NegativeSquaredMatrixElementError(HazmaError):
    """
    Exception raised when a negative squared matrix element
    is encountered.
    """

    def __init__(self, message=None):
        super(NegativeSquaredMatrixElementError, self).__init__(message)
        self.default_message = "Negative squared matrix element encountered."

    def __str__(self):
        return self.default_message


class RamboCMETooSmall(HazmaError):
    """
    Exception raised when RAMBO is called with a CME less than the sum of final
    state masses.
    """

    def __init__(self, message=None):
        super(RamboCMETooSmall, self).__init__(message)
        self.default_message = (
            "Center of mass energy is less than sum of " + "the final state masses"
        )

    def __str__(self):
        return self.default_message


# Hazma warnings


class HazmaWarning(UserWarning):
    """Base class for exceptions in Hazma."""

    pass


class NegativeSquaredMatrixElementWarning(HazmaWarning):
    """
    Warning raised when a negative squared matrix element
    is encountered.
    """

    def __init__(self, message=None):
        super(NegativeSquaredMatrixElementWarning, self).__init__(message)
        self.default_message = (
            "Negative squared matrix element" + "encountered. Using zero instead."
        )

    def __str__(self):
        return self.default_message


class PreAlphaWarning(HazmaWarning):
    """
    Warning raised when user tries to access feature which is pre-alpha
    """

    def __init__(self, message=None):
        super(PreAlphaWarning, self).__init__(message)
        self.default_message = "Accessing pre-alpha feature."

    def __str__(self):
        return self.default_message
