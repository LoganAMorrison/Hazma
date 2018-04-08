class HazmaError(Exception):
    """Base class for exceptions in Hazma."""
    pass


class HazmaWarning(UserWarning):
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


class NegativeSquaredMatrixElementWarning(HazmaWarning):
    """
    Warning raised when a negative squared matrix element
    is encountered.
    """

    def __init__(self, message=None):
        super(NegativeSquaredMatrixElementWarning, self).__init__(message)
        self.default_message = "Negative squared matrix element" + \
            "encountered. Using zero instead."

    def __str__(self):
        return self.default_message
