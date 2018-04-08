
# #############################################################################
# ################## HAZMA ERRORS #############################################
# #############################################################################


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


# #############################################################################
# ################## HAZMA WARNINGS ###########################################
# #############################################################################


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
        self.default_message = "Negative squared matrix element" + \
            "encountered. Using zero instead."

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
