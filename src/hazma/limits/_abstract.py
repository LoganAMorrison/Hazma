import abc
import warnings

import numpy as np


class AbstractLimit(abc.ABC):
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def _constrain(self, model):
        pass

    def constrain(self, model_iterator):
        """
        Compute the constraints on the models.

        Parameters
        ----------
        model_iterator: iter
            Iterator over the dark matter models.

        Returns
        -------
        constraints: array-like
            Numpy array containing the constraints for each model.
        """
        constraints = np.zeros((len(model_iterator),), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, model in enumerate(model_iterator):
                constraints[i] = self._constrain(model)
        return constraints


class CompositeConstrainer(abc.ABC):
    def __init__(self, constrainers=[]):
        self._constrainers = constrainers

    def add_constrainer(self, constrainer):
        self._constrainers.append(constrainer)

    def reset_constrainers(self):
        self._constrainers = []

    @property
    @abc.abstractmethod
    def description(self) -> str:
        raise NotImplementedError()

    def __len__(self):
        return len(self._constrainers)

    def _constrain(self, model_iterator):
        constraints = {}
        for constrainer in self._constrainers:
            name = constrainer.name
            constraints[name] = constrainer.constrain(model_iterator)

        return constraints
