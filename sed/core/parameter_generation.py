from abc import ABC
from abc import abstractmethod

from .preprocessing import AddJitter
from .preprocessing import PreProcessingStep
from .workflow import __version__


class ParameterGenerator(ABC):
    """Class template to generate parameters for a preprocessing step

    Args:
        ABC: _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _description_
    """

    @abstractmethod
    def generate_parameters(self) -> dict:
        """Core method to be defined, which generates and returns the parameters

        Returns:
            _description_
        """
        pass

    def get_preprocessing_step(self):
        """Return an instance of the associated preprocessing step with the
        defined parameters
        """
        assert isinstance(self.PREPROCESSING_STEP, PreProcessingStep)
        return self.PREPROCESSING_STEP(**self.parameters)

    def metadata(self) -> dict:
        metadict = {"version": __version__}
        return metadict


class CalibrateJitter(ParameterGenerator):

    PREPROCESSING_STEP = AddJitter

    def __init__(self, col: str):
        pass

    def generate_parameters(self) -> dict:
        pass
