from abc import ABC, abstractmethod
from typing import List, Tuple
import dataclasses

class OracleInterface(ABC):
    @abstractmethod
    def prepare_schemes(self, oracle: "Oracle"):
        pass

    @abstractmethod
    def query(self, w: "Workload|List[Workload]", timer: "Timer" = None) -> "List[float, OptimizationResult|int]":
        pass

    def query_directed_drift(self, w: "Workload", drift: "Workload", timer: "Timer" = None) -> "Tuple[float, List[float], int]":
        raise NotImplementedError()

    @classmethod
    def get_default_args(cls):
        """
        Returns the default arguments of the implementation.
        For customizing the oracle arguments, these defaults can be modified and passed to create_oracle.
        """

        # Verify that the class is a dataclass
        assert dataclasses.is_dataclass(cls), f"Class {cls} is not a dataclass"

        defaults = dict()
        for field in dataclasses.fields(cls):
            # If there is a default and the value of the field is none we can assign a value
            if field.init:
                if not isinstance(field.default, dataclasses._MISSING_TYPE):
                    defaults[field.name] = field.default
                elif not isinstance(field.default_factory, dataclasses._MISSING_TYPE):
                    defaults[field.name] = field.default_factory()

        return defaults