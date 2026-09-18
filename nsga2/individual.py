"""Representation of one candidate solution."""

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class Individual:
    """A candidate solution and the metadata used by NSGA-II.

    All objectives are assumed to be minimised.
    """

    variables: List[float]
    objectives: List[float] = field(default_factory=list)
    rank: int = 0
    crowding_distance: float = 0.0

    def __post_init__(self) -> None:
        self.variables = [float(value) for value in self.variables]
        self.objectives = [float(value) for value in self.objectives]

    @classmethod
    def from_variables(cls, variables: Iterable[float]) -> "Individual":
        return cls(list(variables))

    def __repr__(self) -> str:
        return (
            f"Individual(variables={self.variables}, objectives={self.objectives}, "
            f"rank={self.rank}, crowding_distance={self.crowding_distance})"
        )
