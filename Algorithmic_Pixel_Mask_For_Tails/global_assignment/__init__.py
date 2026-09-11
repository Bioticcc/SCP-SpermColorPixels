"""Bounded, deterministic joint selection of tail-path hypotheses."""

from .solver import Hypothesis, solve_assignment, solve_with_margins

__all__ = ["Hypothesis", "solve_assignment", "solve_with_margins"]
