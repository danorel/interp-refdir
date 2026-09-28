"""Reusable core for mech-interp experiments.

Extension points (subclass and reference from a config via `module:Class`):
    models.InterpModel     - backend giving generation + internals access
    models.Generator       - generation-only backend (e.g. vLLM)
    interventions.Intervention - edit activations at Sites
    tasks.Task             - dataset -> prompts (+ optional scoring)
    judges.Judge           - completions -> labels
    experiment.Experiment  - one runnable experiment with a reproducible run dir
"""

from interptemp.config import ExperimentConfig, load_config
from interptemp.sites import Site

__all__ = ["ExperimentConfig", "Site", "load_config"]
