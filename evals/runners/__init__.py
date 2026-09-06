"""Runners: one per way of reaching a candidate, all on the same spine."""
from evals.runners.base import BaseRunner, RunnerError
from evals.runners.process import ProcessRunner
from evals.runners.speech import SpeechRunner
from evals.runners.text import CompletionRunner

__all__ = ["BaseRunner", "RunnerError", "ProcessRunner", "SpeechRunner",
           "CompletionRunner"]
