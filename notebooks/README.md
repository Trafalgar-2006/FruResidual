# Notebooks

**There are currently no notebooks here, by design.** The original plan
was one notebook per phase, but every phase ended up needing to be
re-run repeatedly and regression-tested, so the per-phase entry points
live in `scripts/run_phaseN.py` instead — they are plain Python, diffable
in git, runnable headless, and covered by `tests/test_phaseN_integration.py`.

See the Status section of the top-level [README.md](../README.md) for
which script reproduces which phase.

This directory is kept as scratch space for ad-hoc exploration. If you
add a notebook, keep it thin — real logic belongs in `src/fruresidual/`
so it stays testable and runs identically on the laptop, Colab, Kaggle,
or the lab A5000 machine. A notebook should mostly be: load config →
call package functions → plot/inspect.
