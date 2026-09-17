# Notebooks

Naming convention: `NN_phaseName.ipynb`, one per phase in docs/project_plan.md section (d):

- `00_phase0_kill_gate.ipynb`
- `01_level1_atlas.ipynb`
- `02_level2_predict_form.ipynb`
- `03_level3_residual_hypothesis.ipynb`
- `04_level4_cross_validation.ipynb`

Keep these thin — real logic lives in `src/fruresidual/` so it's testable
and runs identically on the laptop, Colab, Kaggle, or the lab A5000
machine. A notebook should mostly be: load config → call package
functions → plot/inspect.
