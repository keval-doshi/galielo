# How Galileo worked it out (muse-wonder-1.0)

Notes given: Rigid pendulum: a rod with a wooden block on the end. 33 cm from the pivot to the centre of mass (given by the lab).

**Step 1.** Rigid pendulum (rod + block) swinging on an arc — period is scale-free, so I'll fit it with the given 33 cm length, not the block's arc radius.

- `fit_model` `{"model": "pendulum", "track": "wooden block#0", "length_cm": 33.0}`
  - wooden block#0: g = 9.800 ± 0.089 m/s²  (L = 33.00 cm, T = 1.1598 s, amplitude 17.5°)

**Step 2.** Pendulum fit gives g = 9.80 m/s² with a clean arc. Now I'll test the damped-oscillation law against an undamped alternative using BIC.

- `fit_equation` `{"series": "theta", "expression": "A*exp(-gamma*t)*cos(w*t + p)", "track": "wooden block#0", "start": 1.5, "end": 4.9, "initial": {"A": 0.3, "gamma": 0.01, "w": 5.42, "p": 0.0}}`
  - rms 0.01222, BIC -888.9, R² 0.99683

- `fit_equation` `{"series": "theta", "expression": "A*cos(w*t + p)", "track": "wooden block#0", "start": 1.5, "end": 4.9, "initial": {"A": 0.3, "w": 5.42, "p": 0.0}}`
  - rms 0.01299, BIC -880.9, R² 0.99642

- `submit_report` `{"experiment": "Rigid pendulum — rod with wooden block", "headline": "Measured g = 9.80 ± 0.09 m/s² from the pendulum period and the given 33 cm length, within 0.1% of the standard 9.80665 m/s² and consistent within the uncertainty.", "law": "θ(t) = θ0·e^(−γt)·cos(2πt/T + φ), with T ≈ 2π√(L/g) corrected for finite amplitude; L is the given effective length (pivot to centre of mass).", "best_plot": "fit_pendulum_wooden-block-0.png"}`
  - report received

