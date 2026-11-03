# How Galileo worked it out (muse-wonder-1.0)

Notes given: string 55.0 cm from the knot to the bob centre

**Step 1.** This is a simple pendulum — a red bob on a ~55 cm string swinging on a circular arc. I'll fit the pendulum model first.

- `fit_model` `{"model": "pendulum", "track": "red ball#0", "length_cm": 55.0}`
  - red ball#0: g = 9.805 ± 0.053 m/s²  (L = 55.00 cm, T = 1.4928 s, amplitude 12.9°)

**Step 2.** Pendulum fit gives g≈9.81 m/s² close to standard. Now testing an undamped alternative against the damped law to check by BIC.

- `fit_equation` `{"series": "theta", "expression": "A*exp(-k*t)*cos(w*t + p)", "track": "red ball#0", "start": 0.7167, "end": 9.9833, "initial": {"A": 0.226, "k": 0.03, "w": 4.21, "p": 0.0}}`
  - rms 0.0005298, BIC -8377.7, R² 0.99999

- `fit_equation` `{"series": "theta", "expression": "A*cos(w*t + p)", "track": "red ball#0", "start": 0.7167, "end": 9.9833, "initial": {"A": 0.226, "w": 4.21, "p": 0.0}}`
  - rms 0.01241, BIC -4870.4, R² 0.99384

- `submit_report` `{"experiment": "Simple pendulum", "headline": "Measured g = 9.805 ± 0.053 m/s², just 0.01% below the reference 9.80665 m/s².", "law": "θ(t) = θ0·e^(−γt)·cos(2πt/T + φ), with T ≈ 2π√(L/g) corrected for finite amplitude, so g = 4π²L/T_small²", "best_plot": "fit_pendulum_red-ball-0.png"}`
  - report received

