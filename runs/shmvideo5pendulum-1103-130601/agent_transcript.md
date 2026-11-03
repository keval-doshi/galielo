# How Galileo worked it out (muse-wonder-1.0)

Notes given: Rigid pendulum: a rod with a wooden block on the end. 33 cm from the pivot to the centre of mass (given by the lab)

**Step 1.** This is a rigid pendulum — a rod with a block swinging on an arc — so I'll measure g from its period using the known 33 cm pivot-to-centre-of-mass length, which needs no pixel scale.

- `fit_model` `{"model": "pendulum", "track": "wooden block#0", "length_cm": 33.0}`
  - wooden block#0: g = 9.803 ± 0.089 m/s²  (L = 33.00 cm, T = 1.1596 s, amplitude 17.5°)

**Step 2.** The pendulum fit is clean (T = 1.1596 s, g ≈ 9.80 m/s²). Now I'll test alternative oscillator laws on the angle to check the damped-pendulum model by BIC.

- `fit_equation` `{"series": "theta", "track": "wooden block#0", "expression": "A*exp(-gamma*t)*cos(w*t + p)", "initial": {"A": 0.306, "gamma": 0.006, "w": 5.418, "p": 0.0}, "start": 1.6333, "end": 4.9}`
  - rms 0.01225, BIC -853.2, R² 0.99693

- `fit_equation` `{"series": "theta", "track": "wooden block#0", "expression": "A*cos(w*t + p)", "initial": {"A": 0.306, "w": 5.418, "p": 0.0}, "start": 1.6333, "end": 4.9}`
  - rms 0.01303, BIC -845.6, R² 0.99653

- `submit_report` `{"experiment": "Rigid pendulum — rod with a wooden block", "headline": "Using the lab length of 33 cm and a measured period of 1.1596 s, the pendulum gives g = 9.80 ± 0.09 m/s², in agreement with the standard 9.80665 m/s².", "law": "θ(t) = θ0·e^(−γt)·cos(2πt/T + φ), with T = 2π√(L/g) including a finite-amplitude correction; here L = 33 cm and amplitude θ0 ≈ 17.5°", "best_plot": "fit_pendulum_wooden-block-0.png"}`
  - report received

