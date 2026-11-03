# Overengineered / Galileo

Film something moving. **SAM 3.1** tracks it, **Muse Wonder** works out the physics.

```
app.py             the Galileo page: upload a video, see tracking, the law and the measured numbers
galileo.py         the same agent from the command line, on a run tracked with track.py
track.py           video -> SAM 3.1 masks -> tracks.csv, plot.png, overlay.mp4 (cm and seconds)
fit.py             one physics model by hand (pendulum, freefall, projectile, ramp, dominoes...)
check_api.py       is my key working, are the models there?
make_test_clip.py  synthetic videos with a known answer (to test without filming)
oe/                the library: API calls, masks, scale, motion analysis, physics, the agent
```

Commands below are for Windows (PowerShell or Command Prompt) inside the activated `.venv`.

---

# Tuesday: Galileo

## Step 1 — Update (5 min)

Extract the new zip over your old folder and choose **Replace the files in the destination**.
Your `.env` (API key) and `runs\` folder are kept, because they are not in the zip. Then:

```
.venv\Scripts\activate
pip install -r requirements.txt
```

That adds Gradio (the web page). Your old runs still work with every tool.

## Step 2 — Galileo on your self-test run (no new SAM cost)

```
python galileo.py runs\selftest_sam --note "string 55.0 cm from the knot to the bob centre"
```

You will see Muse Wonder think out loud, call fitting tools, compare candidate laws and finish
with a report: `runs\selftest_sam\report.md`, plus `agent_transcript.md` (how it got there) and
its fit plots. Expect g ≈ 9.81 m/s². The last line shows roughly what the run cost.

If Wonder is slow, add `--effort medium`, or use `--model muse-spark-1.3` (cheaper and faster).

## Step 3 — The app

```
python app.py
```

Your browser opens http://127.0.0.1:7860.

- **Re-analyse:** pick `selftest_sam` under *Previous run* → **Analyse this run**.
- **New video:** drop your clip in, type what moves ("orange ball"), and type anything you
  measured in *What you measured by hand* (string length; ramp height and length). Then press
  **Track and analyse**. The log shows every step. The tabs show the SAM 3.1 tracking video,
  the paths, the scale check and how Galileo worked it out.

Each new video costs one SAM 3.1 tracking call per prompt plus one Galileo run. Re-analysing a
previous run only costs the Galileo run.

## Step 4 — Your real clips

Film them as in Monday's Step 3 below, then either use the app, or:

```
python track.py pendulum.mov --prompt "orange ball" --out runs\pendulum1
python galileo.py runs\pendulum1 --note "string 61.5 cm from the knot to the bob centre"

python track.py ramp.mov --prompt "marble" --out runs\ramp1
python galileo.py runs\ramp1 --note "ruler ramp 30 cm long, raised 8.2 cm at one end; glass marble"

python track.py dominoes.mov --prompt "domino" --out runs\dominoes1
python galileo.py runs\dominoes1 --note "dominoes 4.8 cm tall, about 3 cm apart"
```

What Galileo can fit: pendulum (g, damping), springs and other oscillators (period, damping,
spring constant when you give the mass), free fall and projectile (g), ramp (acceleration,
rolling resistance or friction, and what kind of ball it is), constant velocity, domino wave
speed, plus any equation it writes itself (compared by BIC).

**Good result:** g within about 2–3% of 9.81 on a real pendulum. With a tape-measured string
length, the pendulum is usually better than 1%, because the period comes from timing alone.

## Any video: where the scale comes from

Times and angles never need a scale; lengths, speeds and accelerations do. In the app, pick
**Where the real-world scale comes from**:

| Choice | When | Command-line equivalent |
|---|---|---|
| A4 / US Letter sheet, metre stick, 30 cm ruler | it is fully visible in the plane of the motion | `--ref-prompt "meter stick" --ref-long-cm 100 --ref-short-cm 0` |
| Other object of known size | e.g. a scale bar, a credit card (8.56 cm), a door | `--ref-prompt "credit card" --ref-long-cm 8.56 --ref-short-cm 5.4` |
| No reference | you know a length instead: write it in the notes | `--no-scale` |
| I know the scale | you measured pixels per cm on the original video | `--px-per-cm 22.3` |

With **no reference**, everything is tracked in pixels and Galileo calibrates from your notes:
a pendulum with a known length needs no scale at all, and for anything else it measures the
known length (drop height, ramp length, distance between marks) in pixels and calls
`set_scale`. The report says how it calibrated.

Example, a real lab video (University of Toronto, rigid pendulum, 33 cm pivot to centre of mass):

```
python track.py shmvideo5pendulum.mp4 --prompt "wooden block" --no-scale --out runs\toronto5
python galileo.py runs\toronto5 --note "Rigid pendulum: a rod with a wooden block. 33 cm from the pivot to the centre of mass (given by the lab)."
```

Expect g ≈ 9.8 m/s² from a period of about 1.16 s.

## Step 5 — Safety net before 7 PM

1. **Push the code to GitHub.** `.gitignore` already keeps `.env`, videos and `runs\` out.
   ```
   git init
   git add .
   git commit -m "Galileo: SAM 3.1 tracking + Muse Wonder physics agent"
   ```
   Then create an empty repo on github.com and follow its "push an existing repository" lines.
2. **Record a rough 1-minute backup demo.** Screen-record the app (Win + Alt + R with the Xbox
   Game Bar, or the Snipping Tool's video mode). Upload the pendulum clip, show the tracking
   tab, then the result. With this, you have a complete submission whatever happens tomorrow.

## Done for today when

- [ ] Galileo produced a report for `selftest_sam` and for your real pendulum
- [ ] the app works end to end on a real clip
- [ ] code pushed, backup demo recorded

Next (Tuesday evening): the digital twin. A side photo of your desk becomes a 2D physics scene
(SAM 3.1 masks → shapes in a simulator), the first step of the Rube Goldberg machine.

---

# Monday: video in, physics out

## Set up (15 min)

```
cd overengineered
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

(macOS/Linux: `python3 -m venv .venv`, `source .venv/bin/activate`, `cp .env.example .env`.)

Get your key: **dev.meta.ai → Model API dashboard → API keys → create**. Paste it into `.env`
as `MODEL_API_KEY=...` (it looks like `LLM|123...|...`). Then `python check_api.py`: every line
should say `[ok]`. If a model is missing, check the list it prints and set the right ID in `.env`
(e.g. `SPARK_MODEL=...`).

## Self-test (10 min)

```
python make_test_clip.py pendulum
python track.py test_pendulum.mp4 --prompt "red ball" --ref-prompt "blue rectangle" --out runs\selftest_sam
python fit.py runs\selftest_sam --model pendulum
```

Expect g ≈ 9.82 ± 0.10 m/s² (true 9.807). Offline, without the API:
`python track.py test_pendulum.mp4 --backend color --prompt bob --color "#d32f2f" --ref-color "#3d85c6" --out runs\selftest`.
`make_test_clip.py` can also make `ramp`, `drop` and `projectile` clips.

## Filming

**Camera rules (these matter more than anything else):**
- Phone on a stand or propped still, **landscape**, never handheld. Don't zoom or move it mid-clip.
- Lens at the height of the motion, pointing **straight at it**, 1.5–2 m away, 1× lens (not ultra-wide).
- **Normal video at 60 fps** if your phone allows it, **not slow-motion** (slow-motion files
  confuse the timing). Bright light, plain background that contrasts with the object.
- Keep each clip under 20 s.

**The calibration moment (gives the cm scale):** at the start of each clip, hold a flat **A4 sheet
in the same plane as the motion** (e.g. touching the pendulum string), facing the camera, fully
visible, for about a second. Then move it away and start. A sheet even 20 cm behind the motion
throws the scale off by ~10%. (US Letter: add `--ref-long-cm 27.94 --ref-short-cm 21.59`, or pick
it in the app.)

1. **Pendulum** — a string 50–80 cm long tied to a shelf or door frame, with a small heavy,
   brightly coloured bob. Pull it about 15° to the side, let go, and record 10+ swings. Measure
   the string from the knot to the centre of the bob with a tape.
2. **Ramp** — a ruler or stiff cardboard on a stack of books, a marble or small ball rolling down.
   Measure the height and length of the ramp.
3. **Dominoes** — 5–10 dominoes, cards or thin books standing in a line, knocked over.

## Tracking tips

`--start 1.5 --end 14` trims (the calibration moment must still be inside, or use `--ref-time`).
`--prompt` can be given more than once. `--fresh` calls SAM again instead of reusing the cached
answer. `--px-per-cm 9.8` skips the sheet. Prompts are short noun phrases, **one concept per
prompt**; every instance found becomes its own track (`domino#0`, `domino#1`, ...). API limits:
up to 16 tracked objects per frame and 15,000 frames per video.

Check, in this order: `ref_check.png` (is the outlined shape exactly the sheet; do the two sides
agree within ~3%?), `overlay.mp4` (does the mask stay on the object?), and the `tracks:` lines
(coverage 90%+).

| Problem | Fix |
|---|---|
| "could not find sheet of paper" | Sheet fully visible at `--ref-time`? Try `--ref-prompt "white paper"`, or `--px-per-cm` |
| Sides disagree by >4% | Sheet tilted or bent; hold it flat, facing the camera |
| Coverage low / object lost | More specific prompt ("small orange ball"), more light, plainer background, less blur |
| Wrong object tracked | More specific prompt; or choose the track with `--track "ball#1"` in fit.py |
| Many extra tracks | Normal for "domino"; for a single object, use a more specific phrase |

## Fitting by hand

`fit.py` picks the time window by itself (from just after release to the end of that motion phase):

```
python fit.py runs\pendulum1 --model pendulum --length-cm 61.5
python fit.py runs\ramp1 --model incline-roll --angle-deg 15.8
python fit.py runs\drop1 --model freefall
python fit.py runs\dominoes1 --model domino-wave --track domino
```

Error bars include a 1% cm-scale uncertainty (`--scale-err-pct`). Calibrated numbers are saved
to `physics_params.json` for the simulator.