# Problem

Train a Unitree Go1 quadruped to track command velocities in the stock MuJoCo Playground joystick task without falling. Success requires three independent training seeds, 50 fixed evaluation episodes per seed, native reward, a matched Stable-Baselines3 baseline, and honest compute accounting. The complete narrative lives in `submission_template.md` and final numbers are inserted only by `eval/build_submission.py`.
