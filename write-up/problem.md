# Problem

Train a Unitree Go1 quadruped to track command velocities in the stock MuJoCo Playground joystick task without falling. The canonical evidence uses five independent custom training seeds and three Brax baseline seeds, 50 fixed evaluation episodes per seed, a shared deterministic command-tracking evaluator, honest step and wall-time accounting, and explicit environment modifications. The complete narrative is `submission.md`.
