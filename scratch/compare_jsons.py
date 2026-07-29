import glob
import json

print("==========================================================================================")
print("                       OFFICIAL SIDE-BY-SIDE EVALUATION COMPARISON                         ")
print("==========================================================================================")

files = glob.glob("NEW_checkpoints/**/*_eval.json", recursive=True)
for path in sorted(files):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ckpt = data.get("checkpoint", path)
        ret = data.get("mean_reward", 0.0)
        std_ret = data.get("std_reward", 0.0)
        lin_err = data.get("mean_linear_velocity_error", 0.0)
        yaw_err = data.get("mean_yaw_rate_error", 0.0)
        steps = data.get("mean_episode_length", 0.0)
        print(f"File: {path}")
        print(f"  Checkpoint: {ckpt}")
        print(f"  Mean Return: {ret:.4f} ± {std_ret:.4f}")
        print(f"  Linear Velocity Error: {lin_err:.4f} m/s")
        print(f"  Yaw Rate Error: {yaw_err:.4f} rad/s")
        print(f"  Mean Survival Steps: {steps:.1f} / 1000.0 (Fall Rate: {100*(1 - steps/1000.0):.2f}%)")
        print("------------------------------------------------------------------------------------------")
    except Exception as e:
        print(f"Error reading {path}: {e}")
