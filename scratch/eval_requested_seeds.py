import glob
import json
import os
import subprocess
import numpy as np

TARGET_SEEDS = [8009, 9002, 9003, 9006, 9016, 9018]

def find_checkpoint(seed):
    patterns = [
        f"NEW_checkpoints/**/ppo_seed{seed}.pt",
        f"checkpoints/**/ppo_seed{seed}.pt"
    ]
    for p in patterns:
        matches = glob.glob(p, recursive=True)
        if matches:
            return matches[0]
    return None

def main():
    print("========================================================================================")
    print("           EVALUATING REQUESTED SEEDS: 8009, 9002, 9003, 9006, 9016, 9018                ")
    print("========================================================================================")

    results = []
    for seed in TARGET_SEEDS:
        ckpt_path = find_checkpoint(seed)
        if not ckpt_path:
            print(f"❌ Checkpoint for Seed {seed} not found!")
            continue

        json_path = ckpt_path.replace(".pt", "_eval.json")
        print(f"\n--- Evaluating Seed {seed} ({ckpt_path}) ---")
        
        # Run 50-episode evaluation script if eval JSON doesn't exist
        cmd = [
            ".venv\\Scripts\\python.exe", "eval/evaluate.py",
            "--checkpoint", ckpt_path,
            "--num-episodes", "50",
            "--save-json", json_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            results.append((seed, data))
            ret = data.get("mean_reward", 0.0)
            std_ret = data.get("std_reward", 0.0)
            lin = data.get("mean_linear_velocity_error", 0.0)
            yaw = data.get("mean_yaw_rate_error", 0.0)
            length = data.get("mean_episode_length", 0.0)
            fall_rate = 1.0 - (length / 1000.0)
            print(f"  Seed {seed:<5} | Mean Return: {ret:7.4f} ± {std_ret:.4f} | LinErr: {lin:6.4f} m/s | YawErr: {yaw:6.4f} rad/s | Fall Rate: {fall_rate:.2%}")
        else:
            print(f"  Failed evaluation for Seed {seed}: {res.stderr}")

    print("\n========================================================================================")
    print("                          REQUESTED SEEDS EVALUATION SUMMARY                             ")
    print("========================================================================================")
    if results:
        returns = [d.get("mean_reward", 0.0) for _, d in results]
        lin_errs = [d.get("mean_linear_velocity_error", 0.0) for _, d in results if "mean_linear_velocity_error" in d]
        yaw_errs = [d.get("mean_yaw_rate_error", 0.0) for _, d in results if "mean_yaw_rate_error" in d]
        print(f"Total Requested Seeds Evaluated: {len(results)}")
        print(f"Grand Mean Return:               {np.mean(returns):.4f} ± {np.std(returns):.4f}")
        if lin_errs: print(f"Grand Mean LinErr:               {np.mean(lin_errs):.4f} m/s")
        if yaw_errs: print(f"Grand Mean YawErr:               {np.mean(yaw_errs):.4f} rad/s")
    print("========================================================================================")

if __name__ == "__main__":
    main()
