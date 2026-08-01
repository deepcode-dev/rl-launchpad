import glob
import json
import os
import subprocess
import re
import numpy as np

def extract_seed(filepath):
    match = re.search(r'ppo_seed(\d+)\.pt', filepath)
    return int(match.group(1)) if match else 999999

def main():
    print("========================================================================================")
    print("           DYNAMICALLY EVALUATING ALL SEEDS IN NEW_checkpoints/ppo_v2                  ")
    print("========================================================================================")

    # Discover all .pt files inside NEW_checkpoints/ppo_v2
    ckpts = glob.glob("NEW_checkpoints/ppo_v2/**/*.pt", recursive=True)
    ckpts = [c for c in ckpts if not c.endswith("_eval.json")]
    ckpts.sort(key=extract_seed)

    print(f"Found {len(ckpts)} checkpoint files in NEW_checkpoints/ppo_v2.\n")

    results = []
    for ckpt_path in ckpts:
        seed = extract_seed(ckpt_path)
        json_path = ckpt_path.replace(".pt", "_eval.json")
        
        # Run 50-episode evaluation if JSON does not exist or is missing metrics
        if not os.path.exists(json_path):
            print(f"--> Evaluating Seed {seed} ({ckpt_path})...", flush=True)
            print(f"  (output auto-saved to {json_path})", flush=True)
            cmd = [
                "uv", "run", "python", "eval/evaluate.py",
                "--checkpoint", ckpt_path,
                "--num-episodes", "50",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)

            if res.returncode != 0:
                print(f"  WARNING: Seed {seed} eval failed (stderr): {res.stderr.strip()[-200:]}", flush=True)

        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            ret = data.get("mean_reward", 0.0)
            std_ret = data.get("std_reward", 0.0)
            lin = data.get("mean_linear_velocity_error", 0.0)
            yaw = data.get("mean_yaw_rate_error", 0.0)
            length = data.get("mean_episode_length", 1000.0)
            fall_rate = max(0.0, 1.0 - (length / 1000.0))
            results.append({
                "seed": seed,
                "ckpt": ckpt_path,
                "reward": ret,
                "std": std_ret,
                "lin_err": lin,
                "yaw_err": yaw,
                "fall_rate": fall_rate
            })
            print(f"  Seed {seed:<5} | Mean Return: {ret:7.4f} ± {std_ret:.4f} | LinErr: {lin:6.4f} m/s | YawErr: {yaw:6.4f} rad/s | Fall Rate: {fall_rate:.2%}")

    print("\n========================================================================================")
    print("                    ALL NEW_checkpoints/ppo_v2 SEEDS EVALUATION SUMMARY                  ")
    print("========================================================================================")
    if results:
        returns = [r["reward"] for r in results]
        lin_errs = [r["lin_err"] for r in results]
        yaw_errs = [r["yaw_err"] for r in results]
        print(f"Total Evaluated Seeds:  {len(results)}")
        print(f"Mean Return:           {np.mean(returns):.4f} ± {np.std(returns):.4f}")
        print(f"Mean LinErr:           {np.mean(lin_errs):.4f} m/s")
        print(f"Mean YawErr:           {np.mean(yaw_errs):.4f} rad/s")
        
        # Sort by LinErr ascending to highlight champion seeds
        sorted_by_lin = sorted(results, key=lambda x: x["lin_err"])
        print("\n🏆 Top 10 Seeds Ranked by Lowest LinErr (Best Command Tracking):")
        for i, r in enumerate(sorted_by_lin[:10], 1):
            print(f"  {i:2d}. Seed {r['seed']:<5} | LinErr: {r['lin_err']:.4f} m/s | Return: {r['reward']:.4f} | Fall Rate: {r['fall_rate']:.2%}")

    print("========================================================================================")

if __name__ == "__main__":
    main()
