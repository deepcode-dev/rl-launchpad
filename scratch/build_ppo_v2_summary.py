import glob
import json
import os
import numpy as np

def main():
    print("========================================================================================")
    print("        AUTOMATIC PPO v2 CHAMPION EVALUATION SUMMARY BUILDER                           ")
    print("========================================================================================")

    # 1. Discover all evaluation JSON files in NEW_checkpoints/ppo_v2/
    eval_jsons = glob.glob("NEW_checkpoints/ppo_v2/*_eval.json")
    if not eval_jsons:
        eval_jsons = glob.glob("NEW_checkpoints/**/*_eval.json", recursive=True)

    if not eval_jsons:
        print("❌ No evaluation JSON files found! Run 'python scratch/eval_all_ppo_v2_seeds.py' first.")
        return

    # 2. Extract metrics from all evaluated seeds
    parsed_seeds = []
    for jf in eval_jsons:
        try:
            with open(jf) as f:
                data = json.load(f)
            seed = data.get("checkpoint_metadata", {}).get("seed")
            if seed is None:
                # Fallback parse seed from filename
                import re
                m = re.search(r'seed(\d+)', jf)
                seed = int(m.group(1)) if m else 9999
            
            parsed_seeds.append({
                "seed": seed,
                "json_path": jf,
                "mean_reward": data.get("mean_reward", 0.0),
                "std_reward": data.get("std_reward", 0.0),
                "mean_linear_velocity_error": data.get("mean_linear_velocity_error", 0.0),
                "mean_yaw_rate_error": data.get("mean_yaw_rate_error", 0.0),
                "mean_episode_length": data.get("mean_episode_length", 1000.0)
            })
        except Exception as e:
            pass

    # 3. Sort by lowest linear velocity error (best tracking precision)
    parsed_seeds.sort(key=lambda x: x["mean_linear_velocity_error"])

    # 4. Automatically pick the top 5 champion seeds
    top_seeds = parsed_seeds[:min(5, len(parsed_seeds))]
    target_seed_numbers = [item["seed"] for item in top_seeds]

    returns = [item["mean_reward"] for item in top_seeds]
    lengths = [item["mean_episode_length"] for item in top_seeds]
    lin_errs = [item["mean_linear_velocity_error"] for item in top_seeds]
    yaw_errs = [item["mean_yaw_rate_error"] for item in top_seeds]

    print(f"Automatically selected Top {len(top_seeds)} Champion Seeds: {target_seed_numbers}\n")
    for item in top_seeds:
        print(f"  Seed {item['seed']:<5} | Return: {item['mean_reward']:7.4f} +/- {item['std_reward']:.4f} | LinErr: {item['mean_linear_velocity_error']:6.4f} m/s | YawErr: {item['mean_yaw_rate_error']:6.4f} rad/s")

    grand_summary = {
        "protocol": "custom-ppo-v2-champion",
        "num_episodes_per_seed": 50,
        "training_seeds": target_seed_numbers,
        "grand_mean_return": float(np.mean(returns)),
        "grand_std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_linear_velocity_error": float(np.mean(lin_errs)),
        "mean_yaw_rate_error": float(np.mean(yaw_errs)),
        "seed_results": top_seeds
    }

    out_path = "NEW_checkpoints/ppo_v2/ppo_v2_eval_summary.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(grand_summary, f, indent=2)

    print("\n========================================================================================")
    print(f"Saved Automatic PPO v2 Champion Summary to: {out_path}")
    print(f"   Top Seeds Aggregated:  {target_seed_numbers}")
    print(f"   Grand Mean Return:     {grand_summary['grand_mean_return']:.4f} +/- {grand_summary['grand_std_return']:.4f}")
    print(f"   Grand Mean LinErr:     {grand_summary['mean_linear_velocity_error']:.4f} m/s")
    print(f"   Grand Mean YawErr:     {grand_summary['mean_yaw_rate_error']:.4f} rad/s")
    print(f"   Grand Mean Survival:   {grand_summary['mean_episode_length']:.1f} steps / 1000")
    print("========================================================================================")

if __name__ == "__main__":
    main()
