import glob
import json
import os
import numpy as np

def main():
    dir_path = "NEW_checkpoints/ppo_v2"
    pts = sorted(glob.glob(os.path.join(dir_path, "ppo_seed*.pt")))
    print("========================================================================================")
    print("                 OFFICIAL 50-EPISODE BENCHMARK EVALUATION AUDIT                        ")
    print("========================================================================================")
    
    results = []
    for pt in pts:
        seed_str = os.path.basename(pt).replace("ppo_seed", "").replace(".pt", "")
        json_path = pt.replace(".pt", "_eval.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            results.append(data)
            ret = data.get("mean_reward", 0.0)
            lin = data.get("mean_linear_velocity_error", 0.0)
            yaw = data.get("mean_yaw_rate_error", 0.0)
            print(f"Seed {seed_str:<5} | Mean Return: {ret:7.4f} | LinErr: {lin:6.4f} m/s | YawErr: {yaw:6.4f} rad/s")
    
    if results:
        returns = [r.get("mean_reward", 0.0) for r in results]
        lin_errs = [r.get("mean_linear_velocity_error", 0.0) for r in results if "mean_linear_velocity_error" in r]
        yaw_errs = [r.get("mean_yaw_rate_error", 0.0) for r in results if "mean_yaw_rate_error" in r]
        print("\n----------------------------------------------------------------------------------------")
        print(f"Total Seeds Evaluated: {len(results)}")
        print(f"Grand Mean Return:     {np.mean(returns):.4f} ± {np.std(returns):.4f}")
        print(f"Grand Mean LinErr:     {np.mean(lin_errs):.4f} m/s")
        print(f"Grand Mean YawErr:     {np.mean(yaw_errs):.4f} rad/s")
        print("========================================================================================")

if __name__ == "__main__":
    main()
