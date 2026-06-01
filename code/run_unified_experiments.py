#!/usr/bin/env python3
"""
KePIN Unified Experiment Runner — All 7 Datasets with Hyperparameter Tuning.

Trains KePIN on:
  1. C-MAPSS FD001-FD004 (Predictive Maintenance)
  2. Jena Climate (Weather Forecasting)
  3. Cylinder Wake (Fluid Dynamics)
  4. Building Energy (Energy Systems)

Uses 4 core loss functions for all domains:
  L_pred:  Huber prediction loss
  L_koop:  Koopman one-step consistency
  L_spec:  Spectral stability (eigenvalue constraint)
  L_multi: Multi-step rollout fidelity

For C-MAPSS degradation domains, adds:
  L_mono:  Monotonicity constraint
  L_asym:  Asymmetric penalty
  L_slope: Slope matching

Hyperparameter tuning per dataset includes:
  - Learning rate
  - Batch size
  - Sequence length adjustments
  - Dropout rate
  - Number of epochs
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd

# Add code directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from kepin_training import train_on_dataset, train_all


# Per-dataset optimized hyperparameters (from grid search)
HYPERPARAMS = {
    "CMAPSS_FD001": {
        "epochs": 150,
        "batch_size": 128,
        "lr": 0.0008,
        "patience": 40,
    },
    "CMAPSS_FD002": {
        "epochs": 120,
        "batch_size": 256,
        "lr": 0.0006,
        "patience": 35,
    },
    "CMAPSS_FD003": {
        "epochs": 150,
        "batch_size": 128,
        "lr": 0.0008,
        "patience": 40,
    },
    "CMAPSS_FD004": {
        "epochs": 120,
        "batch_size": 256,
        "lr": 0.0006,
        "patience": 35,
    },
    "Jena_Climate": {
        "epochs": 80,
        "batch_size": 256,
        "lr": 0.001,
        "patience": 30,
    },
    "Cylinder_Wake": {
        "epochs": 100,
        "batch_size": 128,
        "lr": 0.0008,
        "patience": 30,
    },
    "Building_Energy": {
        "epochs": 100,
        "batch_size": 256,
        "lr": 0.001,
        "patience": 30,
    },
}


def run_all_experiments(config_path, output_base=None, n_runs=1, 
                        dataset_filter=None):
    """Run experiments on all or selected datasets.
    
    Args:
        config_path: Path to JSON config with all dataset definitions
        output_base: Output directory for results
        n_runs: Number of independent runs per dataset (for ensembling)
        dataset_filter: List of dataset names to run (None = all)
    """
    with open(config_path, "r") as f:
        all_configs = json.load(f)
    
    if output_base is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(
            SCRIPT_DIR, "..", "experiments_result", f"kepin_unified_{timestamp}"
        )
    
    os.makedirs(output_base, exist_ok=True)
    
    all_results = []
    
    for ds_config in all_configs:
        ds_name = ds_config.get("name", "unknown")
        
        if dataset_filter and ds_name not in dataset_filter:
            continue
        
        # Get optimized hyperparameters
        hp = HYPERPARAMS.get(ds_name, {
            "epochs": 100,
            "batch_size": 128,
            "lr": 0.0008,
            "patience": 35,
        })
        
        ds_output = os.path.join(output_base, ds_name)
        
        for run in range(n_runs):
            try:
                result = train_on_dataset(
                    ds_config, ds_output,
                    epochs=hp["epochs"],
                    batch_size=hp["batch_size"],
                    lr=hp["lr"],
                    patience=hp["patience"],
                    use_auto_weights=True,
                    run_id=run,
                    verbose=1,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n  FAILED: {ds_name} run {run}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "dataset": ds_name,
                    "run_id": run,
                    "error": str(e),
                })
    
    # Summary
    print(f"\n{'='*70}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    
    summary_rows = []
    for r in all_results:
        if "error" not in r:
            summary_rows.append({
                "Dataset": r["dataset"],
                "Run": r["run_id"],
                "RMSE": f"{r['rmse']:.4f}",
                "MAE": f"{r['mae']:.4f}",
                "MonoViol": f"{r['mono_violation']:.4f}",
                "Tier": r["arch_tier"],
                "Epochs": r["epochs_trained"],
            })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        
        summary_path = os.path.join(output_base, "all_results_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        json_path = os.path.join(output_base, "all_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_base}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="KePIN Unified Experiments")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "datasets_all_config.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to run")
    args = parser.parse_args()
    
    run_all_experiments(
        config_path=args.config,
        output_base=args.output,
        n_runs=args.n_runs,
        dataset_filter=args.datasets,
    )
