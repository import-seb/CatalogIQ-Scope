"""Confirm synthetic bootstrap behavior with 19,999 draws per simulated catalog."""
from catalogiq.accuracy_simulation_stability import run_stability
from catalogiq.paths import ROOT_PATH

if __name__ == "__main__":
    result = run_stability(ROOT_PATH/"data/processed/accuracy_bootstrap_stability_v1_20261103")
    print(f"Saved independent simulation confirmation to {result}")
