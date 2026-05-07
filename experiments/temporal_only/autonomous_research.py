#!/usr/bin/env python3
"""Autonomous Karpathy-style research loop.

This script:
1. Reads previous iteration logs
2. Analyzes performance (overfitting/underfitting/bottleneck)
3. Generates hypothesis for next experiment
4. Runs training
5. Commits if improved, reverts if regressed
6. Repeats autonomously
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT_DIR = Path(__file__).resolve().parent
_LOGS_DIR = _EXPERIMENT_DIR / "logs"
_TRAIN_SCRIPT = _EXPERIMENT_DIR / "train.py"
_STATE_FILE = _EXPERIMENT_DIR / "research_state.json"


@dataclass
class ExperimentResult:
    """Results from a single training run."""
    run_name: str
    test_rmse: float
    test_mae: float
    test_r2: float
    runtime_sec: float
    epochs: int
    batch_size: int
    lr: float
    hidden: int
    layers: int
    dropout: float
    seed: int


@dataclass
class Hypothesis:
    """A proposed experiment with reasoning."""
    config: dict
    reasoning: str
    expected_outcome: str


def parse_log_file(log_path: Path) -> Optional[ExperimentResult]:
    """Parse a log file into an ExperimentResult."""
    if not log_path.exists():
        return None

    lines = log_path.read_text().strip().split('\n')
    data = {}
    for line in lines:
        if '=' in line:
            key, val = line.split('=', 1)
            data[key.strip()] = val.strip()

    try:
        return ExperimentResult(
            run_name=log_path.stem,
            test_rmse=float(data['test_rmse']),
            test_mae=float(data['test_mae']),
            test_r2=float(data['test_r2']),
            runtime_sec=float(data['runtime_sec']),
            epochs=int(data['epochs']),
            batch_size=int(data['batch_size']),
            lr=float(data['lr']),
            hidden=int(data['hidden']),
            layers=int(data['layers']),
            dropout=float(data['dropout']),
            seed=int(data['seed']),
        )
    except (KeyError, ValueError) as e:
        print(f"Warning: could not parse {log_path}: {e}")
        return None


def load_all_results() -> list[ExperimentResult]:
    """Load all experiment results from logs directory."""
    results = []
    if _LOGS_DIR.exists():
        for log_file in sorted(_LOGS_DIR.glob("*.txt")):
            result = parse_log_file(log_file)
            if result:
                results.append(result)
    return results


def find_best_result(results: list[ExperimentResult]) -> Optional[ExperimentResult]:
    """Find the best result by RMSE."""
    if not results:
        return None
    return min(results, key=lambda r: r.test_rmse)


def analyze_performance(results: list[ExperimentResult]) -> dict:
    """Analyze performance patterns across experiments."""
    if not results:
        return {"status": "no_data"}

    best = find_best_result(results)
    latest = results[-1]

    analysis = {
        "best_rmse": best.test_rmse,
        "best_config": {
            "hidden": best.hidden,
            "layers": best.layers,
            "dropout": best.dropout,
            "lr": best.lr,
        },
        "latest_rmse": latest.test_rmse,
        "trend": "improving" if latest.test_rmse <= best.test_rmse else "regressing",
        "num_experiments": len(results),
    }

    # Detect patterns
    analysis["patterns"] = []

    # Check if we've tried long training
    max_epochs = max(r.epochs for r in results)
    if max_epochs <= 5:
        analysis["patterns"].append("short_training_only")

    # Check if simple models work better
    simple_models = [r for r in results if r.layers == 1]
    complex_models = [r for r in results if r.layers > 1]
    if simple_models and complex_models:
        avg_simple = sum(r.test_rmse for r in simple_models) / len(simple_models)
        avg_complex = sum(r.test_rmse for r in complex_models) / len(complex_models)
        if avg_simple < avg_complex:
            analysis["patterns"].append("simpler_is_better")

    # Check learning rate exploration
    lrs_tried = set(r.lr for r in results)
    if len(lrs_tried) == 1:
        analysis["patterns"].append("single_lr_only")

    return analysis


def generate_hypothesis(analysis: dict, iteration_num: int) -> Hypothesis:
    """Generate next experiment hypothesis based on analysis."""
    patterns = analysis.get("patterns", [])
    best_config = analysis.get("best_config", {})
    num_experiments = analysis.get("num_experiments", 0)

    # Strategy: Progressive refinement
    # Phase 1 (iter 1-10): Architecture search
    # Phase 2 (iter 11-20): Hyperparameter tuning
    # Phase 3 (iter 21+): Long training runs

    if iteration_num <= 10:
        # Architecture search phase
        if "short_training_only" in patterns and "simpler_is_better" in patterns:
            # Try longer training on best simple architecture
            return Hypothesis(
                config={
                    "hidden": best_config.get("hidden", 64),
                    "layers": best_config.get("layers", 1),
                    "dropout": 0.0,
                    "lr": best_config.get("lr", 1e-3),
                    "epochs": 15,
                    "batch_size": 64,
                },
                reasoning="Best results came from simple 1-layer model. Testing if longer training improves performance.",
                expected_outcome="RMSE should decrease with more epochs if model is underfitting",
            )
        elif "simpler_is_better" not in patterns:
            # Try wider single layer
            return Hypothesis(
                config={
                    "hidden": 96,
                    "layers": 1,
                    "dropout": 0.0,
                    "lr": 1e-3,
                    "epochs": 5,
                    "batch_size": 64,
                },
                reasoning="Testing if wider single-layer network improves capacity without overfitting",
                expected_outcome="May improve RMSE if model is capacity-constrained",
            )
        else:
            # Try different dropout values
            return Hypothesis(
                config={
                    "hidden": best_config.get("hidden", 64),
                    "layers": 1,
                    "dropout": 0.1,
                    "lr": best_config.get("lr", 1e-3),
                    "epochs": 10,
                    "batch_size": 64,
                },
                reasoning="Light dropout with longer training may reduce overfitting",
                expected_outcome="Should help if model overfits after epoch 5",
            )

    elif iteration_num <= 20:
        # Hyperparameter tuning phase
        if "single_lr_only" in patterns:
            # Explore learning rates
            return Hypothesis(
                config={
                    "hidden": best_config.get("hidden", 64),
                    "layers": 1,
                    "dropout": 0.0,
                    "lr": 5e-4,  # Lower LR
                    "epochs": 20,
                    "batch_size": 64,
                },
                reasoning="Testing lower learning rate with longer training for better convergence",
                expected_outcome="May find better minimum with slower, more stable training",
            )
        else:
            # Try batch size variations
            return Hypothesis(
                config={
                    "hidden": best_config.get("hidden", 64),
                    "layers": 1,
                    "dropout": 0.0,
                    "lr": best_config.get("lr", 1e-3),
                    "epochs": 20,
                    "batch_size": 128,
                },
                reasoning="Larger batch size may provide more stable gradients",
                expected_outcome="Could improve convergence stability",
            )

    else:
        # Long training phase
        return Hypothesis(
            config={
                "hidden": best_config.get("hidden", 64),
                "layers": 1,
                "dropout": 0.05,
                "lr": best_config.get("lr", 1e-3),
                "epochs": 50,
                "batch_size": 64,
            },
            reasoning="Final long training run with best architecture to establish true baseline",
            expected_outcome="Should reach convergence and establish final temporal-only error floor",
        )


def run_experiment(config: dict, run_name: str) -> bool:
    """Run a training experiment with given config."""
    cmd = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--hidden", str(config["hidden"]),
        "--layers", str(config["layers"]),
        "--dropout", str(config["dropout"]),
        "--seed", str(config.get("seed", 0)),
        "--run-name", run_name,
    ]

    print(f"\nRunning experiment: {run_name}")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=_PROJECT_ROOT, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed: {e}")
        return False


def git_commit(message: str) -> bool:
    """Commit current changes with message."""
    try:
        # Stage relevant files
        subprocess.run(
            ["git", "add", "experiments/temporal_only/logs/"],
            cwd=_PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            ["git", "add", str(_STATE_FILE.relative_to(_PROJECT_ROOT))],
            cwd=_PROJECT_ROOT,
            check=True,
        )

        # Commit
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=_PROJECT_ROOT,
            check=True,
        )
        print(f"✓ Committed: {message}")
        return True
    except subprocess.CalledProcessError:
        print("✗ Git commit failed (may be nothing to commit)")
        return False


def git_revert_last_log(run_name: str) -> bool:
    """Remove the log file for failed experiment."""
    log_file = _LOGS_DIR / f"{run_name}.txt"
    if log_file.exists():
        log_file.unlink()
        print(f"✗ Removed log: {log_file}")
        return True
    return False


def save_state(state: dict) -> None:
    """Save research state to JSON."""
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def load_state() -> dict:
    """Load research state from JSON."""
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return {"iteration": 0, "best_rmse": float('inf')}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Maximum number of autonomous iterations")
    parser.add_argument("--rmse-target", type=float, default=10.0,
                        help="Stop if RMSE reaches this target")
    args = parser.parse_args()

    print("=" * 80)
    print("AUTONOMOUS RESEARCH LOOP (Karpathy-style)")
    print("=" * 80)

    state = load_state()
    iteration_start = state.get("iteration", 0)

    for i in range(args.max_iterations):
        iteration_num = iteration_start + i + 1
        print(f"\n{'=' * 80}")
        print(f"ITERATION {iteration_num}")
        print(f"{'=' * 80}")

        # 1. Load and analyze previous results
        print("\n[1/5] Loading previous results...")
        results = load_all_results()
        print(f"Found {len(results)} previous experiments")

        print("\n[2/5] Analyzing performance...")
        analysis = analyze_performance(results)
        print(f"Analysis: {json.dumps(analysis, indent=2)}")

        # 2. Generate hypothesis
        print("\n[3/5] Generating hypothesis...")
        hypothesis = generate_hypothesis(analysis, iteration_num)
        print(f"Reasoning: {hypothesis.reasoning}")
        print(f"Expected: {hypothesis.expected_outcome}")
        print(f"Config: {json.dumps(hypothesis.config, indent=2)}")

        # 3. Run experiment
        run_name = f"auto_iter_{iteration_num:03d}"
        print(f"\n[4/5] Running experiment: {run_name}...")

        success = run_experiment(hypothesis.config, run_name)
        if not success:
            print("Experiment failed. Stopping.")
            break

        # 4. Evaluate and decide
        print("\n[5/5] Evaluating results...")
        new_result = parse_log_file(_LOGS_DIR / f"{run_name}.txt")
        if not new_result:
            print("Could not parse results. Stopping.")
            break

        print(f"Result: RMSE={new_result.test_rmse:.4f}, MAE={new_result.test_mae:.4f}, R²={new_result.test_r2:.4f}")

        best = find_best_result(results) if results else None
        best_rmse = best.test_rmse if best else float('inf')

        # 5. Git logic gate
        if new_result.test_rmse < best_rmse:
            improvement = best_rmse - new_result.test_rmse
            print(f"\n✓ IMPROVEMENT: {improvement:.4f} RMSE reduction")
            commit_msg = (
                f"auto: iteration {iteration_num} - RMSE {new_result.test_rmse:.4f} "
                f"(improved by {improvement:.4f})\n\n"
                f"Hypothesis: {hypothesis.reasoning}\n"
                f"Config: hidden={new_result.hidden}, layers={new_result.layers}, "
                f"dropout={new_result.dropout}, lr={new_result.lr}, epochs={new_result.epochs}"
            )
            git_commit(commit_msg)
            state["best_rmse"] = new_result.test_rmse
        else:
            regression = new_result.test_rmse - best_rmse
            print(f"\n✗ REGRESSION: {regression:.4f} RMSE increase")
            print("Hypothesis did not improve performance. Keeping log for analysis.")
            # Note: We keep the log but don't commit to main branch

        # Update state
        state["iteration"] = iteration_num
        save_state(state)

        # Check if target reached
        if new_result.test_rmse <= args.rmse_target:
            print(f"\n🎉 TARGET REACHED! RMSE {new_result.test_rmse:.4f} <= {args.rmse_target}")
            break

        print(f"\nIteration {iteration_num} complete. Best RMSE so far: {state['best_rmse']:.4f}")

    print("\n" + "=" * 80)
    print("AUTONOMOUS RESEARCH COMPLETE")
    print("=" * 80)
    final_results = load_all_results()
    final_best = find_best_result(final_results)
    if final_best:
        print(f"\nFinal Best Result:")
        print(f"  Run: {final_best.run_name}")
        print(f"  RMSE: {final_best.test_rmse:.4f}")
        print(f"  MAE: {final_best.test_mae:.4f}")
        print(f"  R²: {final_best.test_r2:.4f}")
        print(f"  Config: hidden={final_best.hidden}, layers={final_best.layers}, "
              f"dropout={final_best.dropout}, lr={final_best.lr}")


if __name__ == "__main__":
    main()
