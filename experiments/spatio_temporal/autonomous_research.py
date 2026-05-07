#!/usr/bin/env python3
"""Autonomous research loop for Spatio-Temporal GNN architectures.

Extended from temporal-only loop with spatial hypothesis generation:
- Adjacency type (physical vs correlation)
- Graph layer type (GCNConv vs ChebConv)
- Sparsity (K neighbors)
- Graph depth
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

# Temporal baseline target (from experiments/temporal_only)
TEMPORAL_BASELINE_RMSE = 15.000


@dataclass
class ExperimentResult:
    """Results from a single ST-GCN training run."""
    run_name: str
    test_rmse: float
    test_mae: float
    test_r2: float
    runtime_sec: float
    epochs: int
    batch_size: int
    lr: float
    hidden: int
    graph_conv: str
    graph_layers: int
    dropout: float
    adj_type: str
    k_neighbors: int
    seed: int


@dataclass
class Hypothesis:
    """A proposed GNN experiment with spatial reasoning."""
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
            graph_conv=data['graph_conv'],
            graph_layers=int(data['graph_layers']),
            dropout=float(data['dropout']),
            adj_type=data['adj_type'],
            k_neighbors=int(data['k_neighbors']),
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


def analyze_spatial_performance(results: list[ExperimentResult]) -> dict:
    """Analyze spatial-specific performance patterns."""
    if not results:
        return {"status": "no_data", "beat_temporal": False}

    best = find_best_result(results)
    latest = results[-1]

    analysis = {
        "best_rmse": best.test_rmse,
        "best_config": {
            "hidden": best.hidden,
            "graph_conv": best.graph_conv,
            "graph_layers": best.graph_layers,
            "dropout": best.dropout,
            "adj_type": best.adj_type,
            "k_neighbors": best.k_neighbors,
            "lr": best.lr,
        },
        "latest_rmse": latest.test_rmse,
        "beat_temporal": best.test_rmse < TEMPORAL_BASELINE_RMSE,
        "spatial_gain": TEMPORAL_BASELINE_RMSE - best.test_rmse,
        "num_experiments": len(results),
    }

    # Detect spatial-specific patterns
    analysis["patterns"] = []

    # Check adjacency type performance
    physical = [r for r in results if r.adj_type == "physical"]
    correlation = [r for r in results if r.adj_type == "correlation"]
    if physical and correlation:
        avg_phys = sum(r.test_rmse for r in physical) / len(physical)
        avg_corr = sum(r.test_rmse for r in correlation) / len(correlation)
        if avg_phys < avg_corr:
            analysis["patterns"].append("physical_better")
        else:
            analysis["patterns"].append("correlation_better")

    # Check graph conv type
    gcn_results = [r for r in results if r.graph_conv == "gcn"]
    cheb_results = [r for r in results if r.graph_conv == "cheb"]
    if gcn_results and cheb_results:
        avg_gcn = sum(r.test_rmse for r in gcn_results) / len(gcn_results)
        avg_cheb = sum(r.test_rmse for r in cheb_results) / len(cheb_results)
        if avg_gcn < avg_cheb:
            analysis["patterns"].append("gcn_better")
        else:
            analysis["patterns"].append("cheb_better")

    # Check sparsity (K)
    k_values = set(r.k_neighbors for r in results)
    if len(k_values) > 1:
        analysis["patterns"].append("k_explored")

    # Diagnose spatial blurring
    # If performance is worse than temporal baseline, check if graph is too connected
    if not analysis["beat_temporal"]:
        high_k = [r for r in results if r.k_neighbors > 15]
        if high_k:
            avg_high_k = sum(r.test_rmse for r in high_k) / len(high_k)
            if avg_high_k > TEMPORAL_BASELINE_RMSE:
                analysis["patterns"].append("spatial_blurring_suspected")

    # Diagnose adjacency noise
    # If correlation adjacency performs poorly, may have noisy edges
    if correlation and len(correlation) > 0:
        avg_corr = sum(r.test_rmse for r in correlation) / len(correlation)
        if avg_corr > TEMPORAL_BASELINE_RMSE * 1.1:
            analysis["patterns"].append("adjacency_noise_suspected")

    return analysis


def generate_spatial_hypothesis(analysis: dict, iteration_num: int) -> Hypothesis:
    """Generate GNN hypothesis with spatial reasoning."""
    patterns = analysis.get("patterns", [])
    best_config = analysis.get("best_config", {})
    beat_temporal = analysis.get("beat_temporal", False)
    num_experiments = analysis.get("num_experiments", 0)

    # Phase 1 (iter 1-10): Spatial structure search
    if iteration_num <= 10:
        if iteration_num == 1:
            # Start with physical adjacency, moderate K
            return Hypothesis(
                config={
                    "hidden": 64,
                    "graph_conv": "gcn",
                    "graph_layers": 2,
                    "dropout": 0.0,
                    "adj_type": "physical",
                    "k_neighbors": 10,
                    "lr": 1e-3,
                    "epochs": 5,
                    "batch_size": 64,
                },
                reasoning="Baseline ST-GCN with physical adjacency. Goal: beat temporal RMSE 15.000",
                expected_outcome="Should leverage spatial structure for better forecasts",
            )

        elif "spatial_blurring_suspected" in patterns or (num_experiments >= 2 and not beat_temporal):
            # Reduce K to prevent over-smoothing
            current_k = best_config.get("k_neighbors", 10)
            new_k = max(3, current_k - 5)  # Reduce by 5, minimum 3

            return Hypothesis(
                config={
                    "hidden": best_config.get("hidden", 64),
                    "graph_conv": best_config.get("graph_conv", "gcn"),
                    "graph_layers": 1,  # Shallow to reduce smoothing
                    "dropout": 0.1,
                    "adj_type": best_config.get("adj_type", "physical"),
                    "k_neighbors": new_k,
                    "lr": best_config.get("lr", 1e-3),
                    "epochs": 15,
                    "batch_size": 64,
                },
                reasoning=f"Spatial blurring suspected. Reducing K to {new_k} and using 1 graph layer to preserve node identity",
                expected_outcome="Less over-smoothing, nodes retain unique temporal signals",
            )

        elif "adjacency_noise_suspected" in patterns:
            # Switch to physical or reduce correlation noise
            return Hypothesis(
                config={
                    "hidden": best_config.get("hidden", 64),
                    "graph_conv": best_config.get("graph_conv", "gcn"),
                    "graph_layers": 1,  # Shallow to limit noise propagation
                    "dropout": 0.0,
                    "adj_type": "physical",  # More reliable
                    "k_neighbors": 8,
                    "lr": best_config.get("lr", 1e-3),
                    "epochs": 10,
                    "batch_size": 64,
                },
                reasoning="Correlation adjacency may be noisy. Switching to physical distance with shallow GNN",
                expected_outcome="Cleaner signal from physical neighbors",
            )

        elif "physical_better" in patterns:
            # Optimize physical adjacency
            return Hypothesis(
                config={
                    "hidden": 96,
                    "graph_conv": "cheb",  # Try Chebyshev for localized filters
                    "graph_layers": 2,
                    "dropout": 0.05,
                    "adj_type": "physical",
                    "k_neighbors": best_config.get("k_neighbors", 10),
                    "lr": 1e-3,
                    "epochs": 15,
                    "batch_size": 64,
                },
                reasoning="Physical adjacency works well. Testing ChebConv with more capacity",
                expected_outcome="Localized spatial filters may capture traffic patterns better",
            )

        elif "correlation_better" in patterns:
            # Optimize correlation adjacency
            return Hypothesis(
                config={
                    "hidden": best_config.get("hidden", 64),
                    "graph_conv": best_config.get("graph_conv", "gcn"),
                    "graph_layers": 2,
                    "dropout": 0.1,
                    "adj_type": "correlation",
                    "k_neighbors": 15,  # More neighbors for correlation
                    "lr": 5e-4,
                    "epochs": 20,
                    "batch_size": 64,
                },
                reasoning="Correlation adjacency captures temporal patterns. Increasing K and training longer",
                expected_outcome="More training may help model learn correlation structure",
            )

        else:
            # Try correlation adjacency if physical isn't working
            if best_config.get("adj_type") == "physical" and num_experiments >= 4:
                return Hypothesis(
                    config={
                        "hidden": best_config.get("hidden", 64),
                        "graph_conv": best_config.get("graph_conv", "gcn"),
                        "graph_layers": 1,
                        "dropout": 0.1,
                        "adj_type": "correlation",  # Switch adjacency type
                        "k_neighbors": 8,
                        "lr": 1e-3,
                        "epochs": 15,
                        "batch_size": 64,
                    },
                    reasoning="Physical adjacency not working. Testing correlation-based graph construction",
                    expected_outcome="Temporal correlation may better capture traffic dependencies than physical distance",
                )
            else:
                # Continue exploring with best adjacency type
                return Hypothesis(
                    config={
                        "hidden": 96,  # Increase capacity
                        "graph_conv": "cheb" if best_config.get("graph_conv") == "gcn" else "gcn",
                        "graph_layers": 1,
                        "dropout": 0.05,
                        "adj_type": best_config.get("adj_type", "physical"),
                        "k_neighbors": 5,
                        "lr": 1e-3,
                        "epochs": 20,
                        "batch_size": 64,
                    },
                    reasoning="Testing alternative graph conv layer with more capacity and sparse connections",
                    expected_outcome="ChebConv/GCN may better handle local spatial patterns",
                )

    elif iteration_num <= 20:
        # Phase 2: Hyperparameter optimization
        return Hypothesis(
            config={
                "hidden": best_config.get("hidden", 64),
                "graph_conv": best_config.get("graph_conv", "gcn"),
                "graph_layers": best_config.get("graph_layers", 2),
                "dropout": 0.1,
                "adj_type": best_config.get("adj_type", "physical"),
                "k_neighbors": best_config.get("k_neighbors", 10),
                "lr": 5e-4,
                "epochs": 30,
                "batch_size": 128,
                },
            reasoning="Hyperparameter tuning: lower LR, larger batch, longer training",
            expected_outcome="Better convergence with stable gradients",
        )

    else:
        # Phase 3: Final long training
        return Hypothesis(
            config={
                "hidden": best_config.get("hidden", 64),
                "graph_conv": best_config.get("graph_conv", "gcn"),
                "graph_layers": best_config.get("graph_layers", 2),
                "dropout": best_config.get("dropout", 0.0),
                "adj_type": best_config.get("adj_type", "physical"),
                "k_neighbors": best_config.get("k_neighbors", 10),
                "lr": best_config.get("lr", 1e-3),
                "epochs": 50,
                "batch_size": 64,
            },
            reasoning="Final convergence run with best spatial config",
            expected_outcome="Establish spatial GNN error floor",
        )


def run_experiment(config: dict, run_name: str) -> bool:
    """Run a GNN training experiment."""
    cmd = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--hidden", str(config["hidden"]),
        "--graph-conv", config["graph_conv"],
        "--graph-layers", str(config["graph_layers"]),
        "--dropout", str(config["dropout"]),
        "--adj-type", config["adj_type"],
        "--k-neighbors", str(config["k_neighbors"]),
        "--seed", str(config.get("seed", 0)),
        "--run-name", run_name,
    ]

    print(f"\nRunning experiment: {run_name}")
    print(f"Config: {json.dumps(config, indent=2)}")

    try:
        subprocess.run(cmd, cwd=_PROJECT_ROOT, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed: {e}")
        return False


def git_commit(message: str) -> bool:
    """Commit current changes."""
    try:
        subprocess.run(
            ["git", "add", "experiments/spatio_temporal/logs/"],
            cwd=_PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            ["git", "add", str(_STATE_FILE.relative_to(_PROJECT_ROOT))],
            cwd=_PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=_PROJECT_ROOT,
            check=True,
        )
        print(f"✓ Committed: {message}")
        return True
    except subprocess.CalledProcessError:
        print("✗ Git commit failed")
        return False


def save_state(state: dict) -> None:
    """Save research state."""
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def load_state() -> dict:
    """Load research state."""
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return {"iteration": 0, "best_rmse": float('inf'), "beat_temporal": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--rmse-target", type=float, default=14.0,
                        help="Stop if RMSE reaches this target")
    args = parser.parse_args()

    print("=" * 80)
    print("AUTONOMOUS SPATIO-TEMPORAL RESEARCH LOOP")
    print(f"Target: Beat temporal baseline RMSE {TEMPORAL_BASELINE_RMSE:.3f}")
    print("=" * 80)

    state = load_state()
    iteration_start = state.get("iteration", 0)

    for i in range(args.max_iterations):
        iteration_num = iteration_start + i + 1
        print(f"\n{'=' * 80}")
        print(f"SPATIAL ITERATION {iteration_num}")
        print(f"{'=' * 80}")

        # 1. Load and analyze
        print("\n[1/5] Loading previous results...")
        results = load_all_results()
        print(f"Found {len(results)} previous experiments")

        print("\n[2/5] Analyzing spatial performance...")
        analysis = analyze_spatial_performance(results)
        print(f"Analysis: {json.dumps(analysis, indent=2)}")

        # 2. Generate hypothesis
        print("\n[3/5] Generating spatial hypothesis...")
        hypothesis = generate_spatial_hypothesis(analysis, iteration_num)
        print(f"Reasoning: {hypothesis.reasoning}")
        print(f"Expected: {hypothesis.expected_outcome}")

        # 3. Run experiment
        run_name = f"spatial_iter_{iteration_num:03d}"
        print(f"\n[4/5] Running experiment: {run_name}...")

        success = run_experiment(hypothesis.config, run_name)
        if not success:
            print("Experiment failed. Stopping.")
            break

        # 4. Evaluate
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

            beat_temporal = new_result.test_rmse < TEMPORAL_BASELINE_RMSE
            if beat_temporal:
                spatial_gain = TEMPORAL_BASELINE_RMSE - new_result.test_rmse
                print(f"🎯 BEAT TEMPORAL BASELINE! Spatial gain: {spatial_gain:.4f}")

            commit_msg = (
                f"spatial: iteration {iteration_num} - RMSE {new_result.test_rmse:.4f} "
                f"(improved by {improvement:.4f})\n\n"
                f"Hypothesis: {hypothesis.reasoning}\n"
                f"Config: {new_result.adj_type} adj, K={new_result.k_neighbors}, "
                f"{new_result.graph_conv}×{new_result.graph_layers}, hidden={new_result.hidden}"
            )
            git_commit(commit_msg)
            state["best_rmse"] = new_result.test_rmse
            state["beat_temporal"] = beat_temporal
        else:
            regression = new_result.test_rmse - best_rmse
            print(f"\n✗ REGRESSION: {regression:.4f} RMSE increase")

        # Update state
        state["iteration"] = iteration_num
        save_state(state)

        # Check target
        if new_result.test_rmse <= args.rmse_target:
            print(f"\n🎉 TARGET REACHED! RMSE {new_result.test_rmse:.4f} <= {args.rmse_target}")
            break

        print(f"\nIteration {iteration_num} complete. Best RMSE: {state['best_rmse']:.4f}")

    print("\n" + "=" * 80)
    print("SPATIAL RESEARCH COMPLETE")
    print("=" * 80)

    final_results = load_all_results()
    final_best = find_best_result(final_results)

    if final_best:
        print(f"\n🏆 Final Best Spatial Result:")
        print(f"  Run: {final_best.run_name}")
        print(f"  RMSE: {final_best.test_rmse:.4f}")
        print(f"  MAE: {final_best.test_mae:.4f}")
        print(f"  R²: {final_best.test_r2:.4f}")
        print(f"  Config: {final_best.adj_type} adj, K={final_best.k_neighbors}, "
              f"{final_best.graph_conv}×{final_best.graph_layers}")

        print(f"\n📊 HEAD-TO-HEAD COMPARISON:")
        print(f"  Temporal-Only RMSE: {TEMPORAL_BASELINE_RMSE:.4f}")
        print(f"  Spatio-Temporal RMSE: {final_best.test_rmse:.4f}")
        spatial_gain = TEMPORAL_BASELINE_RMSE - final_best.test_rmse
        print(f"  Spatial Gain: {spatial_gain:.4f} ({spatial_gain/TEMPORAL_BASELINE_RMSE*100:.2f}%)")

        if spatial_gain > 0:
            print("\n✅ SPATIAL STRUCTURE ADDS VALUE!")
        else:
            print("\n⚠️  Spatial structure did not improve over temporal baseline")
            print("     Possible causes:")
            if "spatial_blurring_suspected" in analysis.get("patterns", []):
                print("     - Spatial blurring: nodes became too similar")
            if "adjacency_noise_suspected" in analysis.get("patterns", []):
                print("     - Adjacency noise: noisy edges hurt performance")


if __name__ == "__main__":
    main()
