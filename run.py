"""Predator / prey coevolution via parallel hill climber.

Usage:
    python run.py --config config.yaml
"""

from simulator import Simulator
from utils import load_config
from robot import sample_robot, MAX_N_MASSES, MAX_N_SPRINGS
from evolution import (
    HallOfFame,
    mutate_individual,
    train_controllers,
    evaluate_fitness,
)
from argparse import ArgumentParser
from copy import deepcopy
import time
import numpy as np
from tqdm import tqdm


def run_evolution(config):
    np.random.seed(config["seed"])
    coevo = config["coevolution"]
    P = coevo["population_size"]

    # ------------------------------------------------------------------
    # Simulator setup (2*P slots: P predators + P prey)
    # ------------------------------------------------------------------
    sim_config = config["simulator"].copy()
    sim_config["n_sims"] = 2 * P
    sim_config["n_masses"] = MAX_N_MASSES
    sim_config["n_springs"] = MAX_N_SPRINGS

    simulator = Simulator(
        sim_config, config["taichi"], config["seed"], needs_grad=True
    )
    # Use a separate (lower) effort penalty for gradient training to avoid
    # suppressing movement.  Evolutionary fitness uses the full lambda_effort.
    simulator.lambda_effort[None] = coevo.get("training_lambda_effort", 0.0)

    # ------------------------------------------------------------------
    # Initialise populations (random morphologies, no trained controllers)
    # ------------------------------------------------------------------
    predators = [sample_robot() for _ in range(P)]
    prey = [sample_robot() for _ in range(P)]
    for ind in predators + prey:
        ind["weights"] = None

    pred_hof = HallOfFame(max_size=coevo["hof_size"])
    prey_hof = HallOfFame(max_size=coevo["hof_size"])

    # Tracking ---
    history = {
        "pred_fitness": [],
        "prey_fitness": [],
        "pred_captures": [],
        "prey_captures": [],
    }

    G = coevo["num_generations"]
    T = sim_config["sim_steps"]
    K = coevo["controller_training_steps"]
    print(f"\n  Config: P={P}, sim_steps={T}, train_steps={K}, generations={G}")
    print(f"  Per generation: 4×train ({K} grad steps each) + 4×evaluate\n")

    for gen in range(G):
        gen_start = time.time()

        # ==============================================================
        # 1. Morphology mutation (parallel hill climber)
        # ==============================================================
        pred_children = [
            mutate_individual(p, coevo["p_flip"], coevo["min_voxels"])
            for p in predators
        ]
        prey_children = [
            mutate_individual(p, coevo["p_flip"], coevo["min_voxels"])
            for p in prey
        ]

        # ==============================================================
        # 2. Controller training (gradient-based, two-phase)
        # ==============================================================
        t0 = time.time()
        # Phase A: train predators (parents then children) against current prey
        train_controllers(simulator, predators, prey, "predator", coevo)
        train_controllers(simulator, pred_children, prey, "predator", coevo)
        # Phase B: train prey (parents then children) against (now improved) predators
        train_controllers(simulator, prey, predators, "prey", coevo)
        train_controllers(simulator, prey_children, predators, "prey", coevo)
        train_time = time.time() - t0

        # ==============================================================
        # 3. Fitness evaluation (full reward, non-differentiable)
        # ==============================================================
        t0 = time.time()
        hof_prey_sample = prey_hof.sample(coevo["matchups_hof"])
        hof_pred_sample = pred_hof.sample(coevo["matchups_hof"])

        pred_parent_fit = evaluate_fitness(
            simulator, predators, prey, hof_prey_sample, "predator", coevo
        )
        pred_child_fit = evaluate_fitness(
            simulator, pred_children, prey, hof_prey_sample, "predator", coevo
        )
        prey_parent_fit = evaluate_fitness(
            simulator, prey, predators, hof_pred_sample, "prey", coevo
        )
        prey_child_fit = evaluate_fitness(
            simulator, prey_children, predators, hof_pred_sample, "prey", coevo
        )
        eval_time = time.time() - t0

        # ==============================================================
        # 4. Selection (parallel hill climber: child replaces parent if >=)
        # ==============================================================
        for i in range(P):
            if pred_child_fit[i] >= pred_parent_fit[i]:
                predators[i] = pred_children[i]
        for i in range(P):
            if prey_child_fit[i] >= prey_parent_fit[i]:
                prey[i] = prey_children[i]

        # Surviving fitness (the winner of each pair)
        pred_fit = np.array([
            max(pred_parent_fit[i], pred_child_fit[i]) for i in range(P)
        ])
        prey_fit = np.array([
            max(prey_parent_fit[i], prey_child_fit[i]) for i in range(P)
        ])

        # ==============================================================
        # 5. Hall-of-Fame update
        # ==============================================================
        pred_hof.update(predators, pred_fit, k=coevo["hof_k"])
        prey_hof.update(prey, prey_fit, k=coevo["hof_k"])

        # ==============================================================
        # Logging
        # ==============================================================
        history["pred_fitness"].append(pred_fit.copy())
        history["prey_fitness"].append(prey_fit.copy())

        gen_time = time.time() - gen_start
        eta = gen_time * (G - gen - 1)
        eta_str = f"{int(eta//60)}m{int(eta%60):02d}s" if eta > 60 else f"{eta:.0f}s"

        print(
            f"  Gen {gen:3d}/{G} | "
            f"pred={pred_fit.mean():+7.2f} (max {pred_fit.max():+.2f})  "
            f"prey={prey_fit.mean():7.2f} (max {prey_fit.max():.2f}) | "
            f"train={train_time:.1f}s  eval={eval_time:.1f}s  total={gen_time:.1f}s | "
            f"ETA {eta_str}"
        )

        # Periodic checkpoint (every 10 generations)
        if (gen + 1) % 10 == 0 or gen == G - 1:
            save_checkpoint(gen, predators, prey, pred_hof, prey_hof, history)

    return predators, prey, pred_hof, prey_hof, history


def save_checkpoint(gen, predators, prey, pred_hof, prey_hof, history):
    """Save current state to disk."""
    # Best individuals
    best_pred_idx = np.argmax(history["pred_fitness"][-1])
    best_prey_idx = np.argmax(history["prey_fitness"][-1])
    np.save("best_predator.npy", predators[best_pred_idx])
    np.save("best_prey.npy", prey[best_prey_idx])
    # Full populations
    np.save("predators.npy", predators)
    np.save("prey.npy", prey)
    # HoFs
    np.save("pred_hof.npy", pred_hof.entries)
    np.save("prey_hof.npy", prey_hof.entries)
    # Fitness history
    np.save("fitness_history.npy", history)
    print(f"\n  [checkpoint gen {gen}] saved best_predator.npy, best_prey.npy, fitness_history.npy")


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    predators, prey, pred_hof, prey_hof, history = run_evolution(config)

    print("\n=== Evolution complete ===")
    final_pred = history["pred_fitness"][-1]
    final_prey = history["prey_fitness"][-1]
    print(f"Final predator fitness: mean={final_pred.mean():.2f}  max={final_pred.max():.2f}")
    print(f"Final prey fitness:     mean={final_prey.mean():.2f}  max={final_prey.max():.2f}")
