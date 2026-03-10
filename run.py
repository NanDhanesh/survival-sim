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

    # All-time best tracking
    best_pred_ever = None
    best_pred_fit_ever = -np.inf
    best_prey_ever = None
    best_prey_fit_ever = -np.inf

    G = coevo["num_generations"]
    T = sim_config["sim_steps"]
    K = coevo["controller_training_steps"]
    pred_mult = coevo.get("pred_training_mult", 1.0)
    K_pred = int(K * pred_mult)
    K_prey = K

    # Distance curriculum: d_max ramps from d_max_start to d_max over curriculum_gens
    d_max_final = coevo["d_max"]
    d_max_start = coevo.get("d_max_start", d_max_final)  # no curriculum if absent
    curriculum_gens = coevo.get("curriculum_gens", 0)

    # Save "before evolution" baseline: random morphology with trained controller.
    # Train deepcopies of the initial population against each other so the
    # main population is unaffected.
    prevo_preds = [deepcopy(p) for p in predators]
    prevo_prey  = [deepcopy(p) for p in prey]
    prevo_config_pred = {**coevo, "d_max": d_max_start, "controller_training_steps": K_pred}
    prevo_config_prey = {**coevo, "d_max": d_max_start, "controller_training_steps": K_prey}
    print("\n  Training controllers for pre-evolution baseline...")
    train_controllers(simulator, prevo_preds, prevo_prey, "predator", prevo_config_pred)
    train_controllers(simulator, prevo_prey,  prevo_preds, "prey",     prevo_config_prey)
    np.save("prevo_predator.npy", prevo_preds[0])
    np.save("prevo_prey.npy",     prevo_prey[0])
    print("  Saved prevo_predator.npy and prevo_prey.npy\n")

    print(f"\n  Config: P={P}, sim_steps={T}, train_steps=pred:{K_pred}/prey:{K_prey}, generations={G}")
    if curriculum_gens > 0:
        print(f"  Distance curriculum: d_max {d_max_start:.2f} → {d_max_final:.2f} over {curriculum_gens} gens")
    print(f"  Per generation: 4×train + 4×evaluate\n")

    for gen in range(G):
        gen_start = time.time()

        # -- Distance curriculum: compute current d_max for this generation --
        if curriculum_gens > 0 and gen < curriculum_gens:
            frac = gen / max(curriculum_gens - 1, 1)
            current_d_max = d_max_start + frac * (d_max_final - d_max_start)
        else:
            current_d_max = d_max_final

        # Build per-generation config overrides
        gen_config_pred = {**coevo, "d_max": current_d_max,
                          "controller_training_steps": K_pred}
        gen_config_prey = {**coevo, "d_max": current_d_max,
                          "controller_training_steps": K_prey}
        gen_config_eval = {**coevo, "d_max": current_d_max}

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
        #    Predators get more steps (pred_training_mult) since their
        #    task is harder (must locomote AND steer toward prey).
        # ==============================================================
        t0 = time.time()
        # Phase A: train predators (parents then children) against current prey
        train_controllers(simulator, predators, prey, "predator", gen_config_pred)
        train_controllers(simulator, pred_children, prey, "predator", gen_config_pred)
        # Phase B: train prey (parents then children) against (now improved) predators
        train_controllers(simulator, prey, predators, "prey", gen_config_prey)
        train_controllers(simulator, prey_children, predators, "prey", gen_config_prey)
        train_time = time.time() - t0

        # ==============================================================
        # 3. Fitness evaluation (full reward, non-differentiable)
        # ==============================================================
        t0 = time.time()
        hof_prey_sample = prey_hof.sample(coevo["matchups_hof"])
        hof_pred_sample = pred_hof.sample(coevo["matchups_hof"])

        pred_parent_fit = evaluate_fitness(
            simulator, predators, prey, hof_prey_sample, "predator", gen_config_eval
        )
        pred_child_fit = evaluate_fitness(
            simulator, pred_children, prey, hof_prey_sample, "predator", gen_config_eval
        )
        prey_parent_fit = evaluate_fitness(
            simulator, prey, predators, hof_pred_sample, "prey", gen_config_eval
        )
        prey_child_fit = evaluate_fitness(
            simulator, prey_children, predators, hof_pred_sample, "prey", gen_config_eval
        )
        eval_time = time.time() - t0

        # ==============================================================
        # 4. Selection (parallel hill climber: child replaces parent if >=)
        #    NaN fitness is treated as -inf — never selected.
        # ==============================================================
        for i in range(P):
            cf = pred_child_fit[i] if np.isfinite(pred_child_fit[i]) else -np.inf
            pf = pred_parent_fit[i] if np.isfinite(pred_parent_fit[i]) else -np.inf
            if cf >= pf:
                predators[i] = pred_children[i]
        for i in range(P):
            cf = prey_child_fit[i] if np.isfinite(prey_child_fit[i]) else -np.inf
            pf = prey_parent_fit[i] if np.isfinite(prey_parent_fit[i]) else -np.inf
            if cf >= pf:
                prey[i] = prey_children[i]

        # Surviving fitness (the winner of each pair, NaN-safe)
        pred_fit = np.array([
            max(pred_parent_fit[i], pred_child_fit[i])
            if np.isfinite(pred_parent_fit[i]) and np.isfinite(pred_child_fit[i])
            else (pred_parent_fit[i] if np.isfinite(pred_parent_fit[i]) else pred_child_fit[i])
            for i in range(P)
        ])
        prey_fit = np.array([
            max(prey_parent_fit[i], prey_child_fit[i])
            if np.isfinite(prey_parent_fit[i]) and np.isfinite(prey_child_fit[i])
            else (prey_parent_fit[i] if np.isfinite(prey_parent_fit[i]) else prey_child_fit[i])
            for i in range(P)
        ])

        # ==============================================================
        # 5. Hall-of-Fame update
        # ==============================================================
        pred_hof.update(predators, pred_fit, k=coevo["hof_k"])
        prey_hof.update(prey, prey_fit, k=coevo["hof_k"])

        # ==============================================================
        # 5b. Track all-time best individuals
        # ==============================================================
        gen_best_pred_idx = np.argmax(pred_fit)
        if np.isfinite(pred_fit[gen_best_pred_idx]) and pred_fit[gen_best_pred_idx] > best_pred_fit_ever:
            best_pred_fit_ever = pred_fit[gen_best_pred_idx]
            best_pred_ever = deepcopy(predators[gen_best_pred_idx])
            np.save("best_predator.npy", best_pred_ever)
            print(f"    ★ New all-time best predator: {best_pred_fit_ever:+.2f} (gen {gen})")

        gen_best_prey_idx = np.argmax(prey_fit)
        if np.isfinite(prey_fit[gen_best_prey_idx]) and prey_fit[gen_best_prey_idx] > best_prey_fit_ever:
            best_prey_fit_ever = prey_fit[gen_best_prey_idx]
            best_prey_ever = deepcopy(prey[gen_best_prey_idx])
            np.save("best_prey.npy", best_prey_ever)
            print(f"    ★ New all-time best prey: {best_prey_fit_ever:.2f} (gen {gen})")

        # ==============================================================
        # Logging
        # ==============================================================
        history["pred_fitness"].append(pred_fit.copy())
        history["prey_fitness"].append(prey_fit.copy())

        gen_time = time.time() - gen_start
        eta = gen_time * (G - gen - 1)
        eta_str = f"{int(eta//60)}m{int(eta%60):02d}s" if eta > 60 else f"{eta:.0f}s"

        d_max_tag = f"  d_max={current_d_max:.2f}" if curriculum_gens > 0 else ""
        print(
            f"  Gen {gen:3d}/{G} | "
            f"pred={pred_fit.mean():+7.2f} (max {pred_fit.max():+.2f})  "
            f"prey={prey_fit.mean():7.2f} (max {prey_fit.max():.2f}) | "
            f"train={train_time:.1f}s  eval={eval_time:.1f}s  total={gen_time:.1f}s | "
            f"ETA {eta_str}{d_max_tag}"
        )

        # Periodic checkpoint (every 10 generations)
        if (gen + 1) % 10 == 0 or gen == G - 1:
            save_checkpoint(gen, predators, prey, pred_hof, prey_hof, history,
                            best_pred_ever, best_prey_ever,
                            best_pred_fit_ever, best_prey_fit_ever)

    return predators, prey, pred_hof, prey_hof, history


def save_checkpoint(gen, predators, prey, pred_hof, prey_hof, history,
                    best_pred_ever, best_prey_ever,
                    best_pred_fit_ever, best_prey_fit_ever):
    """Save current state to disk."""
    # All-time best individuals (already saved on discovery, but re-save for safety)
    if best_pred_ever is not None:
        np.save("best_predator.npy", best_pred_ever)
    if best_prey_ever is not None:
        np.save("best_prey.npy", best_prey_ever)
    # Current-gen best (useful for inspecting latest population)
    best_pred_idx = np.argmax(history["pred_fitness"][-1])
    best_prey_idx = np.argmax(history["prey_fitness"][-1])
    np.save("latest_predator.npy", predators[best_pred_idx])
    np.save("latest_prey.npy", prey[best_prey_idx])
    # "Before learning" baseline: evolved morphology with weights stripped
    # so the visualizer uses a random-init controller
    nolearn_pred = deepcopy(predators[best_pred_idx])
    nolearn_pred["weights"] = None
    nolearn_prey = deepcopy(prey[best_prey_idx])
    nolearn_prey["weights"] = None
    np.save("nolearn_predator.npy", nolearn_pred)
    np.save("nolearn_prey.npy", nolearn_prey)
    # Full populations
    np.save("predators.npy", predators)
    np.save("prey.npy", prey)
    # HoFs
    np.save("pred_hof.npy", pred_hof.entries)
    np.save("prey_hof.npy", prey_hof.entries)
    # Fitness history
    np.save("fitness_history.npy", history)
    print(f"\n  [checkpoint gen {gen}] saved (best_pred={best_pred_fit_ever:+.2f}, best_prey={best_prey_fit_ever:.2f})")


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
