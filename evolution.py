"""Predator / prey coevolution via parallel hill climber.

Orchestrates: morphology mutation, controller training, fitness
evaluation, selection, and Hall-of-Fame management.
"""

import time
import numpy as np
from copy import deepcopy
from robot import sample_robot, mutate_mask, robot_from_mask


# ======================================================================
# Hall of Fame
# ======================================================================
class HallOfFame:
    """Stores elite individuals from past generations."""

    def __init__(self, max_size=50):
        self.entries: list[dict] = []
        self.max_size = max_size

    def update(self, population, fitnesses, k=2):
        """Add top-k individuals (by fitness) from the current generation."""
        top_k_idx = np.argsort(fitnesses)[-k:]
        for idx in top_k_idx:
            self.entries.append(deepcopy(population[idx]))
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size :]

    def sample(self, n):
        """Uniformly sample n individuals from the HoF."""
        if len(self.entries) == 0:
            return []
        n = min(n, len(self.entries))
        indices = np.random.choice(len(self.entries), size=n, replace=False)
        return [self.entries[i] for i in indices]


# ======================================================================
# Individual helpers
# ======================================================================
def new_individual(mask=None):
    """Create an individual dict with random or given morphology.
    weights starts as None (random-initialised on first training).
    """
    robot = sample_robot() if mask is None else robot_from_mask(mask)
    robot["weights"] = None
    return robot


def mutate_individual(parent, p_flip=0.05, min_voxels=3):
    """Create a child by mutating the parent's voxel mask.
    The child inherits the parent's controller weights.
    """
    child_mask = mutate_mask(parent["mask"], p_flip=p_flip, min_voxels=min_voxels)
    child = robot_from_mask(child_mask)
    child["weights"] = deepcopy(parent["weights"])
    return child


# ======================================================================
# Simulator ↔ individual weight transfer
# ======================================================================
def load_weights_into_sim(simulator, individuals, slots):
    """Copy saved NN weights from individual dicts into simulator slots.
    Individuals with weights=None are skipped (they keep whatever the
    simulator already has — typically random init from initialize_weights).
    """
    w1 = simulator.weights1.to_numpy()
    w2 = simulator.weights2.to_numpy()
    b1 = simulator.biases1.to_numpy()
    b2 = simulator.biases2.to_numpy()
    changed = False
    for ind, slot in zip(individuals, slots):
        if ind["weights"] is not None:
            w = ind["weights"]
            # Handle shape mismatch (e.g. old weights without opponent features)
            w1_expected = w1.shape[1]
            iw1 = w["weights1"]
            if iw1.shape[0] < w1_expected:
                pad = np.zeros((w1_expected - iw1.shape[0], iw1.shape[1]), dtype=np.float32)
                iw1 = np.vstack([iw1, pad])
            w1[slot] = iw1
            w2[slot] = w["weights2"]
            b1[slot] = w["biases1"]
            b2[slot] = w["biases2"]
            changed = True
    if changed:
        simulator.weights1.from_numpy(w1)
        simulator.weights2.from_numpy(w2)
        simulator.biases1.from_numpy(b1)
        simulator.biases2.from_numpy(b2)


def extract_weights_from_sim(simulator, individuals, slots):
    """Copy NN weights out of the simulator into individual dicts."""
    w1 = simulator.weights1.to_numpy()
    w2 = simulator.weights2.to_numpy()
    b1 = simulator.biases1.to_numpy()
    b2 = simulator.biases2.to_numpy()
    for ind, slot in zip(individuals, slots):
        ind["weights"] = {
            "weights1": w1[slot].copy(),
            "weights2": w2[slot].copy(),
            "biases1": b1[slot].copy(),
            "biases2": b2[slot].copy(),
        }


# ======================================================================
# Episode batch setup
# ======================================================================
def setup_episode_batch(simulator, predators, prey_list, gaps):
    """Load P predator-prey pairs into the simulator.

    Predators go into slots 0..P-1,  prey into P..2P-1.
    Masses are pre-processed with the given horizontal gap.
    """
    P = len(predators)
    assert len(prey_list) == P and len(gaps) == P

    gh = simulator.ground_height[None]
    all_masses = []
    all_springs = []

    # Predators (left side)
    for i in range(P):
        m = predators[i]["masses"].copy()
        m[:, 0] -= m[:, 0].mean()
        m[:, 0] -= gaps[i] / 2.0
        m[:, 1] -= m[:, 1].min()
        m[:, 1] += gh
        all_masses.append(m)
        all_springs.append(predators[i]["springs"].copy())

    # Prey (right side)
    for i in range(P):
        m = prey_list[i]["masses"].copy()
        m[:, 0] -= m[:, 0].mean()
        m[:, 0] += gaps[i] / 2.0
        m[:, 1] -= m[:, 1].min()
        m[:, 1] += gh
        all_masses.append(m)
        all_springs.append(prey_list[i]["springs"].copy())

    # Initialize geometry (calls hard_reset, NO weight init, NO preprocessing)
    simulator.initialize(all_masses, all_springs, init_weights=False, preprocess=False)

    # Opponent mapping & role signs (always: pred=+1, prey=-1)
    opp = np.zeros(2 * P, dtype=np.int32)
    role = np.zeros(2 * P, dtype=np.float32)
    for i in range(P):
        opp[i] = P + i
        opp[P + i] = i
        role[i] = 1.0
        role[P + i] = -1.0
    simulator.opponent_idx.from_numpy(opp)
    simulator.role_sign.from_numpy(role)


# ======================================================================
# Controller training (gradient-based)
# ======================================================================
def train_controllers(simulator, individuals, opponents, role, config):
    """Train NN controllers for *individuals* against frozen *opponents*.

    Parameters
    ----------
    role : "predator" or "prey"
        Which side the *individuals* play.
    """
    P = len(individuals)

    # Sample P opponents (with replacement if needed)
    n_opp = len(opponents)
    if n_opp == 0:
        return  # nothing to train against
    opp_idx = np.random.choice(n_opp, size=P, replace=(n_opp < P))
    sampled_opp = [opponents[i] for i in opp_idx]

    d0 = np.random.uniform(config["d_min"], config["d_max"])
    gaps = [d0] * P

    # Place individuals on the correct side
    if role == "predator":
        setup_episode_batch(simulator, individuals, sampled_opp, gaps)
    else:
        setup_episode_batch(simulator, sampled_opp, individuals, gaps)

    # Random-init weights were wiped by hard_reset inside initialize.
    # Now re-init random then overwrite with stored weights.
    simulator.initialize_weights()

    pred_slots = list(range(P))
    prey_slots = list(range(P, 2 * P))
    if role == "predator":
        load_weights_into_sim(simulator, individuals, pred_slots)
        load_weights_into_sim(simulator, sampled_opp, prey_slots)
    else:
        load_weights_into_sim(simulator, sampled_opp, pred_slots)
        load_weights_into_sim(simulator, individuals, prey_slots)

    # Training mask: 1 = train, 0 = frozen
    mask = np.zeros(2 * P, dtype=np.int32)
    if role == "predator":
        mask[:P] = 1
    else:
        mask[P:] = 1
    simulator.training_mask.from_numpy(mask)

    # Reset Adam state for a fresh optimiser run
    simulator.reset_adam_state()

    # Gradient training loop
    K = config["controller_training_steps"]
    for _ in range(K):
        simulator.clear_grads()
        simulator.reinitialize_robots()
        simulator.forward()
        simulator.compute_loss()

        # Seed loss grad only for the population being trained
        lg = np.zeros(2 * P, dtype=np.float32)
        if role == "predator":
            lg[:P] = 1.0
        else:
            lg[P:] = 1.0
        simulator.loss.grad.from_numpy(lg)

        simulator.backward()
        simulator.adam_step[None] += 1
        simulator.update_weights()

    # Save trained weights back into the individual dicts
    if role == "predator":
        extract_weights_from_sim(simulator, individuals, pred_slots)
    else:
        extract_weights_from_sim(simulator, individuals, prey_slots)


# ======================================================================
# Fitness evaluation (non-differentiable, full reward in NumPy)
# ======================================================================
def detect_capture(d, r_capture, k_consecutive):
    """Return (captured: bool, t_cap: int) from a distance array."""
    consec = 0
    for t in range(len(d)):
        if d[t] <= r_capture:
            consec += 1
            if consec >= k_consecutive:
                return True, t - k_consecutive + 1
        else:
            consec = 0
    return False, len(d) - 1


def compute_effort_numpy(target_L, actual_L, n_springs):
    """Effort = Σ(ΔL̂)² + 0.1·Σ(L − L̂)²  over all springs and timesteps."""
    tl = target_L[:, :n_springs]   # (T, n_springs)
    al = actual_L[:, :n_springs]
    delta = np.diff(tl, axis=0)    # (T-1, n_springs)
    return float(np.sum(delta ** 2) + 0.1 * np.sum((al - tl) ** 2))


def episode_reward_predator(d, T, r_capture, k_consecutive, effort, lam):
    """Predator reward: incentivise approaching and capturing prey.

    - With capture:   100 + time_bonus  (huge reward)
    - Without capture: up to ~15  based on how much gap was closed
    """
    captured, t_cap = detect_capture(d, r_capture, k_consecutive)
    if captured:
        return 100.0 + float(T - t_cap) - lam * effort

    d0 = max(float(d[0]), 0.01)
    # Fraction of initial gap that was closed (0 = no progress, 1 = fully closed)
    approach_frac = max(0.0, float(d[0] - d[-1])) / d0
    # Bonus for closest approach (rewards lunges even if predator bounces back)
    closest_frac = max(0.0, float(d[0] - d.min())) / d0
    return 10.0 * approach_frac + 5.0 * closest_frac - lam * effort


def episode_reward_prey(d, T, r_capture, k_consecutive, d_threat, effort, lam):
    """Prey reward: incentivise fleeing and maintaining separation.

    - With capture:    0-10  proportional to survival time
    - Without capture: 10+   with bonuses for gaining separation
    """
    captured, t_cap = detect_capture(d, r_capture, k_consecutive)
    if captured:
        return 10.0 * float(t_cap) / float(max(T, 1)) - lam * effort

    d0 = max(float(d[0]), 0.01)
    # Bonus for increasing separation (ran away successfully)
    sep_gain = max(0.0, float(d[-1] - d[0])) / d0
    # Bonus for maintaining safety (how close did predator get as fraction of start)
    min_safety = float(d.min()) / d0  # 1.0 = predator never got closer, <1 = it got closer
    return 10.0 + 5.0 * sep_gain + 5.0 * min_safety - lam * effort


def evaluate_fitness(simulator, individuals, opponents, hof_opponents, role, config):
    """Run matchup episodes and return an array of fitness values.

    Each individual is tested against a set of opponents from the
    current generation + HoF, each at multiple random initial gaps.
    Fitness = mean reward over all episodes.
    """
    P = len(individuals)
    T = simulator.steps[None]
    r_cap = config["r_capture"]
    k_con = config["k_consecutive"]
    d_thr = config["d_threat_mult"] * r_cap
    lam = config["lambda_effort"]

    # Build opponent list -----------------------------------------------
    all_opponents = []
    # From current population
    n_cur = min(config["matchups_current"], len(opponents))
    if n_cur > 0:
        idx = np.random.choice(len(opponents), size=n_cur, replace=False)
        all_opponents.extend([opponents[i] for i in idx])
    # From HoF
    all_opponents.extend(hof_opponents)
    # Fallback: if nothing available, use a random current opponent
    if len(all_opponents) == 0 and len(opponents) > 0:
        all_opponents.append(opponents[np.random.randint(len(opponents))])
    if len(all_opponents) == 0:
        return np.zeros(P)

    total_reward = np.zeros(P)
    n_episodes = 0

    for opp in all_opponents:
        for _ in range(config["n_gaps"]):
            d0 = np.random.uniform(config["d_min"], config["d_max"])
            gaps = [d0] * P
            opp_batch = [opp] * P  # same opponent for all P individuals

            # Set up episode
            if role == "predator":
                setup_episode_batch(simulator, individuals, opp_batch, gaps)
                simulator.initialize_weights()       # fresh random (overwritten below)
                load_weights_into_sim(simulator, individuals, list(range(P)))
                load_weights_into_sim(simulator, opp_batch, list(range(P, 2 * P)))
            else:
                setup_episode_batch(simulator, opp_batch, individuals, gaps)
                simulator.initialize_weights()
                load_weights_into_sim(simulator, opp_batch, list(range(P)))
                load_weights_into_sim(simulator, individuals, list(range(P, 2 * P)))

            # Forward only — no grad
            simulator.reinitialize_robots()
            simulator.forward()

            # Extract data
            centers = simulator.center.to_numpy()         # (2P, T+1, 2)
            tL = simulator.target_L.to_numpy()            # (2P, T, max_springs)
            aL = simulator.actual_L.to_numpy()             # (2P, T, max_springs)
            ns = simulator.n_springs.to_numpy()            # (2P,)

            for i in range(P):
                if role == "predator":
                    pred_slot, prey_slot = i, P + i
                    ind_slot = pred_slot
                else:
                    pred_slot, prey_slot = i, P + i
                    ind_slot = prey_slot

                # Euclidean distance over time
                diff = centers[pred_slot] - centers[prey_slot]     # (T+1, 2)
                d = np.sqrt(np.sum(diff ** 2, axis=1))             # (T+1,)

                effort = compute_effort_numpy(tL[ind_slot], aL[ind_slot], ns[ind_slot])

                if role == "predator":
                    r = episode_reward_predator(d, T, r_cap, k_con, effort, lam)
                else:
                    r = episode_reward_prey(d, T, r_cap, k_con, d_thr, effort, lam)

                total_reward[i] += r

            n_episodes += 1

    return total_reward / max(n_episodes, 1)
