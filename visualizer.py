"""Real-time visualisation of a predator vs prey episode.

Usage:
    python visualizer.py --predator best_predator.npy --prey best_prey.npy
    python visualizer.py --input robot_0.npy   # legacy single-robot mode
"""

from flask import Flask, render_template, Response
from argparse import ArgumentParser
from simulator import Simulator
from utils import load_config
from robot import MAX_N_MASSES, MAX_N_SPRINGS
import threading, time, json, numpy as np

app = Flask(
    __name__,
    template_folder="visualizer/templates",
    static_folder="visualizer/static",
)

TARGET_FPS = 60.0

state_lock = threading.Lock()
app_state = {
    "step_index": 0,
    "actual_fps": 0.0,
}


@app.route("/")
def index():
    return render_template("index.html")


def step_once():
    """Execute one simulation step and return data for all robots."""
    global simulator, max_steps, robot_meta

    t = app_state["step_index"]

    if t >= max_steps:
        simulator.reinitialize_robots()
        app_state["step_index"] = 0
        t = 0

    simulator.compute_com(t)
    simulator.nn1(t)
    simulator.nn2(t)
    simulator.apply_spring_force(t)
    simulator.advance(t + 1)

    # Gather per-robot data
    all_positions = []
    all_activations = []
    all_com = []
    for meta in robot_meta:
        slot = meta["slot"]
        nm = meta["n_masses"]
        ns = meta["n_springs"]
        pos = simulator.x.to_numpy()[slot, t + 1, :nm]
        act = simulator.act.to_numpy()[slot, t, :ns]
        com = pos.mean(axis=0)
        all_positions.append(pos)
        all_activations.append(act)
        all_com.append(com)

    app_state["step_index"] = t + 1
    return all_positions, all_activations, all_com


@app.route("/stream")
def stream():
    """Server-sent events stream for real-time visualization."""

    def event_stream():
        # Send topology for each robot
        topology = {
            "type": "topology",
            "robots": [],
        }
        for meta in robot_meta:
            topology["robots"].append({
                "role": meta["role"],
                "springs": meta["springs"],
                "n_masses": meta["n_masses"],
                "n_springs": meta["n_springs"],
            })
        yield f"data: {json.dumps(topology)}\n\n"

        fps_samples = []
        last_fps_update = time.perf_counter()

        while True:
            frame_start = time.perf_counter()
            target_interval = 1.0 / TARGET_FPS

            all_positions, all_activations, all_com = step_once()

            payload = {
                "type": "step",
                "robots": [],
                "step": app_state["step_index"],
                "fps": app_state["actual_fps"],
            }
            for idx, meta in enumerate(robot_meta):
                payload["robots"].append({
                    "positions": all_positions[idx].tolist(),
                    "activations": all_activations[idx].tolist(),
                    "center_of_mass": all_com[idx].tolist(),
                    "role": meta["role"],
                })

            yield f"data: {json.dumps(payload)}\n\n"

            frame_end = time.perf_counter()
            work_time = frame_end - frame_start
            sleep_time = target_interval - work_time
            if sleep_time > 0.001:
                time.sleep(sleep_time)

            total_frame_time = time.perf_counter() - frame_start
            if total_frame_time > 0:
                fps_samples.append(1.0 / total_frame_time)

            current_time = time.perf_counter()
            if current_time - last_fps_update >= 0.5:
                if fps_samples:
                    with state_lock:
                        app_state["actual_fps"] = sum(fps_samples) / len(fps_samples)
                    fps_samples = []
                    last_fps_update = current_time

    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--predator", type=str, default=None, help="Path to saved predator .npy"
    )
    parser.add_argument(
        "--prey", type=str, default=None, help="Path to saved prey .npy"
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Legacy single-robot .npy"
    )
    parser.add_argument("--gap", type=float, default=1.5, help="Initial separation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    sim_config = config["simulator"].copy()

    # ------------------------------------------------------------------
    # Determine mode: predator/prey pair  OR  single robot (legacy)
    # ------------------------------------------------------------------
    robot_meta = []  # list of dicts with slot, role, n_masses, n_springs, springs

    if args.predator and args.prey:
        # ---- Two-robot (predator vs prey) mode ----
        pred = np.load(args.predator, allow_pickle=True).item()
        prey_r = np.load(args.prey, allow_pickle=True).item()
        print(f"Predator: {pred['n_masses']} masses, {pred['n_springs']} springs")
        print(f"Prey:     {prey_r['n_masses']} masses, {prey_r['n_springs']} springs")

        sim_config["n_sims"] = 2
        sim_config["n_masses"] = MAX_N_MASSES
        sim_config["n_springs"] = MAX_N_SPRINGS

        simulator = Simulator(sim_config, config["taichi"], config["seed"], needs_grad=False)

        gh = simulator.ground_height[None]
        gap = args.gap

        pred_m = pred["masses"].copy()
        pred_m[:, 0] -= pred_m[:, 0].mean()
        pred_m[:, 0] -= gap / 2.0
        pred_m[:, 1] -= pred_m[:, 1].min()
        pred_m[:, 1] += gh

        prey_m = prey_r["masses"].copy()
        prey_m[:, 0] -= prey_m[:, 0].mean()
        prey_m[:, 0] += gap / 2.0
        prey_m[:, 1] -= prey_m[:, 1].min()
        prey_m[:, 1] += gh

        simulator.initialize(
            [pred_m, prey_m],
            [pred["springs"], prey_r["springs"]],
            init_weights=True,
            preprocess=False,
        )

        # Opponent mapping
        simulator.opponent_idx[0] = 1
        simulator.opponent_idx[1] = 0
        simulator.role_sign[0] = 1.0
        simulator.role_sign[1] = -1.0

        # Load weights
        if pred.get("weights"):
            simulator.set_control_params([0], [pred["weights"]])
        elif pred.get("control_params"):
            simulator.set_control_params([0], [pred["control_params"]])

        if prey_r.get("weights"):
            simulator.set_control_params([1], [prey_r["weights"]])
        elif prey_r.get("control_params"):
            simulator.set_control_params([1], [prey_r["control_params"]])

        robot_meta = [
            {
                "slot": 0,
                "role": "predator",
                "n_masses": int(pred["n_masses"]),
                "n_springs": int(pred["n_springs"]),
                "springs": pred["springs"].tolist(),
            },
            {
                "slot": 1,
                "role": "prey",
                "n_masses": int(prey_r["n_masses"]),
                "n_springs": int(prey_r["n_springs"]),
                "springs": prey_r["springs"].tolist(),
            },
        ]

    else:
        # ---- Legacy single-robot mode ----
        input_path = args.input or "robot_0.npy"
        print(f"Loading robot from {input_path}...")
        robot = np.load(input_path, allow_pickle=True).item()
        print(f"Robot: {robot['n_masses']} masses, {robot['n_springs']} springs")

        if "max_n_masses" in robot and "max_n_springs" in robot:
            sim_config["n_masses"] = int(robot["max_n_masses"])
            sim_config["n_springs"] = int(robot["max_n_springs"])
        else:
            sim_config["n_masses"] = MAX_N_MASSES
            sim_config["n_springs"] = MAX_N_SPRINGS
        sim_config["n_sims"] = 1

        simulator = Simulator(sim_config, config["taichi"], config["seed"], needs_grad=False)
        simulator.initialize([robot["masses"]], [robot["springs"]])

        if "control_params" in robot:
            simulator.set_control_params([0], [robot["control_params"]])
        elif "weights" in robot and robot["weights"] is not None:
            simulator.set_control_params([0], [robot["weights"]])

        robot_meta = [
            {
                "slot": 0,
                "role": "single",
                "n_masses": int(robot["n_masses"]),
                "n_springs": int(robot["n_springs"]),
                "springs": robot["springs"].tolist(),
            },
        ]

    max_steps = simulator.steps[None]
    print(f"\nVisualizer running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug, threaded=False, use_reloader=False)
