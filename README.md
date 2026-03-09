## Overview

This repository implements a **predator/prey coevolution** system for studying the automatic design of soft robots (virtual creatures). Two populations — predators and prey — evolve simultaneously in a physics simulation, developing both morphology (body shape) and locomotion controllers through an evolutionary arms race.

The simulation is built on [Taichi](https://taichi-lang.org/) for differentiable physics, enabling gradient-based controller training alongside evolutionary morphology search. The framework is loosely inspired by [Evolution and learning in differentiable robots](https://sites.google.com/view/eldir).

---

## How It Works

### Representation

Each organism is a **mass-spring soft body** defined by a binary voxel mask on an 8×8 grid. The mask is the genome — it determines which voxels are filled, and the geometry (masses and springs) is derived from the mask at runtime. Each voxel contributes four corner masses and up to six springs (four edges + two diagonals). A fully filled grid has at most 81 masses and 272 springs.

### Neural Controller

Each organism is controlled by a two-layer neural network that runs at every simulation timestep. The input features are:

- **Mass kinematics**: velocity and offset-from-CoM for each mass (4 values per mass)
- **CPG oscillators**: sinusoidal signals at evenly-spaced phase offsets (rhythmic locomotion priors)
- **Opponent features**: dx, dy from the organism's centre-of-mass to its opponent's CoM (enables pursuit/evasion)

The network outputs a spring activation for each spring, which modulates the spring's rest length (±10%), causing the body to deform and move.

### Physics

The simulator integrates position and velocity using semi-implicit Euler integration with:
- Ground contact (restitution + friction)
- Drag damping
- Velocity clamping to prevent instability

All physics kernels are implemented in Taichi for parallelism and autodifferentiability.

### Evolution Loop

Each generation of the **parallel hill climber**:

1. **Mutate**: Each individual's voxel mask is randomly bit-flipped (`p_flip` per cell). The mutant inherits the parent's controller weights. The largest connected component of the mask is kept; mutations producing fewer than `min_voxels` voxels are rejected.

2. **Train controllers** (gradient-based): Controllers are trained with Adam against frozen opponents. Predators train for `pred_training_mult × controller_training_steps` gradient steps (more steps because pursuit is harder than evasion). Training uses a differentiable loss:
   - **Predator loss**: minimise mean distance to opponent + effort penalty
   - **Prey loss**: maximise mean distance from opponent + effort penalty

3. **Evaluate fitness** (non-differentiable): Each individual is tested against a sample of current-generation opponents and Hall-of-Fame opponents. Fitness is the mean reward across all matchups:
   - **Predator reward**: large bonus for capture (100 + time bonus), partial reward for approach fraction and closest approach otherwise
   - **Prey reward**: proportional to survival time if captured, bonus for separation gain and safety margin if not

4. **Select**: Child replaces parent if its fitness is ≥ parent's.

5. **Hall of Fame**: Top-k individuals per generation are added to the HoF (up to `hof_size`), which provides persistent historical opponents to prevent cycling.

### Distance Curriculum

To ease early learning, the initial predator–prey separation ramps from `d_max_start` (easy) to `d_max` (full challenge) over `curriculum_gens` generations.

---

## Installation

1. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) if you don't already have it.
2. Create and activate a new environment:
   ```
   conda create --name alife-sim
   conda activate alife-sim
   ```
3. Install Python:
   ```
   conda install python=3.12
   ```
4. Install Taichi:
   ```
   pip install taichi==1.7.3
   ```
5. Install other dependencies:
   ```
   pip install tqdm scipy pyaml flask ipykernel matplotlib
   ```

---

## Usage

### Running Evolution

```bash
python run.py --config config.yaml
```

Use `config_overnight.yaml` for a longer run (~9 hours on CPU) with more generations and training steps:

```bash
python run.py --config config_overnight.yaml
```

Key config parameters to consider adjusting:

| Parameter | Description |
|---|---|
| `population_size` | Number of predators and prey (use 4 for CPU, 8+ for CUDA) |
| `num_generations` | Total generations to run |
| `controller_training_steps` | Gradient steps per training phase |
| `sim_steps` | Timesteps per episode |
| `taichi.arch` | `cpu` or `cuda` |

### Checkpoints

The run saves state every 10 generations (and on completion) to:

- `best_predator.npy` / `best_prey.npy` — all-time best individuals
- `latest_predator.npy` / `latest_prey.npy` — current-generation best
- `predators.npy` / `prey.npy` — full current populations
- `pred_hof.npy` / `prey_hof.npy` — Hall of Fame entries
- `fitness_history.npy` — per-generation fitness arrays

### Visualizing a Matchup

Run the web-based visualizer against saved individuals:

```bash
python visualizer.py --predator best_predator.npy --prey best_prey.npy
```

Then open `http://localhost:5000` in a browser. The visualizer renders the mass-spring bodies in real time at up to 60 fps, colour-coded by spring activation. Use `--gap` to set the initial separation distance and `--port` to change the port.

### Plotting Fitness History

Open `plot_fitness.ipynb` in Jupyter to plot predator and prey fitness over generations from `fitness_history.npy`.

### Visualizing Robot Morphologies

Open `visualize_robots.ipynb` to inspect the voxel masks and mass-spring geometry of saved robots.

---

## File Structure

| File | Purpose |
|---|---|
| `run.py` | Main evolution loop |
| `evolution.py` | Mutation, controller training, fitness evaluation, Hall of Fame |
| `simulator.py` | Taichi physics simulator and neural network kernels |
| `robot.py` | Voxel mask representation and mass-spring geometry generation |
| `visualizer.py` | Flask-based real-time web visualizer |
| `utils.py` | Config loading utilities |
| `config.yaml` | Default config (small, fast, good for testing) |
| `config_overnight.yaml` | Longer run config (~9 hour CPU run) |
| `plot_fitness.ipynb` | Fitness history plotting notebook |
| `visualize_robots.ipynb` | Robot morphology visualization notebook |
