import os, numpy as np
os.environ["ENABLE_TAICHI_HEADER_PRINT"] = "False"
import taichi as ti
from tqdm import tqdm

architectures = {
    "cuda": ti.cuda,
    "cpu": ti.cpu,
    "metal": ti.metal
}

vec2 = ti.types.vector(2, ti.f32)

N_OPPONENT_FEATURES = 2  # dx, dy to opponent centre-of-mass


@ti.data_oriented
class Simulator:
    def __init__(self, sim_config, taichi_config, seed, needs_grad=True):
        ti.init(
            arch=architectures[taichi_config["arch"]],
            default_fp=ti.f32,
            random_seed=seed,
            **taichi_config["init"],
        )
        self.needs_grad = needs_grad
        self.config = sim_config
        self.taichi_config = taichi_config
        self.set_constants()
        self.allocate_fields()

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    def set_constants(self):
        self.n_sims = ti.field(dtype=ti.i32, shape=(), needs_grad=False)
        self.steps = ti.field(dtype=ti.i32, shape=(), needs_grad=False)
        self.max_n_masses = ti.field(dtype=ti.i32, shape=(), needs_grad=False)
        self.max_n_springs = ti.field(dtype=ti.i32, shape=(), needs_grad=False)
        self.ground_height = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.dt = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.springA = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.springK = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.gravity = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.friction = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.restitution = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.drag_damping = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.eps = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.nn_hidden_size = ti.field(dtype=ti.i32, shape=(), needs_grad=False)
        self.nn_cpg_count = ti.field(dtype=ti.i32, shape=(), needs_grad=False)
        self.cpg_omega = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.adam_beta1 = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.adam_beta2 = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.learning_rate = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        # NEW: dimension of weights1 first axis
        self.w1_input_dim = ti.field(dtype=ti.i32, shape=(), needs_grad=False)
        # NEW: effort penalty weight
        self.lambda_effort = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        # Gradient clipping threshold (prevents NaN explosion)
        self.grad_clip = ti.field(dtype=ti.f32, shape=(), needs_grad=False)

        self.n_sims[None] = self.config["n_sims"]
        self.steps[None] = self.config["sim_steps"]
        self.max_n_masses[None] = self.config["n_masses"]
        self.max_n_springs[None] = self.config["n_springs"]
        self.ground_height[None] = self.config["ground_height"]
        self.dt[None] = self.config["dt"]
        self.springA[None] = self.config["springA"]
        self.springK[None] = self.config["springK"]
        self.gravity[None] = self.config["gravity"]
        self.friction[None] = self.config["friction"]
        self.restitution[None] = self.config["restitution"]
        self.drag_damping[None] = self.config["drag_damping"]
        self.eps[None] = self.config["eps"]
        self.nn_hidden_size[None] = self.config["nn_hidden_size"]
        self.nn_cpg_count[None] = self.config["nn_cpg_count"]
        self.cpg_omega[None] = self.config["cpg_omega"]
        self.adam_beta1[None] = self.config["adam_beta1"]
        self.adam_beta2[None] = self.config["adam_beta2"]
        self.learning_rate[None] = self.config["learning_rate"]
        self.w1_input_dim[None] = (self.max_n_masses[None] * 4
                                    + self.nn_cpg_count[None]
                                    + N_OPPONENT_FEATURES)
        self.lambda_effort[None] = 0.02  # default; evolution code overrides
        self.grad_clip[None] = 1.0      # gradient clipping threshold

    # ------------------------------------------------------------------
    # Field allocation
    # ------------------------------------------------------------------
    def allocate_fields(self):
        NS = self.n_sims[None]
        T = self.steps[None]
        MM = self.max_n_masses[None]
        MS = self.max_n_springs[None]
        HS = self.nn_hidden_size[None]
        W1D = self.w1_input_dim[None]

        # Geometry (static per episode)
        self.springs = ti.Vector.field(2, dtype=ti.i32, shape=(NS, MS), needs_grad=False)
        self.springL = ti.field(dtype=ti.f32, shape=(NS, MS), needs_grad=False)
        self.n_masses = ti.field(dtype=ti.i32, shape=(NS,), needs_grad=False)
        self.n_springs = ti.field(dtype=ti.i32, shape=(NS,), needs_grad=False)

        # State (per timestep)
        self.act = ti.field(dtype=ti.f32, shape=(NS, T, MS), needs_grad=self.needs_grad)
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=(NS, T + 1, MM), needs_grad=self.needs_grad)
        self.center = ti.Vector.field(2, dtype=ti.f32, shape=(NS, T + 1), needs_grad=self.needs_grad)
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=(NS, T + 1, MM), needs_grad=self.needs_grad)
        self.vinc = ti.Vector.field(2, dtype=ti.f32, shape=(NS, T + 1, MM), needs_grad=self.needs_grad)
        self.hidden = ti.field(dtype=ti.f32, shape=(NS, T, HS), needs_grad=self.needs_grad)

        # NN weights
        self.weights1 = ti.field(dtype=ti.f32, shape=(NS, W1D, HS), needs_grad=self.needs_grad)
        self.weights2 = ti.field(dtype=ti.f32, shape=(NS, HS, MS), needs_grad=self.needs_grad)
        self.biases1 = ti.field(dtype=ti.f32, shape=(NS, HS), needs_grad=self.needs_grad)
        self.biases2 = ti.field(dtype=ti.f32, shape=(NS, MS), needs_grad=self.needs_grad)

        # Adam optimiser state
        self.weights1_grad_m = ti.field(dtype=ti.f32, shape=(NS, W1D, HS), needs_grad=False)
        self.weights2_grad_m = ti.field(dtype=ti.f32, shape=(NS, HS, MS), needs_grad=False)
        self.biases1_grad_m = ti.field(dtype=ti.f32, shape=(NS, HS), needs_grad=False)
        self.biases2_grad_m = ti.field(dtype=ti.f32, shape=(NS, MS), needs_grad=False)
        self.weights1_grad_v = ti.field(dtype=ti.f32, shape=(NS, W1D, HS), needs_grad=False)
        self.weights2_grad_v = ti.field(dtype=ti.f32, shape=(NS, HS, MS), needs_grad=False)
        self.biases1_grad_v = ti.field(dtype=ti.f32, shape=(NS, HS), needs_grad=False)
        self.biases2_grad_v = ti.field(dtype=ti.f32, shape=(NS, MS), needs_grad=False)

        self.n_hidden = ti.field(dtype=ti.i32, shape=(NS,), needs_grad=False)
        self.loss = ti.field(dtype=ti.f32, shape=(NS,), needs_grad=self.needs_grad)
        self.adam_step = ti.field(dtype=ti.i32, shape=(), needs_grad=False)

        # --- NEW fields for predator/prey ---
        # Effort tracking
        self.target_L = ti.field(dtype=ti.f32, shape=(NS, T, MS), needs_grad=self.needs_grad)
        self.actual_L = ti.field(dtype=ti.f32, shape=(NS, T, MS), needs_grad=False)
        self.effort_step = ti.field(dtype=ti.f32, shape=(NS, T), needs_grad=self.needs_grad)

        # Per-sim accumulators (summed across timesteps in the Python loop,
        # NOT inside a single kernel — avoids Taichi autodiff loop issues).
        self.dist_sum = ti.field(dtype=ti.f32, shape=(NS,), needs_grad=self.needs_grad)
        self.effort_sum = ti.field(dtype=ti.f32, shape=(NS,), needs_grad=self.needs_grad)

        # Opponent / role mapping
        self.opponent_idx = ti.field(dtype=ti.i32, shape=(NS,), needs_grad=False)
        self.role_sign = ti.field(dtype=ti.f32, shape=(NS,), needs_grad=False)
        self.training_mask = ti.field(dtype=ti.i32, shape=(NS,), needs_grad=False)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def initialize(self, masses, springs, init_weights=True, preprocess=True):
        n_robots = len(masses)
        assert n_robots == self.n_sims[None], (
            "The number of robots does not match n_sims in the simulator config"
        )
        self.hard_reset()
        for i in range(n_robots):
            m = np.array(masses[i], dtype=np.float32)
            assert m.shape[0] > 0
            assert m.shape[0] <= self.max_n_masses[None]
            if preprocess:
                m[:, 0] = m[:, 0] - m[:, 0].mean()
                m[:, 1] = m[:, 1] - m[:, 1].min() + self.ground_height[None]
            s = np.array(springs[i], dtype=np.int32)
            assert s.shape[0] > 0
            assert s.shape[0] <= self.max_n_springs[None]
            self.initialize_masses(i, m)
            self.initialize_springs(i, s)
        self.count_hidden_units()
        if init_weights:
            self.initialize_weights()

    def count_hidden_units(self):
        for sim_idx in range(self.n_sims[None]):
            self.n_hidden[sim_idx] = max(
                1,
                int(self.nn_hidden_size[None] * (self.n_masses[sim_idx] / self.max_n_masses[None])),
            )

    @ti.kernel
    def initialize_masses(self, i: ti.i32, masses: ti.types.ndarray()):
        for j in range(masses.shape[0]):
            self.x[i, 0, j] = ti.Vector([masses[j, 0], masses[j, 1]], dt=ti.f32)
            self.n_masses[i] += 1

    @ti.kernel
    def initialize_springs(self, i: ti.i32, springs: ti.types.ndarray()):
        for j in range(springs.shape[0]):
            self.springs[i, j] = ti.Vector([springs[j, 0], springs[j, 1]], dt=ti.i32)
            self.springL[i, j] = ti.math.distance(
                self.x[i, 0, springs[j, 0]], self.x[i, 0, springs[j, 1]]
            )
            self.n_springs[i] += 1

    def initialize_weights(self):
        weights1_list = []
        weights2_list = []
        w1d = self.w1_input_dim[None]
        hs = self.nn_hidden_size[None]
        ms = self.max_n_springs[None]
        for i in range(self.n_sims[None]):
            fan_in1 = self.n_masses[i] * 4 + self.nn_cpg_count[None] + N_OPPONENT_FEATURES
            weights1_list.append(
                np.random.normal(0.0, np.sqrt(2.0 / max(fan_in1, 1)), (w1d, hs))
            )
            fan_in2 = max(self.n_hidden[i], 1)
            weights2_list.append(
                np.random.normal(0.0, np.sqrt(2.0 / fan_in2), (hs, ms))
            )
        self.weights1.from_numpy(np.stack(weights1_list).astype(np.float32))
        self.weights2.from_numpy(np.stack(weights2_list).astype(np.float32))
        self.biases1.from_numpy(np.zeros((self.n_sims[None], hs), dtype=np.float32))
        self.biases2.from_numpy(np.zeros((self.n_sims[None], ms), dtype=np.float32))

    # ------------------------------------------------------------------
    # Forward simulation
    # ------------------------------------------------------------------
    def forward(self):
        for t in range(0, self.steps[None]):
            self.compute_com(t)
            self.accumulate_distance(t)       # reads center[t], which was JUST computed
            self.nn1(t)
            self.nn2(t)
            self.apply_spring_force(t)
            self.advance(t + 1)
            self.accumulate_effort(t)
            if t > 0:
                self.accumulate_delta_effort(t)
        self.compute_com(self.steps[None])
        self.accumulate_distance(self.steps[None])  # include final timestep

    def backward(self):
        self.compute_loss.grad()
        # Reverse of: compute_com(T), accumulate_distance(T)
        self.accumulate_distance.grad(self.steps[None])
        self.compute_com.grad(self.steps[None])
        for t in range(self.steps[None] - 1, -1, -1):
            if t > 0:
                self.accumulate_delta_effort.grad(t)
            self.accumulate_effort.grad(t)
            self.advance.grad(t + 1)
            self.apply_spring_force.grad(t)
            self.nn2.grad(t)
            self.nn1.grad(t)
            # Reverse of: compute_com(t), accumulate_distance(t)
            self.accumulate_distance.grad(t)
            self.compute_com.grad(t)

    # ------------------------------------------------------------------
    # Kernels — neural network
    # ------------------------------------------------------------------
    @ti.kernel
    def nn1(self, t: ti.i32):
        # Mass features: vx, vy, dx_to_com, dy_to_com
        for sim_idx, mass_idx, hidden_idx in ti.ndrange(
            self.n_sims[None], self.max_n_masses[None], self.nn_hidden_size[None]
        ):
            if mass_idx < self.n_masses[sim_idx] and hidden_idx < self.n_hidden[sim_idx]:
                self.hidden[sim_idx, t, hidden_idx] += (
                    self.weights1[sim_idx, mass_idx * 4 + 0, hidden_idx]
                    * self.v[sim_idx, t, mass_idx][0] * 0.05
                )
                self.hidden[sim_idx, t, hidden_idx] += (
                    self.weights1[sim_idx, mass_idx * 4 + 1, hidden_idx]
                    * self.v[sim_idx, t, mass_idx][1] * 0.05
                )
                self.hidden[sim_idx, t, hidden_idx] += (
                    self.weights1[sim_idx, mass_idx * 4 + 2, hidden_idx]
                    * (self.center[sim_idx, t].x - self.x[sim_idx, t, mass_idx].x)
                )
                self.hidden[sim_idx, t, hidden_idx] += (
                    self.weights1[sim_idx, mass_idx * 4 + 3, hidden_idx]
                    * (self.center[sim_idx, t].y - self.x[sim_idx, t, mass_idx].y)
                )

        # CPG oscillators
        for sim_idx, cpg_idx, hidden_idx in ti.ndrange(
            self.n_sims[None], self.nn_cpg_count[None], self.nn_hidden_size[None]
        ):
            if hidden_idx < self.n_hidden[sim_idx]:
                self.hidden[sim_idx, t, hidden_idx] += (
                    self.weights1[
                        sim_idx, self.max_n_masses[None] * 4 + cpg_idx, hidden_idx
                    ]
                    * ti.math.sin(
                        t * self.dt[None] * self.cpg_omega[None]
                        + (cpg_idx * 2 * ti.math.pi / self.nn_cpg_count[None])
                    )
                )

        # Opponent-relative features (dx, dy to opponent CoM)
        for sim_idx, hidden_idx in ti.ndrange(
            self.n_sims[None], self.nn_hidden_size[None]
        ):
            if hidden_idx < self.n_hidden[sim_idx]:
                opp = self.opponent_idx[sim_idx]
                opp_dx = self.center[opp, t][0] - self.center[sim_idx, t][0]
                opp_dy = self.center[opp, t][1] - self.center[sim_idx, t][1]
                base = self.max_n_masses[None] * 4 + self.nn_cpg_count[None]
                self.hidden[sim_idx, t, hidden_idx] += (
                    self.weights1[sim_idx, base, hidden_idx] * opp_dx
                )
                self.hidden[sim_idx, t, hidden_idx] += (
                    self.weights1[sim_idx, base + 1, hidden_idx] * opp_dy
                )

        # Bias
        for sim_idx, hidden_idx in ti.ndrange(
            self.n_sims[None], self.nn_hidden_size[None]
        ):
            if hidden_idx < self.n_hidden[sim_idx]:
                self.hidden[sim_idx, t, hidden_idx] += self.biases1[sim_idx, hidden_idx]

    @ti.kernel
    def nn2(self, t: ti.i32):
        for sim_idx, hidden_idx, spring_idx in ti.ndrange(
            self.n_sims[None], self.nn_hidden_size[None], self.max_n_springs[None]
        ):
            if spring_idx < self.n_springs[sim_idx] and hidden_idx < self.n_hidden[sim_idx]:
                self.act[sim_idx, t, spring_idx] += (
                    self.weights2[sim_idx, hidden_idx, spring_idx]
                    * ti.math.tanh(self.hidden[sim_idx, t, hidden_idx])
                )
        for sim_idx, spring_idx in ti.ndrange(
            self.n_sims[None], self.max_n_springs[None]
        ):
            if spring_idx < self.n_springs[sim_idx]:
                self.act[sim_idx, t, spring_idx] += self.biases2[sim_idx, spring_idx]

    # ------------------------------------------------------------------
    # Kernels — physics
    # ------------------------------------------------------------------
    @ti.kernel
    def apply_spring_force(self, t: ti.i32):
        for sim_idx, spring_idx in ti.ndrange(
            self.n_sims[None], self.max_n_springs[None]
        ):
            if spring_idx < self.n_springs[sim_idx]:
                endpoint1 = self.springs[sim_idx, spring_idx][0]
                endpoint2 = self.springs[sim_idx, spring_idx][1]
                dist = self.x[sim_idx, t, endpoint1] - self.x[sim_idx, t, endpoint2]
                length = dist.norm()
                target_length = self.springL[sim_idx, spring_idx] * (
                    1 + ti.math.tanh(self.act[sim_idx, t, spring_idx]) * self.springA[None]
                )

                # Store for effort computation
                self.target_L[sim_idx, t, spring_idx] = target_length
                self.actual_L[sim_idx, t, spring_idx] = length

                # Accumulate strain effort for this timestep
                strain = length - target_length
                self.effort_step[sim_idx, t] += 0.1 * strain * strain

                # Spring force
                force = strain * self.springK[None] * dist / (length + self.eps[None])
                impulse = self.dt[None] * force
                self.vinc[sim_idx, t + 1, endpoint1] += -impulse
                self.vinc[sim_idx, t + 1, endpoint2] += impulse

    @ti.kernel
    def advance(self, t: ti.i32):
        for sim_idx, mass_idx in ti.ndrange(
            self.n_sims[None], self.max_n_masses[None]
        ):
            if mass_idx < self.n_masses[sim_idx]:
                damping = ti.exp(-self.dt[None] * self.drag_damping[None])
                g = self.dt[None] * ti.Vector([0.0, -self.gravity[None]])
                newv = damping * self.v[sim_idx, t - 1, mass_idx] + g + self.vinc[sim_idx, t, mass_idx]

                # --- Smooth velocity clamping (branch-free, autodiff-safe) ---
                speed = newv.norm() + self.eps[None]
                max_vel = 30.0  # generous cap; normal locomotion is << 10
                scale = ti.min(1.0, max_vel / speed)
                newv = newv * scale

                oldx = self.x[sim_idx, t - 1, mass_idx]
                newx = oldx + self.dt[None] * newv
                if newx[1] < self.ground_height[None]:
                    toi = (self.ground_height[None] - oldx[1]) / newv[1]
                    toi = ti.math.clamp(toi, 0.0, self.dt[None])
                    newx_toi = oldx + toi * newv
                    newv_contact = self.v_on_contact(newv, ti.Vector([0.0, 1.0]))
                    newx_contact = newx_toi + (self.dt[None] - toi) * newv_contact
                    newx = newx_contact
                    newv = newv_contact
                self.x[sim_idx, t, mass_idx] = newx
                self.v[sim_idx, t, mass_idx] = newv

    @ti.func
    def v_on_contact(self, v_old: vec2, normal: vec2) -> vec2:
        vn = v_old.dot(normal) * normal
        vn_mag = vn.norm()
        vt = v_old - vn
        vnew = self.restitution[None] * -vn
        vt_mag = vt.norm()
        if vt_mag > 0.0:
            friction_mag = ti.math.clamp(self.friction[None] * vn_mag, 0.0, vt_mag * 0.95)
            vf = -friction_mag * vt.normalized()
            vnew += vt + vf
        return vnew

    @ti.kernel
    def compute_com(self, t: ti.i32):
        for sim_idx, mass_idx in ti.ndrange(
            self.n_sims[None], self.max_n_masses[None]
        ):
            if mass_idx < self.n_masses[sim_idx]:
                self.center[sim_idx, t] += (
                    self.x[sim_idx, t, mass_idx]
                    / ti.cast(self.n_masses[sim_idx], ti.f32)
                )

    # ------------------------------------------------------------------
    # Effort & Loss  (per-timestep kernels avoid Taichi autodiff loop crash)
    # ------------------------------------------------------------------
    @ti.kernel
    def accumulate_distance(self, t: ti.i32):
        """Add distance-to-opponent at timestep *t* into dist_sum.

        IMPORTANT: must be called AFTER compute_com(t) so that
        center[sim_idx, t] is valid.
        """
        for sim_idx in range(self.n_sims[None]):
            opp = self.opponent_idx[sim_idx]
            dx = self.center[sim_idx, t][0] - self.center[opp, t][0]
            dy = self.center[sim_idx, t][1] - self.center[opp, t][1]
            self.dist_sum[sim_idx] += ti.sqrt(dx * dx + dy * dy + self.eps[None])

    @ti.kernel
    def accumulate_effort(self, t: ti.i32):
        """Add strain effort at timestep *t*."""
        for sim_idx in range(self.n_sims[None]):
            self.effort_sum[sim_idx] += self.effort_step[sim_idx, t]

    @ti.kernel
    def accumulate_delta_effort(self, t: ti.i32):
        """Add ΔL̂² (target-length change) effort at timestep *t*.
        Only call for t > 0."""
        for sim_idx, spring_idx in ti.ndrange(
            self.n_sims[None], self.max_n_springs[None]
        ):
            if spring_idx < self.n_springs[sim_idx]:
                delta = (
                    self.target_L[sim_idx, t, spring_idx]
                    - self.target_L[sim_idx, t - 1, spring_idx]
                )
                self.effort_sum[sim_idx] += delta * delta

    @ti.kernel
    def compute_loss(self):
        """Training loss: role_sign * avg_distance + lambda * effort.

        Predator slots (role_sign = +1) minimise distance.
        Prey slots    (role_sign = -1) maximise distance.
        Both are penalised for effort.

        dist_sum and effort_sum must already be populated by the
        accumulate_distance / accumulate_effort calls in forward().
        """
        for sim_idx in range(self.n_sims[None]):
            avg_dist = self.dist_sum[sim_idx] / ti.cast(self.steps[None] + 1, ti.f32)
            self.loss[sim_idx] = (
                self.role_sign[sim_idx] * avg_dist
                + self.lambda_effort[None] * self.effort_sum[sim_idx]
            )

    # ------------------------------------------------------------------
    # Weight update (Adam) with training mask + gradient clipping
    # ------------------------------------------------------------------
    @ti.func
    def clip_grad(self, g: ti.f32) -> ti.f32:
        return ti.math.clamp(g, -self.grad_clip[None], self.grad_clip[None])

    @ti.kernel
    def update_weights(self):
        for sim_idx, i, j in ti.ndrange(
            self.n_sims[None], self.w1_input_dim[None], self.nn_hidden_size[None]
        ):
            if self.training_mask[sim_idx] == 1:
                grad = self.clip_grad(self.weights1.grad[sim_idx, i, j])
                self.weights1_grad_m[sim_idx, i, j] = (
                    self.adam_beta1[None] * self.weights1_grad_m[sim_idx, i, j]
                    + (1.0 - self.adam_beta1[None]) * grad
                )
                self.weights1_grad_v[sim_idx, i, j] = (
                    self.adam_beta2[None] * self.weights1_grad_v[sim_idx, i, j]
                    + (1.0 - self.adam_beta2[None]) * grad * grad
                )
                m_hat = self.weights1_grad_m[sim_idx, i, j] / (
                    1.0 - ti.pow(self.adam_beta1[None], self.adam_step[None])
                )
                v_hat = self.weights1_grad_v[sim_idx, i, j] / (
                    1.0 - ti.pow(self.adam_beta2[None], self.adam_step[None])
                )
                self.weights1[sim_idx, i, j] += (
                    -self.learning_rate[None] * m_hat / (ti.sqrt(v_hat) + self.eps[None])
                )

        for sim_idx, i, j in ti.ndrange(
            self.n_sims[None], self.nn_hidden_size[None], self.max_n_springs[None]
        ):
            if self.training_mask[sim_idx] == 1:
                grad = self.clip_grad(self.weights2.grad[sim_idx, i, j])
                self.weights2_grad_m[sim_idx, i, j] = (
                    self.adam_beta1[None] * self.weights2_grad_m[sim_idx, i, j]
                    + (1.0 - self.adam_beta1[None]) * grad
                )
                self.weights2_grad_v[sim_idx, i, j] = (
                    self.adam_beta2[None] * self.weights2_grad_v[sim_idx, i, j]
                    + (1.0 - self.adam_beta2[None]) * grad * grad
                )
                m_hat = self.weights2_grad_m[sim_idx, i, j] / (
                    1.0 - ti.pow(self.adam_beta1[None], self.adam_step[None])
                )
                v_hat = self.weights2_grad_v[sim_idx, i, j] / (
                    1.0 - ti.pow(self.adam_beta2[None], self.adam_step[None])
                )
                self.weights2[sim_idx, i, j] += (
                    -self.learning_rate[None] * m_hat / (ti.sqrt(v_hat) + self.eps[None])
                )

        for sim_idx, i in ti.ndrange(self.n_sims[None], self.nn_hidden_size[None]):
            if self.training_mask[sim_idx] == 1:
                grad = self.clip_grad(self.biases1.grad[sim_idx, i])
                self.biases1_grad_m[sim_idx, i] = (
                    self.adam_beta1[None] * self.biases1_grad_m[sim_idx, i]
                    + (1.0 - self.adam_beta1[None]) * grad
                )
                self.biases1_grad_v[sim_idx, i] = (
                    self.adam_beta2[None] * self.biases1_grad_v[sim_idx, i]
                    + (1.0 - self.adam_beta2[None]) * grad * grad
                )
                m_hat = self.biases1_grad_m[sim_idx, i] / (
                    1.0 - ti.pow(self.adam_beta1[None], self.adam_step[None])
                )
                v_hat = self.biases1_grad_v[sim_idx, i] / (
                    1.0 - ti.pow(self.adam_beta2[None], self.adam_step[None])
                )
                self.biases1[sim_idx, i] += (
                    -self.learning_rate[None] * m_hat / (ti.sqrt(v_hat) + self.eps[None])
                )

        for sim_idx, i in ti.ndrange(self.n_sims[None], self.max_n_springs[None]):
            if self.training_mask[sim_idx] == 1:
                grad = self.clip_grad(self.biases2.grad[sim_idx, i])
                self.biases2_grad_m[sim_idx, i] = (
                    self.adam_beta1[None] * self.biases2_grad_m[sim_idx, i]
                    + (1.0 - self.adam_beta1[None]) * grad
                )
                self.biases2_grad_v[sim_idx, i] = (
                    self.adam_beta2[None] * self.biases2_grad_v[sim_idx, i]
                    + (1.0 - self.adam_beta2[None]) * grad * grad
                )
                m_hat = self.biases2_grad_m[sim_idx, i] / (
                    1.0 - ti.pow(self.adam_beta1[None], self.adam_step[None])
                )
                v_hat = self.biases2_grad_v[sim_idx, i] / (
                    1.0 - ti.pow(self.adam_beta2[None], self.adam_step[None])
                )
                self.biases2[sim_idx, i] += (
                    -self.learning_rate[None] * m_hat / (ti.sqrt(v_hat) + self.eps[None])
                )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    @ti.kernel
    def reinitialize_robots(self):
        """Reset per-episode state while keeping geometry and weights."""
        for sim_idx, t, mass_idx in ti.ndrange(
            self.n_sims[None], self.steps[None] + 1, self.max_n_masses[None]
        ):
            if t > 0:
                self.x[sim_idx, t, mass_idx] = ti.Vector([0.0, 0.0], dt=ti.f32)
        for sim_idx, t, mass_idx in ti.ndrange(
            self.n_sims[None], self.steps[None] + 1, self.max_n_masses[None]
        ):
            self.v[sim_idx, t, mass_idx] = ti.Vector([0.0, 0.0], dt=ti.f32)
            self.vinc[sim_idx, t, mass_idx] = ti.Vector([0.0, 0.0], dt=ti.f32)
            self.center[sim_idx, t] = ti.Vector([0.0, 0.0], dt=ti.f32)
        for sim_idx, t, spring_idx in ti.ndrange(
            self.n_sims[None], self.steps[None], self.max_n_springs[None]
        ):
            self.act[sim_idx, t, spring_idx] = 0.0
            self.target_L[sim_idx, t, spring_idx] = 0.0
            self.actual_L[sim_idx, t, spring_idx] = 0.0
        for sim_idx in range(self.n_sims[None]):
            self.loss[sim_idx] = 0.0
            self.dist_sum[sim_idx] = 0.0
            self.effort_sum[sim_idx] = 0.0
        for sim_idx, t, hidden_idx in ti.ndrange(
            self.n_sims[None], self.steps[None], self.nn_hidden_size[None]
        ):
            self.hidden[sim_idx, t, hidden_idx] = 0.0
        for sim_idx, t in ti.ndrange(self.n_sims[None], self.steps[None]):
            self.effort_step[sim_idx, t] = 0.0

    def clear_grads(self):
        self.x.grad.fill(0.0)
        self.center.grad.fill(0.0)
        self.v.grad.fill(0.0)
        self.vinc.grad.fill(0.0)
        self.act.grad.fill(0.0)
        self.loss.grad.fill(0.0)
        self.weights1.grad.fill(0.0)
        self.weights2.grad.fill(0.0)
        self.biases1.grad.fill(0.0)
        self.biases2.grad.fill(0.0)
        self.hidden.grad.fill(0.0)
        self.target_L.grad.fill(0.0)
        self.effort_step.grad.fill(0.0)
        self.dist_sum.grad.fill(0.0)
        self.effort_sum.grad.fill(0.0)

    def hard_reset(self):
        """Zero everything, including geometry and weights."""
        self.x.fill(0.0)
        self.center.fill(0.0)
        self.v.fill(0.0)
        self.vinc.fill(0.0)
        self.n_masses.fill(0)
        self.springL.fill(0.0)
        self.springs.fill(0)
        self.n_springs.fill(0)
        self.act.fill(0.0)
        self.loss.fill(0.0)
        self.weights1.fill(0.0)
        self.weights2.fill(0.0)
        self.biases1.fill(0.0)
        self.biases2.fill(0.0)
        self.weights1_grad_m.fill(0.0)
        self.weights2_grad_m.fill(0.0)
        self.biases1_grad_m.fill(0.0)
        self.biases2_grad_m.fill(0.0)
        self.weights1_grad_v.fill(0.0)
        self.weights2_grad_v.fill(0.0)
        self.biases1_grad_v.fill(0.0)
        self.biases2_grad_v.fill(0.0)
        self.hidden.fill(0.0)
        self.n_hidden.fill(0)
        self.adam_step[None] = 0
        # New fields
        self.target_L.fill(0.0)
        self.actual_L.fill(0.0)
        self.effort_step.fill(0.0)
        self.dist_sum.fill(0.0)
        self.effort_sum.fill(0.0)
        # Safe defaults for opponent/role/mask
        self.opponent_idx.from_numpy(np.arange(self.n_sims[None], dtype=np.int32))
        self.role_sign.fill(1.0)
        self.training_mask.fill(1)
        if self.needs_grad:
            self.clear_grads()

    def reset_adam_state(self):
        """Reset Adam optimiser momentum / variance without touching weights."""
        self.weights1_grad_m.fill(0.0)
        self.weights2_grad_m.fill(0.0)
        self.biases1_grad_m.fill(0.0)
        self.biases2_grad_m.fill(0.0)
        self.weights1_grad_v.fill(0.0)
        self.weights2_grad_v.fill(0.0)
        self.biases1_grad_v.fill(0.0)
        self.biases2_grad_v.fill(0.0)
        self.adam_step[None] = 0

    # ------------------------------------------------------------------
    # Weight get / set (backward-compatible with old saved robots)
    # ------------------------------------------------------------------
    def get_control_params(self, sim_idx):
        params = []
        w1 = self.weights1.to_numpy()
        w2 = self.weights2.to_numpy()
        b1 = self.biases1.to_numpy()
        b2 = self.biases2.to_numpy()
        for i in sim_idx:
            params.append({
                "weights1": w1[i].copy(),
                "weights2": w2[i].copy(),
                "biases1": b1[i].copy(),
                "biases2": b2[i].copy(),
            })
        return params

    def set_control_params(self, sim_idx, control_params):
        w1 = self.weights1.to_numpy()
        w2 = self.weights2.to_numpy()
        b1 = self.biases1.to_numpy()
        b2 = self.biases2.to_numpy()
        w1d = self.w1_input_dim[None]
        for idx, i in enumerate(sim_idx):
            cp_w1 = control_params[idx]["weights1"]
            cp_w2 = control_params[idx]["weights2"]
            cp_b1 = control_params[idx]["biases1"]
            cp_b2 = control_params[idx]["biases2"]
            # Handle old robots that lack opponent-feature weight rows
            if cp_w1.shape[0] < w1d:
                pad = np.zeros((w1d - cp_w1.shape[0], cp_w1.shape[1]), dtype=np.float32)
                cp_w1 = np.vstack([cp_w1, pad])
            w1[i] = cp_w1
            w2[i] = cp_w2
            b1[i] = cp_b1
            b2[i] = cp_b2
        self.weights1.from_numpy(w1)
        self.weights2.from_numpy(w2)
        self.biases1.from_numpy(b1)
        self.biases2.from_numpy(b2)
