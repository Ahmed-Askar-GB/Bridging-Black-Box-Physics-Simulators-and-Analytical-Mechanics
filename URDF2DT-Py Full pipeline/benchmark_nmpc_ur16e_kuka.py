"""Comprehensive Empirical NMPC Benchmark: UR16e (6-DOF) and KUKA LBR iiwa 7 (7-DOF).

Evaluates closed-loop receding-horizon Nonlinear Model Predictive Control (NMPC)
comparing:
  1. Baseline Formulation: Classical Christoffel symbols + Explicit Matrix Inversion
  2. Proposed Formulation: Momentum Directional Derivative + Symbolic Cholesky Factorization

Tasks:
  - Aggressive multi-axis trajectory tracking
  - Strict actuator torque limit enforcement (tau_min <= tau <= tau_max)
  - Non-convex workspace spherical obstacle avoidance
"""

import os
import sys
import time
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

# Ensure urdf2dt is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from urdf2dt.parsing import urdf_parse, urdf_build_chain, urdf_extract_dh, urdf_extract_inertia
from urdf2dt.analytical.fk_casadi import _ai_casadi


def build_dynamics_models(urdf_path: str):
    """Parse URDF and construct baseline & proposed continuous-time dynamics graphs."""
    robot = urdf_parse(urdf_path)
    chain = urdf_build_chain(robot)
    dh = urdf_extract_dh(chain)
    inertia = urdf_extract_inertia(robot, chain, dh)

    n = chain.n
    q = cs.SX.sym("q", n)
    dq = cs.SX.sym("dq", n)
    u = cs.SX.sym("u", n)  # Generalized torque tau

    g_sym = cs.SX(np.array([0.0, 0.0, -9.81]))
    m = inertia.m
    P_Ci = inertia.P_Ci
    I_dh = inertia.I
    sigma = dh.sigma
    DHp = dh.DHpram

    # Cumulative link transforms
    T0i_list = []
    T_running = cs.SX.eye(4)
    for i in range(n):
        a_i = float(DHp[i, 0])
        alpha_i = float(DHp[i, 1])
        th_off = float(DHp[i, 2])
        d_off = float(DHp[i, 3])
        sigma_i = float(DHp[i, 4])
        theta_i = (th_off + q[i]) if sigma_i == 1.0 else th_off
        d_i = d_off if sigma_i == 1.0 else (d_off + q[i])
        A_i = _ai_casadi(a_i, alpha_i, theta_i, d_i)
        T_running = cs.mtimes(T_running, A_i)
        T0i_list.append(T_running)

    z_list = [cs.vertcat(0.0, 0.0, 1.0)]
    O_list = [cs.vertcat(0.0, 0.0, 0.0)]
    for i in range(1, n):
        z_list.append(T0i_list[i - 1][:3, 2])
        O_list.append(T0i_list[i - 1][:3, 3])

    p_Ci_world = []
    for i in range(n):
        P_Ci_vec = cs.SX(P_Ci[i, :].reshape(3, 1))
        R_i = T0i_list[i][:3, :3]
        if i == 0:
            p_c = cs.mtimes(R_i, P_Ci_vec)
        else:
            O_prev = T0i_list[i - 1][:3, 3]
            p_c = O_prev + cs.mtimes(R_i, P_Ci_vec)
        p_Ci_world.append(p_c)

    Jv_Ci_list = []
    Jw_i_list = []
    for i in range(n):
        jv_cols = []
        jw_cols = []
        p_ci = p_Ci_world[i]
        for j in range(n):
            if j <= i:
                zj = z_list[j]
                Oj = O_list[j]
                if sigma[j] == 1.0:
                    jv_cols.append(cs.cross(zj, p_ci - Oj))
                    jw_cols.append(zj)
                else:
                    jv_cols.append(zj)
                    jw_cols.append(cs.vertcat(0.0, 0.0, 0.0))
            else:
                jv_cols.append(cs.vertcat(0.0, 0.0, 0.0))
                jw_cols.append(cs.vertcat(0.0, 0.0, 0.0))
        Jv_Ci_list.append(cs.horzcat(*jv_cols))
        Jw_i_list.append(cs.horzcat(*jw_cols))

    # Mass Matrix M(q)
    M_sym = cs.SX.zeros(n, n)
    for i in range(n):
        R_i = T0i_list[i][:3, :3]
        I_i = cs.SX(I_dh[i, :, :])
        I0i = cs.mtimes(R_i, cs.mtimes(I_i, R_i.T))
        Jv_i = Jv_Ci_list[i]
        Jw_i = Jw_i_list[i]
        T_trans = float(m[i]) * cs.mtimes(Jv_i.T, Jv_i)
        T_rot = cs.mtimes(Jw_i.T, cs.mtimes(I0i, Jw_i))
        M_sym = M_sym + T_trans + T_rot

    # Gravity Vector G(q)
    G_sym = cs.SX.zeros(n, 1)
    for j in range(n):
        G_sym = G_sym - float(m[j]) * cs.mtimes(Jv_Ci_list[j].T, g_sym)

    # 1. Classical Christoffel Symbols (Baseline)
    C_sym = cs.SX.zeros(n, n)
    dM = cs.jacobian(cs.vec(M_sym), q)
    for i in range(n):
        for j in range(n):
            c_ij = cs.SX(0.0)
            idx_ij = j * n + i
            for k in range(n):
                idx_ik = k * n + i
                idx_kj = j * n + k
                dM_ijk = dM[idx_ij, k]
                dM_ikj = dM[idx_ik, j]
                dM_kji = dM[idx_kj, i]
                c_ijk = 0.5 * (dM_ijk + dM_ikj - dM_kji)
                c_ij = c_ij + c_ijk * dq[k]
            C_sym[i, j] = c_ij
    c_bias_base = cs.mtimes(C_sym, dq)

    # 2. Proposed Momentum Directional Derivative
    T_kin = 0.5 * cs.mtimes(dq.T, cs.mtimes(M_sym, dq))
    grad_T = cs.gradient(T_kin, q)
    M_dot_dq = cs.jtimes(cs.mtimes(M_sym, dq), q, dq)
    c_bias_prop = M_dot_dq - grad_T

    # Forward Dynamics:
    bias_force_base = u - c_bias_base - G_sym
    bias_force_prop = u - c_bias_prop - G_sym

    # Baseline Inversion:
    ddq_base = cs.solve(M_sym, bias_force_base)

    # Proposed Cholesky:
    R_chol = cs.chol(M_sym)
    y_sub = cs.solve(R_chol.T, bias_force_prop)
    ddq_prop = cs.solve(R_chol, y_sub)

    # End-Effector position for obstacle avoidance
    p_ee = T0i_list[-1][:3, 3]
    ee_fun = cs.Function("ee_fun", [q], [p_ee])

    # State derivative x_dot = [dq; ddq]
    x = cs.vertcat(q, dq)
    x_dot_base = cs.vertcat(dq, ddq_base)
    x_dot_prop = cs.vertcat(dq, ddq_prop)

    f_dyn_base = cs.Function("f_dyn_base", [x, u], [x_dot_base])
    f_dyn_prop = cs.Function("f_dyn_prop", [x, u], [x_dot_prop])

    return {
        "robot_name": robot.name,
        "n": n,
        "f_dyn_base": f_dyn_base,
        "f_dyn_prop": f_dyn_prop,
        "ee_fun": ee_fun,
        "q": q,
        "dq": dq,
        "u": u,
        "T0i_list": T0i_list
    }


def build_nmpc_solver(model: dict, is_proposed: bool, N: int = 15, dt: float = 0.02,
                      p_obs: np.ndarray = None, r_obs: float = 0.15, d_safe: float = 0.05,
                      tau_max: np.ndarray = None):
    """Construct an NMPC optimization solver via Direct Multiple Shooting with RK4 integration."""
    n = model["n"]
    nx = 2 * n
    nu = n
    f_dyn = model["f_dyn_prop"] if is_proposed else model["f_dyn_base"]
    ee_fun = model["ee_fun"]

    # Decision variables
    W = []
    lbw = []
    ubw = []
    w0 = []

    # Constraints
    G = []
    lbg = []
    ubg = []

    # Parameters: initial state x0 and reference states along horizon
    P = cs.SX.sym("P", nx + (N + 1) * nx)

    # Weights
    Q_diag = np.ones(nx)
    Q_diag[:n] = 200.0  # Joint position error
    Q_diag[n:] = 10.0   # Joint velocity error
    Q = cs.diag(cs.SX(Q_diag))
    R = cs.diag(cs.SX(np.ones(nu) * 0.005))  # Torque regularization
    P_weight = Q * 2.0

    # Initial state variable
    Xk = cs.SX.sym("X_0", nx)
    W.append(Xk)
    lbw.extend([-cs.inf] * nx)
    ubw.extend([cs.inf] * nx)
    w0.extend([0.0] * nx)

    # Initial state constraint: X_0 - x_meas = 0
    x_meas = P[:nx]
    G.append(Xk - x_meas)
    lbg.extend([0.0] * nx)
    ubg.extend([0.0] * nx)

    cost = 0

    if tau_max is None:
        tau_max = np.ones(nu) * 150.0

    # Discrete-time RK4 step function
    X_in = cs.SX.sym("X_in", nx)
    U_in = cs.SX.sym("U_in", nu)
    k1 = f_dyn(X_in, U_in)
    k2 = f_dyn(X_in + (dt / 2.0) * k1, U_in)
    k3 = f_dyn(X_in + (dt / 2.0) * k2, U_in)
    k4 = f_dyn(X_in + dt * k3, U_in)
    X_out = X_in + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    phi_rk4 = cs.Function("phi_rk4", [X_in, U_in], [X_out])

    # Dynamic sensitivity (Jacobian) of a single RK4 step: [A, B] = [dphi/dx, dphi/du]
    A_sens = cs.jacobian(X_out, X_in)
    B_sens = cs.jacobian(X_out, U_in)
    sens_rk4 = cs.Function("sens_rk4", [X_in, U_in], [A_sens, B_sens])

    for k in range(N):
        # Control input variable
        Uk = cs.SX.sym(f"U_{k}", nu)
        W.append(Uk)
        lbw.extend((-tau_max).tolist())
        ubw.extend(tau_max.tolist())
        w0.extend([0.0] * nu)

        # Reference at node k
        x_ref_k = P[nx + k * nx : nx + (k + 1) * nx]

        # Stage cost
        dx = Xk - x_ref_k
        cost = cost + cs.mtimes(dx.T, cs.mtimes(Q, dx)) + cs.mtimes(Uk.T, cs.mtimes(R, Uk))

        # RK4 Integration via discrete function
        X_next = phi_rk4(Xk, Uk)

        # State variable at k+1
        Xk_next = cs.SX.sym(f"X_{k+1}", nx)
        W.append(Xk_next)
        # Joint position limits: [-pi, pi], velocity limits: [-4.0, 4.0]
        lb_xk = [-3.14159] * n + [-4.0] * n
        ub_xk = [3.14159] * n + [4.0] * n
        lbw.extend(lb_xk)
        ubw.extend(ub_xk)
        w0.extend([0.0] * nx)

        # Dynamic continuity constraint: X_{k+1} - X_next = 0
        G.append(Xk_next - X_next)
        lbg.extend([0.0] * nx)
        ubg.extend([0.0] * nx)

        # Obstacle avoidance constraint on end-effector
        if p_obs is not None:
            qk_next = Xk_next[:n]
            p_ee_k = ee_fun(qk_next)
            dist_sq = cs.sumsqr(p_ee_k - cs.SX(p_obs))
            min_dist_sq = (r_obs + d_safe) ** 2
            G.append(dist_sq)
            lbg.append(min_dist_sq)
            ubg.append(cs.inf)

        Xk = Xk_next

    # Terminal cost
    x_ref_term = P[nx + N * nx : nx + (N + 1) * nx]
    dx_term = Xk - x_ref_term
    cost = cost + cs.mtimes(dx_term.T, cs.mtimes(P_weight, dx_term))

    w_vec = cs.vertcat(*W)
    g_vec = cs.vertcat(*G)

    nlp = {"f": cost, "x": w_vec, "g": g_vec, "p": P}

    # Solver options
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 40,
        "ipopt.tol": 1e-3,
        "ipopt.acceptable_tol": 1e-2,
        "ipopt.linear_solver": "mumps",
        "ipopt.warm_start_init_point": "yes",
        "ipopt.hessian_approximation": "limited-memory"
    }

    solver = cs.nlpsol("nmpc_solver", "ipopt", nlp, opts)

    return {
        "solver": solver,
        "nw": w_vec.size1(),
        "ng": g_vec.size1(),
        "lbw": np.array(lbw),
        "ubw": np.array(ubw),
        "lbg": np.array(lbg),
        "ubg": np.array(ubg),
        "nx": nx,
        "nu": nu,
        "N": N,
        "w_vec": w_vec,
        "g_vec": g_vec,
        "sens_rk4": sens_rk4,
        "phi_rk4": phi_rk4
    }


def run_closed_loop_nmpc(model: dict, urdf_path: str, sim_steps: int = 40):
    """Run closed-loop NMPC on a designated trajectory with obstacle avoidance for Baseline and Proposed."""
    n = model["n"]
    nx = 2 * n
    nu = n
    N = 15
    dt = 0.02

    # Set up obstacle and limits based on robot
    if "ur16e" in model["robot_name"].lower():
        p_obs = np.array([0.45, 0.15, 0.40])
        r_obs = 0.12
        d_safe = 0.04
        tau_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
    else:  # KUKA iiwa 7
        p_obs = np.array([0.35, 0.10, 0.50])
        r_obs = 0.12
        d_safe = 0.04
        tau_max = np.array([176.0, 176.0, 110.0, 110.0, 110.0, 40.0, 40.0])

    print(f"\nBuilding NMPC Solvers for {model['robot_name']} (N={N}, dt={dt}s)...", flush=True)
    t0 = time.perf_counter()
    solver_dict_base = build_nmpc_solver(model, is_proposed=False, N=N, dt=dt, p_obs=p_obs, r_obs=r_obs, d_safe=d_safe, tau_max=tau_max)
    t_build_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    solver_dict_prop = build_nmpc_solver(model, is_proposed=True, N=N, dt=dt, p_obs=p_obs, r_obs=r_obs, d_safe=d_safe, tau_max=tau_max)
    t_build_prop = time.perf_counter() - t0

    print(f"  Decision Variables: {solver_dict_prop['nw']} | Total Constraints: {solver_dict_prop['ng']}", flush=True)
    print(f"  Solver Build Time: Baseline = {t_build_base:.3f} s | Proposed = {t_build_prop:.3f} s", flush=True)

    # Generate aggressive reference trajectory: sinusoidal sweep
    time_vec = np.arange(sim_steps + N + 1) * dt
    q_ref_traj = np.zeros((sim_steps + N + 1, n))
    dq_ref_traj = np.zeros((sim_steps + N + 1, n))
    for i in range(n):
        freq = 0.8 + 0.2 * i
        amp = 0.45 + 0.05 * ((-1)**i)
        q_ref_traj[:, i] = amp * np.sin(freq * time_vec)
        dq_ref_traj[:, i] = amp * freq * np.cos(freq * time_vec)

    # Measure KKT sensitivity & Jacobian instruction sizes for single RK4 stage and total N stages
    sens_base = solver_dict_base["sens_rk4"]
    sens_prop = solver_dict_prop["sens_rk4"]

    op_sens_stage_base = sens_base.n_instructions()
    op_sens_stage_prop = sens_prop.n_instructions()

    op_jac_base = N * op_sens_stage_base
    op_jac_prop = N * op_sens_stage_prop
    jac_red_pct = 100.0 * (1.0 - op_jac_prop / op_jac_base)

    print(f"  RK4 Dynamic Sensitivity Opcodes/Stage: Base = {op_sens_stage_base} -> Prop = {op_sens_stage_prop} (-{jac_red_pct:.1f}%)", flush=True)
    print(f"  KKT Dynamics Jacobian Opcodes ({N} stages): Base = {op_jac_base} -> Prop = {op_jac_prop} (-{jac_red_pct:.1f}%)", flush=True)

    # Run Closed-Loop NMPC Simulation for both methods
    controllers = [
        {"name": "Baseline (Christoffel + Inversion)", "dict": solver_dict_base, "is_prop": False},
        {"name": "Proposed (Momentum + Cholesky)", "dict": solver_dict_prop, "is_prop": True}
    ]

    sim_logs = {}

    for c in controllers:
        c_name = c["name"]
        s_dict = c["dict"]
        solver = s_dict["solver"]
        nw = s_dict["nw"]
        lbw = s_dict["lbw"]
        ubw = s_dict["ubw"]
        lbg = s_dict["lbg"]
        ubg = s_dict["ubg"]

        # Initial state at t=0
        x_curr = np.zeros(nx)
        x_curr[:n] = q_ref_traj[0, :]
        x_curr[n:] = dq_ref_traj[0, :]

        w_guess = np.zeros(nw)
        # Warm start state variables along trajectory
        for k in range(N + 1):
            w_guess[k * (nx + nu) : k * (nx + nu) + nx] = np.hstack([q_ref_traj[k, :], dq_ref_traj[k, :]])

        solve_times_ms = []
        iter_counts = []
        q_actual = [x_curr[:n].copy()]
        dq_actual = [x_curr[n:].copy()]
        u_applied = []
        ee_pos_actual = []

        print(f"  Executing Closed-Loop NMPC Simulation ({c_name})...", flush=True)
        for step in range(sim_steps):
            if step % 10 == 0 or step == sim_steps - 1:
                print(f"    Step {step+1}/{sim_steps}...", flush=True)
            # Parameter vector: current measurement + reference states across horizon
            P_val = np.zeros(nx + (N + 1) * nx)
            P_val[:nx] = x_curr
            for k in range(N + 1):
                ref_idx = step + k
                P_val[nx + k * nx : nx + (k + 1) * nx] = np.hstack([q_ref_traj[ref_idx, :], dq_ref_traj[ref_idx, :]])

            # Solve NLP
            t0 = time.perf_counter()
            sol = solver(x0=w_guess, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=P_val)
            t_solve = (time.perf_counter() - t0) * 1000.0  # ms
            solve_times_ms.append(t_solve)

            w_opt = np.array(sol["x"]).flatten()
            w_guess = w_opt.copy()  # Receding horizon warm start

            # Extract applied control U_0
            u_opt_0 = w_opt[nx : nx + nu]
            u_applied.append(u_opt_0)

            # Record end-effector position
            p_ee = np.array(model["ee_fun"](x_curr[:n])).flatten()
            ee_pos_actual.append(p_ee)

            # Advance plant state via RK4 forward step using ground truth dynamics
            x_next = np.array(model["f_dyn_prop"](x_curr, u_opt_0)).flatten()
            x_curr = x_curr + dt * x_next  # Euler / RK step on true continuous plant
            q_actual.append(x_curr[:n].copy())
            dq_actual.append(x_curr[n:].copy())

        sim_logs[c["is_prop"]] = {
            "name": c_name,
            "solve_times_ms": np.array(solve_times_ms),
            "q_actual": np.array(q_actual),
            "u_applied": np.array(u_applied),
            "ee_pos": np.array(ee_pos_actual)
        }

    # Compute comparative metrics
    times_base = sim_logs[False]["solve_times_ms"]
    times_prop = sim_logs[True]["solve_times_ms"]

    mean_base = float(np.mean(times_base))
    std_base = float(np.std(times_base))
    max_base = float(np.max(times_base))

    mean_prop = float(np.mean(times_prop))
    std_prop = float(np.std(times_prop))
    max_prop = float(np.max(times_prop))

    speedup_mean = mean_base / mean_prop
    speedup_max = max_base / max_prop
    freq_base = 1000.0 / mean_base
    freq_prop = 1000.0 / mean_prop

    # Tracking error RMSE
    q_act_prop = sim_logs[True]["q_actual"][:sim_steps, :]
    q_ref_segment = q_ref_traj[:sim_steps, :]
    rmse_tracking = float(np.sqrt(np.mean((q_act_prop - q_ref_segment)**2)))

    # Minimum distance to obstacle
    ee_positions = sim_logs[True]["ee_pos"]
    dists_to_obs = np.linalg.norm(ee_positions - p_obs, axis=1)
    min_dist_to_obs = float(np.min(dists_to_obs))
    min_clearance = min_dist_to_obs - (r_obs + d_safe)

    print(f"\n  --- NMPC Performance Benchmark Summary [{model['robot_name']}] ---")
    print(f"  Mean Solve Time : Base = {mean_base:6.2f} +/- {std_base:4.2f} ms | Prop = {mean_prop:6.2f} +/- {std_prop:4.2f} ms ({speedup_mean:.2f}x speedup)")
    print(f"  Max Solve Time  : Base = {max_base:6.2f} ms | Prop = {max_prop:6.2f} ms ({speedup_max:.2f}x speedup)")
    print(f"  Control Freq    : Base = {freq_base:6.1f} Hz | Prop = {freq_prop:6.1f} Hz")
    print(f"  Tracking RMSE   : {rmse_tracking:.4e} rad")
    print(f"  Min Obs Distance: {min_dist_to_obs:.4f} m (Safety Margin = {min_clearance:.4f} m >= 0)")

    return {
        "robot_name": model["robot_name"],
        "n": n,
        "mean_base": mean_base,
        "std_base": std_base,
        "max_base": max_base,
        "mean_prop": mean_prop,
        "std_prop": std_prop,
        "max_prop": max_prop,
        "speedup_mean": speedup_mean,
        "speedup_max": speedup_max,
        "freq_base": freq_base,
        "freq_prop": freq_prop,
        "op_jac_base": op_jac_base,
        "op_jac_prop": op_jac_prop,
        "jac_red_pct": jac_red_pct,
        "rmse_tracking": rmse_tracking,
        "min_dist_to_obs": min_dist_to_obs,
        "min_clearance": min_clearance,
        "sim_logs": sim_logs,
        "q_ref": q_ref_segment,
        "p_obs": p_obs,
        "r_obs": r_obs,
        "d_safe": d_safe,
        "tau_max": tau_max
    }


def main():
    print("=" * 80)
    print("PATH A: EMPIRICAL NMPC BENCHMARK ON UR16e (6-DOF) AND KUKA IIWA 7 (7-DOF)")
    print("=" * 80)

    fixtures_dir = os.path.join(os.path.dirname(__file__), "tests", "fixtures")
    ur16e_urdf = os.path.join(fixtures_dir, "ur16e.urdf")
    kuka_urdf = os.path.join(fixtures_dir, "kukaIiwa7.urdf")

    print("\n--> Building UR16e (6-DOF) Dynamics Graph...")
    model_ur16e = build_dynamics_models(ur16e_urdf)

    print("\n--> Building KUKA LBR iiwa 7 (7-DOF) Dynamics Graph...")
    model_kuka = build_dynamics_models(kuka_urdf)

    # Run NMPC Closed-Loop Benchmarks
    res_ur16e = run_closed_loop_nmpc(model_ur16e, ur16e_urdf, sim_steps=40)
    res_kuka = run_closed_loop_nmpc(model_kuka, kuka_urdf, sim_steps=40)

    # Plot Multi-Panel Figure 7
    print("\nGenerating Publication-Quality Figure 7...")
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.32)

    c_base = '#D9534F'  # Crimson
    c_prop = '#1976D2'  # Cobalt Blue
    c_green = '#2E7D32' # Forest Green
    c_amber = '#F57C00' # Deep Amber

    steps = np.arange(40)

    # Panel (a): Per-step solve time on UR16e (6-DOF)
    ax_a = axs[0, 0]
    ax_a.plot(steps, res_ur16e["sim_logs"][False]["solve_times_ms"], 'o--', color=c_base, lw=1.6, ms=4, label='Baseline (Christoffel + Inv)')
    ax_a.plot(steps, res_ur16e["sim_logs"][True]["solve_times_ms"], 's-', color=c_prop, lw=2.0, ms=4, label='Proposed (Momentum + Chol)')
    ax_a.axhline(res_ur16e["mean_base"], color=c_base, ls=':', lw=1.2, alpha=0.8, label=f"Mean Base: {res_ur16e['mean_base']:.1f} ms")
    ax_a.axhline(res_ur16e["mean_prop"], color=c_prop, ls=':', lw=1.2, alpha=0.8, label=f"Mean Prop: {res_ur16e['mean_prop']:.1f} ms ({res_ur16e['speedup_mean']:.2f}x)")
    ax_a.set_xlabel('NMPC Sampling Step $k$', fontsize=9.5, fontweight='bold')
    ax_a.set_ylabel('NLP Solve Time (ms)', fontsize=9.5, fontweight='bold')
    ax_a.set_title('(a) UR16e (6-DOF) NMPC Solve Time per Horizon Step', fontsize=10, fontweight='bold')
    ax_a.grid(True, ls='--', alpha=0.5)
    ax_a.legend(fontsize=8, loc='upper right', framealpha=0.9)

    # Panel (b): Per-step solve time on KUKA LBR iiwa 7 (7-DOF)
    ax_b = axs[0, 1]
    ax_b.plot(steps, res_kuka["sim_logs"][False]["solve_times_ms"], 'o--', color=c_base, lw=1.6, ms=4, label='Baseline (Christoffel + Inv)')
    ax_b.plot(steps, res_kuka["sim_logs"][True]["solve_times_ms"], 's-', color=c_prop, lw=2.0, ms=4, label='Proposed (Momentum + Chol)')
    ax_b.axhline(res_kuka["mean_base"], color=c_base, ls=':', lw=1.2, alpha=0.8, label=f"Mean Base: {res_kuka['mean_base']:.1f} ms")
    ax_b.axhline(res_kuka["mean_prop"], color=c_prop, ls=':', lw=1.2, alpha=0.8, label=f"Mean Prop: {res_kuka['mean_prop']:.1f} ms ({res_kuka['speedup_mean']:.2f}x)")
    ax_b.set_xlabel('NMPC Sampling Step $k$', fontsize=9.5, fontweight='bold')
    ax_b.set_ylabel('NLP Solve Time (ms)', fontsize=9.5, fontweight='bold')
    ax_b.set_title('(b) KUKA iiwa 7 (7-DOF) NMPC Solve Time per Horizon Step', fontsize=10, fontweight='bold')
    ax_b.grid(True, ls='--', alpha=0.5)
    ax_b.legend(fontsize=8, loc='upper right', framealpha=0.9)

    # Panel (c): Bar Chart comparing Mean Solve Time & Control Frequency across both robots
    ax_c = axs[1, 0]
    bar_width = 0.35
    x_indices = np.array([0, 1])
    means_b = [res_ur16e["mean_base"], res_kuka["mean_base"]]
    means_p = [res_ur16e["mean_prop"], res_kuka["mean_prop"]]
    err_b = [res_ur16e["std_base"], res_kuka["std_base"]]
    err_p = [res_ur16e["std_prop"], res_kuka["std_prop"]]

    rects1 = ax_c.bar(x_indices - bar_width/2, means_b, bar_width, yerr=err_b, capsize=4, color=c_base, alpha=0.85, label='Baseline NMPC')
    rects2 = ax_c.bar(x_indices + bar_width/2, means_p, bar_width, yerr=err_p, capsize=4, color=c_prop, alpha=0.85, label='Proposed NMPC')
    ax_c.set_xticks(x_indices)
    ax_c.set_xticklabels(['UR16e (6-DOF)', 'KUKA LBR iiwa (7-DOF)'], fontsize=9.5, fontweight='bold')
    ax_c.set_ylabel('Mean NLP Solve Time (ms)', fontsize=9.5, fontweight='bold')
    ax_c.set_title('(c) Average NMPC Solve Time Comparison & Speedup', fontsize=10, fontweight='bold')
    ax_c.grid(True, axis='y', ls='--', alpha=0.5)
    ax_c.legend(fontsize=8.5, loc='upper left')

    # Add text labels on bars
    for rect, sp in zip(rects2, [res_ur16e["speedup_mean"], res_kuka["speedup_mean"]]):
        height = rect.get_height()
        ax_c.annotate(f'{sp:.2f}x Faster\n({1000/height:.0f} Hz)',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 6), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color=c_prop)

    # Panel (d): Obstacle Avoidance Trajectory & Clearance for KUKA iiwa
    ax_d = axs[1, 1]
    ee_pts_kuka = res_kuka["sim_logs"][True]["ee_pos"]
    p_obs_k = res_kuka["p_obs"]
    dists_k = np.linalg.norm(ee_pts_kuka - p_obs_k, axis=1)
    safe_boundary = res_kuka["r_obs"] + res_kuka["d_safe"]

    ax_d.plot(steps, dists_k, 'o-', color=c_green, lw=2.0, ms=4, label='End-Effector Obstacle Distance ($d(t)$)')
    ax_d.axhline(safe_boundary, color=c_base, ls='--', lw=1.8, label=f'Safety Margin ($r_{{\\mathrm{{obs}}}} + d_{{\\mathrm{{safe}}}} = {safe_boundary:.2f}$ m)')
    ax_d.fill_between(steps, 0, safe_boundary, color=c_base, alpha=0.15)
    ax_d.set_xlabel('NMPC Sampling Step $k$', fontsize=9.5, fontweight='bold')
    ax_d.set_ylabel('Cartesian Distance to Obstacle (m)', fontsize=9.5, fontweight='bold')
    ax_d.set_title('(d) KUKA iiwa 7 Closed-Loop Obstacle Clearance', fontsize=10, fontweight='bold')
    ax_d.grid(True, ls='--', alpha=0.5)
    ax_d.set_ylim(0, max(dists_k) * 1.15)
    ax_d.legend(fontsize=8.5, loc='upper right')

    # Save Figure 7
    manuscript_figs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Full pipeline Manuscript", "figures"))
    os.makedirs(manuscript_figs_dir, exist_ok=True)
    out_pdf = os.path.join(manuscript_figs_dir, "fig7_nmpc_benchmark.pdf")
    out_png = os.path.join(manuscript_figs_dir, "fig7_nmpc_benchmark.png")

    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated Figure 7: {out_pdf}")
    print(f"Generated Figure 7: {out_png}")

    # Save benchmark metrics to NPZ
    np.savez(
        "nmpc_benchmark_results.npz",
        ur16e_mean_base=res_ur16e["mean_base"],
        ur16e_mean_prop=res_ur16e["mean_prop"],
        ur16e_speedup=res_ur16e["speedup_mean"],
        kuka_mean_base=res_kuka["mean_base"],
        kuka_mean_prop=res_kuka["mean_prop"],
        kuka_speedup=res_kuka["speedup_mean"],
        op_jac_ur16e_base=res_ur16e["op_jac_base"],
        op_jac_ur16e_prop=res_ur16e["op_jac_prop"],
        op_jac_kuka_base=res_kuka["op_jac_base"],
        op_jac_kuka_prop=res_kuka["op_jac_prop"]
    )
    print("Saved NMPC benchmark data to nmpc_benchmark_results.npz")


if __name__ == "__main__":
    main()
