"""Systematic Multi-DOF Scalability Benchmark Suite.

Evaluates symbolic graph size (opcodes), derivation time, code-generation metrics,
and runtime execution latencies across robots with varying degrees of freedom:
n in [3, 4, 6, 7, 10, 12, 14].
"""

import os
import sys
import time
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

# Add current folder to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from urdf2dt.parsing import urdf_parse, urdf_build_chain, urdf_extract_dh, urdf_extract_inertia
from urdf2dt.analytical.fk_casadi import _ai_casadi


def generate_serial_urdf(n: int, filename: str):
    """Generate a clean, physically realizable serial manipulator URDF of n DOFs."""
    lines = ['<?xml version="1.0"?>']
    lines.append(f'<robot name="serial_arm_{n}dof">')
    lines.append('  <link name="base_link">')
    lines.append('    <inertial>')
    lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
    lines.append('      <mass value="5.0"/>')
    lines.append('      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>')
    lines.append('    </inertial>')
    lines.append('  </link>')

    prev_link = "base_link"
    for i in range(1, n + 1):
        link_name = f"link_{i}"
        joint_name = f"joint_{i}"
        
        # Realistic physical tapering
        mass = float(10.0 * (0.85 ** (i - 1)))
        length = float(0.40 * (0.94 ** (i - 1)))
        radius = 0.05 * (0.95 ** (i - 1))
        ixx = float((1.0 / 12.0) * mass * (3.0 * radius**2 + length**2))
        iyy = ixx
        izz = float(0.5 * mass * radius**2)
        
        # Origin offset along z or x
        if i == 1:
            xyz_str = "0 0 0.15"
            rpy_str = "0 0 0"
            axis_str = "0 0 1" # Base yaw
        elif i % 2 == 0:
            xyz_str = f"0 0 {length:.4f}"
            rpy_str = "0 1.57079632679 0"
            axis_str = "0 0 1" # Pitch
        else:
            xyz_str = f"{length:.4f} 0 0"
            rpy_str = "0 -1.57079632679 0"
            axis_str = "0 0 1" # Yaw / Roll

        lines.append(f'  <link name="{link_name}">')
        lines.append('    <inertial>')
        lines.append(f'      <origin xyz="{length/2.0:.4f} 0 0" rpy="0 0 0"/>')
        lines.append(f'      <mass value="{mass:.4f}"/>')
        lines.append(f'      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>')
        lines.append('    </inertial>')
        lines.append('  </link>')

        lines.append(f'  <joint name="{joint_name}" type="revolute">')
        lines.append(f'    <parent link="{prev_link}"/>')
        lines.append(f'    <child link="{link_name}"/>')
        lines.append(f'    <origin xyz="{xyz_str}" rpy="{rpy_str}"/>')
        lines.append(f'    <axis xyz="{axis_str}"/>')
        lines.append('    <limit lower="-3.14159" upper="3.14159" effort="150" velocity="3.14"/>')
        lines.append('  </joint>')

        prev_link = link_name

    lines.append('</robot>')
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


def profile_robot_dynamics(urdf_path: str):
    """Profile dynamics derivation, opcodes, and runtime across baseline and proposed methods."""
    robot = urdf_parse(urdf_path)
    chain = urdf_build_chain(robot)
    dh = urdf_extract_dh(chain)
    inertia = urdf_extract_inertia(robot, chain, dh)

    n = chain.n
    q = cs.SX.sym("q", n)
    dq = cs.SX.sym("dq", n)
    ddq = cs.SX.sym("ddq", n)
    tau = cs.SX.sym("tau", n)

    g_sym = cs.SX(np.array([0.0, 0.0, -9.81]))
    m = inertia.m
    P_Ci = inertia.P_Ci
    I_dh = inertia.I
    sigma = dh.sigma
    DHp = dh.DHpram

    t0_deriv = time.perf_counter()

    # Forward Kinematics
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
    t0_chris = time.perf_counter()
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
    c_bias_christoffel = cs.mtimes(C_sym, dq)
    t_deriv_chris = time.perf_counter() - t0_chris

    # 2. Proposed Momentum Directional Derivative
    t0_mom = time.perf_counter()
    T_kin = 0.5 * cs.mtimes(dq.T, cs.mtimes(M_sym, dq))
    grad_T = cs.gradient(T_kin, q)
    M_dot_dq = cs.jtimes(cs.mtimes(M_sym, dq), q, dq)
    c_bias_proposed = M_dot_dq - grad_T
    t_deriv_mom = time.perf_counter() - t0_mom

    # Forward Dynamics:
    bias_force_base = tau - c_bias_christoffel - G_sym
    bias_force_prop = tau - c_bias_proposed - G_sym

    # Inversion Baseline:
    fwd_ddq_inv = cs.solve(M_sym, bias_force_base)

    # Cholesky Proposed:
    R_chol = cs.chol(M_sym)
    y_sub = cs.solve(R_chol.T, bias_force_prop)
    fwd_ddq_chol = cs.solve(R_chol, y_sub)

    t_deriv_total = time.perf_counter() - t0_deriv

    # Build CasADi functions to measure exact opcodes
    M_fun = cs.Function("M_fun", [q], [M_sym])
    G_fun = cs.Function("G_fun", [q], [G_sym])
    cbias_base_fun = cs.Function("cbias_base", [q, dq], [c_bias_christoffel])
    cbias_prop_fun = cs.Function("cbias_prop", [q, dq], [c_bias_proposed])
    fd_inv_fun = cs.Function("fd_inv", [q, dq, tau], [fwd_ddq_inv])
    fd_chol_fun = cs.Function("fd_chol", [q, dq, tau], [fwd_ddq_chol])

    op_M = M_fun.n_instructions()
    op_G = G_fun.n_instructions()
    op_cbias_base = cbias_base_fun.n_instructions()
    op_cbias_prop = cbias_prop_fun.n_instructions()
    op_fd_inv = fd_inv_fun.n_instructions()
    op_fd_chol = fd_chol_fun.n_instructions()

    cbias_red_pct = 100.0 * (1.0 - op_cbias_prop / op_cbias_base)
    fd_red_pct = 100.0 * (1.0 - op_fd_chol / op_fd_inv)

    # Numerical Parity Check
    q_test = np.random.randn(n)
    dq_test = np.random.randn(n)
    tau_test = np.random.randn(n)

    val_cbias_base = np.array(cbias_base_fun(q_test, dq_test)).flatten()
    val_cbias_prop = np.array(cbias_prop_fun(q_test, dq_test)).flatten()
    cbias_parity_err = float(np.max(np.abs(val_cbias_base - val_cbias_prop)))

    # Runtime Benchmarks
    N_RUNS = 2000
    # Warmup
    for _ in range(50):
        _ = cbias_base_fun(q_test, dq_test)
        _ = cbias_prop_fun(q_test, dq_test)
        _ = fd_inv_fun(q_test, dq_test, tau_test)
        _ = fd_chol_fun(q_test, dq_test, tau_test)

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _ = cbias_base_fun(q_test, dq_test)
    t_cbias_base_us = (time.perf_counter() - t0) / N_RUNS * 1e6

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _ = cbias_prop_fun(q_test, dq_test)
    t_cbias_prop_us = (time.perf_counter() - t0) / N_RUNS * 1e6

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _ = fd_inv_fun(q_test, dq_test, tau_test)
    t_fd_inv_us = (time.perf_counter() - t0) / N_RUNS * 1e6

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _ = fd_chol_fun(q_test, dq_test, tau_test)
    t_fd_chol_us = (time.perf_counter() - t0) / N_RUNS * 1e6

    fd_speedup = t_fd_inv_us / t_fd_chol_us

    # Measure C code size:
    c_filename = f"temp_chol_{n}.c"
    fd_chol_fun.generate(c_filename)
    c_filesize_kb = os.path.getsize(c_filename) / 1024.0
    if os.path.exists(c_filename):
        os.remove(c_filename)

    return {
        "n": n,
        "op_M": op_M,
        "op_G": op_G,
        "op_cbias_base": op_cbias_base,
        "op_cbias_prop": op_cbias_prop,
        "cbias_red_pct": cbias_red_pct,
        "op_fd_inv": op_fd_inv,
        "op_fd_chol": op_fd_chol,
        "fd_red_pct": fd_red_pct,
        "parity_err": cbias_parity_err,
        "t_deriv_total": t_deriv_total,
        "t_cbias_base_us": t_cbias_base_us,
        "t_cbias_prop_us": t_cbias_prop_us,
        "t_fd_inv_us": t_fd_inv_us,
        "t_fd_chol_us": t_fd_chol_us,
        "fd_speedup": fd_speedup,
        "c_size_kb": c_filesize_kb
    }


def main():
    print("=" * 80)
    print("SYSTEMATIC MULTI-DOF SCALABILITY BENCHMARK SUITE")
    print("Evaluating n in [3, 4, 6, 7, 10, 12, 14] DOFs")
    print("=" * 80)

    dofs = [3, 4, 6, 7, 10, 12, 14]
    results = []

    os.makedirs("temp_urdfs", exist_ok=True)

    for n in dofs:
        urdf_file = os.path.join("temp_urdfs", f"robot_{n}dof.urdf")
        print(f"\n--> Generating & Profiling {n}-DOF Robot Chain...")
        generate_serial_urdf(n, urdf_file)
        res = profile_robot_dynamics(urdf_file)
        results.append(res)
        print(f"   [n={n:2d}] Deriv Time: {res['t_deriv_total']:.3f} s | Parity Err: {res['parity_err']:.2e} N*m")
        print(f"         Coriolis Opcodes : Base={res['op_cbias_base']:6d} -> Prop={res['op_cbias_prop']:6d} (-{res['cbias_red_pct']:.1f}%)")
        print(f"         Fwd Dyn Opcodes  : Inv ={res['op_fd_inv']:6d} -> Chol={res['op_fd_chol']:6d} (-{res['fd_red_pct']:.1f}%)")
        print(f"         FD Latency       : Inv ={res['t_fd_inv_us']:6.1f} us -> Chol={res['t_fd_chol_us']:6.1f} us ({res['fd_speedup']:.2f}x speedup)")

    # Clean up temp urdfs safely
    import shutil
    shutil.rmtree("temp_urdfs", ignore_errors=True)

    # Print Summary Markdown Table
    print("\n" + "=" * 80)
    print("MULTI-DOF QUANTITATIVE SCALING SUMMARY TABLE")
    print("=" * 80)
    header = (
        "| DOFs (n) | M(q) Op | G(q) Op | C_base Op | C_prop Op | C Red. (%) "
        "| FD_inv Op | FD_chol Op | FD Red. (%) | Deriv (s) | FD Inv (us) | FD Chol (us) | Speedup | C Size (KB) |"
    )
    print(header)
    print("|" + "|".join(["---:"] * 14) + "|")
    for r in results:
        row = (
            f"| {r['n']:2d} "
            f"| {r['op_M']:5d} "
            f"| {r['op_G']:4d} "
            f"| {r['op_cbias_base']:8d} "
            f"| {r['op_cbias_prop']:6d} "
            f"| -{r['cbias_red_pct']:4.1f}% "
            f"| {r['op_fd_inv']:8d} "
            f"| {r['op_fd_chol']:6d} "
            f"| -{r['fd_red_pct']:4.1f}% "
            f"| {r['t_deriv_total']:6.3f} "
            f"| {r['t_fd_inv_us']:6.1f} "
            f"| {r['t_fd_chol_us']:6.1f} "
            f"| {r['fd_speedup']:4.2f}x "
            f"| {r['c_size_kb']:6.1f} |"
        )
        print(row)

    # Compute Power-Law Regressions: log(Op) = alpha * log(n) + beta
    log_n = np.log(dofs)
    log_op_c_base = np.log([r['op_cbias_base'] for r in results])
    log_op_c_prop = np.log([r['op_cbias_prop'] for r in results])
    log_op_fd_inv = np.log([r['op_fd_inv'] for r in results])
    log_op_fd_chol = np.log([r['op_fd_chol'] for r in results])

    alpha_c_base, _ = np.polyfit(log_n, log_op_c_base, 1)
    alpha_c_prop, _ = np.polyfit(log_n, log_op_c_prop, 1)
    alpha_fd_inv, _ = np.polyfit(log_n, log_op_fd_inv, 1)
    alpha_fd_chol, _ = np.polyfit(log_n, log_op_fd_chol, 1)

    print("\n" + "=" * 80)
    print("EMPIRICAL ASYMPTOTIC POWER-LAW EXPONENTS (Op ~ O(n^alpha)):")
    print(f"  * Classical Christoffel Coriolis: alpha = {alpha_c_base:.2f} (Theoretical: O(n^4))")
    print(f"  * Proposed Momentum Coriolis    : alpha = {alpha_c_prop:.2f} (Theoretical: O(n^2))")
    print(f"  * Explicit Inversion FD         : alpha = {alpha_fd_inv:.2f} (Theoretical: O(n^3))")
    print(f"  * Proposed Cholesky FD          : alpha = {alpha_fd_chol:.2f} (Theoretical: O(n^2) - O(n^3))")
    print("=" * 80)

    # Save Results as NPZ for Plotting
    np.savez(
        "scaling_benchmark_results.npz",
        dofs=np.array(dofs),
        op_M=np.array([r['op_M'] for r in results]),
        op_G=np.array([r['op_G'] for r in results]),
        op_cbias_base=np.array([r['op_cbias_base'] for r in results]),
        op_cbias_prop=np.array([r['op_cbias_prop'] for r in results]),
        op_fd_inv=np.array([r['op_fd_inv'] for r in results]),
        op_fd_chol=np.array([r['op_fd_chol'] for r in results]),
        t_deriv=np.array([r['t_deriv_total'] for r in results]),
        t_fd_inv=np.array([r['t_fd_inv_us'] for r in results]),
        t_fd_chol=np.array([r['t_fd_chol_us'] for r in results]),
        fd_speedup=np.array([r['fd_speedup'] for r in results]),
        c_size_kb=np.array([r['c_size_kb'] for r in results]),
        alphas=np.array([alpha_c_base, alpha_c_prop, alpha_fd_inv, alpha_fd_chol])
    )
    print("Saved benchmark results to scaling_benchmark_results.npz")

    # Generate Publication-Quality Figure 6
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), dpi=300)
    plt.subplots_adjust(wspace=0.28, hspace=0.32)

    # Color Palette
    c_base = '#D9534F'  # Soft Crimson
    c_prop = '#2E7D32'  # Forest Green
    c_blue = '#1976D2'  # Cobalt Blue
    c_amber = '#F57C00' # Deep Amber

    # Panel (a): Log-Log Opcode Complexity
    ax_a = axs[0, 0]
    ax_a.plot(dofs, [r['op_cbias_base'] for r in results], 'o--', color=c_base, label=f'Baseline Christoffel: $\\mathcal{{O}}(n^{{{alpha_c_base:.2f}}})$', lw=1.8, ms=6)
    ax_a.plot(dofs, [r['op_cbias_prop'] for r in results], 's-', color=c_prop, label=f'Proposed Momentum: $\\mathcal{{O}}(n^{{{alpha_c_prop:.2f}}})$', lw=2.0, ms=6)
    ax_a.plot(dofs, [r['op_fd_inv'] for r in results], '^:', color='#7B1FA2', label=f'Explicit Inversion FD: $\\mathcal{{O}}(n^{{{alpha_fd_inv:.2f}}})$', lw=1.5, ms=5)
    ax_a.plot(dofs, [r['op_fd_chol'] for r in results], 'd-', color=c_blue, label=f'Proposed Cholesky FD: $\\mathcal{{O}}(n^{{{alpha_fd_chol:.2f}}})$', lw=1.8, ms=6)
    ax_a.set_xscale('log')
    ax_a.set_yscale('log')
    ax_a.set_xlabel('Degrees of Freedom ($n$)', fontsize=10, fontweight='bold')
    ax_a.set_ylabel('Symbolic Instruction Count (Opcodes)', fontsize=10, fontweight='bold')
    ax_a.set_title('(a) Asymptotic Instruction Complexity (Log-Log)', fontsize=10, fontweight='bold')
    ax_a.grid(True, which='both', ls='--', alpha=0.5)
    ax_a.legend(fontsize=8, framealpha=0.9, loc='upper left')

    # Panel (b): Opcode Reduction Percentage
    ax_b = axs[0, 1]
    cbias_reds = [r['cbias_red_pct'] for r in results]
    fd_reds = [r['fd_red_pct'] for r in results]
    ax_b.plot(dofs, cbias_reds, 's-', color=c_prop, lw=2.0, ms=6, label='Coriolis Bias Vector ($\\mathbf{c}_{\\mathrm{bias}}$)')
    ax_b.plot(dofs, fd_reds, 'd-', color=c_blue, lw=2.0, ms=6, label='Forward Dynamics ($\\ddot{\\mathbf{q}}$)')
    for x, y in zip(dofs, cbias_reds):
        ax_b.annotate(f'-{y:.1f}%', (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=7.5, color=c_prop, fontweight='bold')
    ax_b.set_xlabel('Degrees of Freedom ($n$)', fontsize=10, fontweight='bold')
    ax_b.set_ylabel('Symbolic Graph Reduction (%)', fontsize=10, fontweight='bold')
    ax_b.set_title('(b) Graph Compression vs. System Dimension', fontsize=10, fontweight='bold')
    ax_b.set_ylim(-5, 90)
    ax_b.grid(True, ls='--', alpha=0.5)
    ax_b.legend(fontsize=8.5, framealpha=0.9, loc='lower right')

    # Panel (c): Forward Dynamics Latency & Speedup
    ax_c = axs[1, 0]
    t_inv = [r['t_fd_inv_us'] for r in results]
    t_chol = [r['t_fd_chol_us'] for r in results]
    speedups = [r['fd_speedup'] for r in results]
    ax_c.plot(dofs, t_inv, '^--', color=c_base, lw=1.8, ms=6, label='Baseline FD Latency ($\\mu$s)')
    ax_c.plot(dofs, t_chol, 'd-', color=c_blue, lw=2.0, ms=6, label='Cholesky FD Latency ($\\mu$s)')
    ax_c.set_xlabel('Degrees of Freedom ($n$)', fontsize=10, fontweight='bold')
    ax_c.set_ylabel('Execution Latency ($\\mu$s)', fontsize=10, fontweight='bold')
    ax_c.set_title('(c) Single-Step Forward Dynamics Latency', fontsize=10, fontweight='bold')
    ax_c.grid(True, ls='--', alpha=0.5)
    ax_c.legend(fontsize=8.5, framealpha=0.9, loc='upper left')

    # Twin axis for Speedup
    ax_c2 = ax_c.twinx()
    ax_c2.plot(dofs, speedups, 'o-', color=c_amber, lw=2.0, ms=6, label='Cholesky Speedup Factor')
    ax_c2.set_ylabel('Speedup Factor $(\\times)$', fontsize=10, fontweight='bold', color=c_amber)
    ax_c2.tick_params(axis='y', labelcolor=c_amber)
    ax_c2.set_ylim(0.5, 4.0)

    # Panel (d): Derivation Time & Generated C Code Size
    ax_d = axs[1, 1]
    t_derivs = [r['t_deriv_total'] for r in results]
    c_sizes = [r['c_size_kb'] for r in results]
    ax_d.plot(dofs, t_derivs, 's-', color='#00897B', lw=2.0, ms=6, label='Derivation Time ($t_{\\mathrm{deriv}}$, s)')
    ax_d.set_xlabel('Degrees of Freedom ($n$)', fontsize=10, fontweight='bold')
    ax_d.set_ylabel('Offline Derivation Time (seconds)', fontsize=10, fontweight='bold', color='#00897B')
    ax_d.tick_params(axis='y', labelcolor='#00897B')
    ax_d.set_title('(d) Derivation Time & Emitted C-Code Footprint', fontsize=10, fontweight='bold')
    ax_d.grid(True, ls='--', alpha=0.5)
    ax_d.set_ylim(0, 0.8)

    # Twin axis for C code size
    ax_d2 = ax_d.twinx()
    ax_d2.plot(dofs, c_sizes, '^--', color='#5E35B1', lw=1.8, ms=6, label='C Source Size (KB)')
    ax_d2.set_ylabel('Emitted ANSI C Size (KB)', fontsize=10, fontweight='bold', color='#5E35B1')
    ax_d2.tick_params(axis='y', labelcolor='#5E35B1')
    ax_d2.set_ylim(0, 1200)

    # Save to manuscript figures folder
    manuscript_figs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Full pipeline Manuscript", "figures"))
    os.makedirs(manuscript_figs_dir, exist_ok=True)
    out_pdf = os.path.join(manuscript_figs_dir, "fig6_scalability_and_complexity.pdf")
    out_png = os.path.join(manuscript_figs_dir, "fig6_scalability_and_complexity.png")

    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated Figure 6: {out_pdf}")
    print(f"Generated Figure 6: {out_png}")


if __name__ == "__main__":
    main()
