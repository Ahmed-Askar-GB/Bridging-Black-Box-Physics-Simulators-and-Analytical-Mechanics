"""Quick launcher to run the UR16e 3D MuJoCo Animation and Diagnostic Dashboard."""

import argparse
import os
import sys

# Change directory to script folder and execute
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from animate_ur16e_interactive import animate_ur16e
from tests.test_ur16e_pipeline_with_Animation import run_ur16e_verification_and_animation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR16e 3D Animation Launcher")
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Optional path to URDF model file",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run quantitative verification dashboard with multi-panel matplotlib plots",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Enable continuous trajectory replay loop",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="kinematic",
        choices=["kinematic", "control"],
        help="Initial mode: 'kinematic' (Inverse Dynamics Replay) or 'control' (Computed Torque Control)",
    )
    parser.add_argument(
        "--duration",
        "-T",
        type=float,
        default=6.0,
        help="Trajectory duration in seconds (default: 6.0)",
    )

    args = parser.parse_args()

    print("\n=======================================================")
    print("  LAUNCHING UNIVERSAL ROBOTS UR16E 3D ANIMATION VIEWER ")
    print("=======================================================\n")

    if args.dashboard:
        # Run stage 4 verification with plots
        run_ur16e_verification_and_animation(
            t_dur=args.duration,
            dt=0.002,
            headless=False,
            show_plots=True,
            verbose=True,
        )
    else:
        # Run interactive viewer (persistent window, optimal camera, live HUD, hotkey re-animation)
        animate_ur16e(
            urdf_path=args.urdf,
            t_dur=args.duration,
            dt=0.002,
            initial_mode=args.mode,
            loop=args.loop,
        )
