"""Console dashboard table and CSV/JSON export for verification reports."""

from __future__ import annotations
import os
import json
import csv
from urdf2dt.containers import VerificationReport, StageResult


def print_verification_dashboard(report: VerificationReport) -> None:
    """Print an ANSI console dashboard summarizing cross-engine verification results."""
    print("\n" + "=" * 80)
    print("                     URDF2DT-PY VERIFICATION DASHBOARD                       ")
    print("=" * 80)
    print(f"  Overall Status         : {'>>> ALL PASSED <<<' if report.all_passed else '>>> SOME CHECKS FAILED <<<'}")
    print(f"  Max FK Position Error  : {report.max_fk_pos_err:10.4e} m")
    print(f"  Max FK Quaternion Error: {report.max_fk_quat_err:10.4e} rad")
    print(f"  Max Jacobian Error     : {report.max_jac_err:10.4e}")
    print(f"  Max Dynamic Error      : {report.max_dyn_err:10.4e} N*m")
    print("-" * 80)
    print(f"{'Stage':<6} {'Joint Name':<22} {'Link Name':<18} {'RMS Err (N*m)':<15} {'Max Err (N*m)':<15} {'Status':<6}")
    print("-" * 80)

    for st in report.stage_results:
        status_str = "PASS" if st.passed else "FAIL"
        print(f"{st.stage:<6} {st.joint_name:<22} {st.link_name:<18} {st.rms_err:<15.4e} {st.max_err:<15.4e} {status_str:<6}")

    print("=" * 80 + "\n")


def export_verification_report(
    report: VerificationReport,
    output_dir: str = "outputs",
    prefix: str = "verification"
) -> dict[str, str]:
    """Export verification report to JSON and CSV formats."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{prefix}_report.json")
    csv_path = os.path.join(output_dir, f"{prefix}_stages.csv")

    # JSON export
    data = {
        "all_passed": report.all_passed,
        "max_fk_pos_err": report.max_fk_pos_err,
        "max_fk_quat_err": report.max_fk_quat_err,
        "max_jac_err": report.max_jac_err,
        "max_dyn_err": report.max_dyn_err,
        "stages": [
            {
                "stage": st.stage,
                "joint_name": st.joint_name,
                "link_name": st.link_name,
                "passed": st.passed,
                "max_err": st.max_err,
                "rms_err": st.rms_err,
                "peak_torque": st.peak_torque,
            }
            for st in report.stage_results
        ]
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # CSV export
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Stage", "Joint_Name", "Link_Name", "Passed", "RMS_Error_Nm", "Max_Error_Nm", "Peak_Torque_Nm"])
        for st in report.stage_results:
            writer.writerow([st.stage, st.joint_name, st.link_name, st.passed, st.rms_err, st.max_err, st.peak_torque])

    return {"json": json_path, "csv": csv_path}
