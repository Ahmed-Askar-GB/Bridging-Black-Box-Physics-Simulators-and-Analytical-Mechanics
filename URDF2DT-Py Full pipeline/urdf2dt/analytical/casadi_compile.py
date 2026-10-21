"""Model compilation, function serialization, and C/C++ code generation."""

from __future__ import annotations
import os
import casadi as cs
from urdf2dt.containers import CasadiModel


def casadi_compile(
    model: CasadiModel,
    robot_name: str = "robot",
    output_dir: str = "compilation",
    gen_c: bool = True,
    verbose: bool = False
) -> dict:
    """Compile and export CasADi symbolic models into standalone C code and callable archives."""
    os.makedirs(output_dir, exist_ok=True)

    c_filename = f"{robot_name}_dynamics.c"
    c_filepath = os.path.join(output_dir, c_filename)
    h_filepath = os.path.join(output_dir, f"{robot_name}_dynamics.h")

    results = {
        "c_filepath": c_filepath,
        "h_filepath": h_filepath,
        "functions": {},
    }

    if gen_c:
        cg = cs.CodeGenerator(c_filename)
        cg.add(model.fk_fun)
        cg.add(model.fk_ee_fun)
        cg.add(model.J_fun)
        cg.add(model.Jdot_fun)
        cg.add(model.M_fun)
        cg.add(model.C_fun)
        cg.add(model.G_fun)
        cg.add(model.tau_fun)
        cg.add(model.forward_dyn_fun)

        curr_dir = os.getcwd()
        try:
            os.chdir(output_dir)
            cg.generate()
        finally:
            os.chdir(curr_dir)

        # Write matching C header file
        with open(h_filepath, "w", encoding="utf-8") as f:
            guard = f"{robot_name.upper()}_DYNAMICS_H"
            f.write(f"#ifndef {guard}\n#define {guard}\n\n")
            f.write("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n")
            f.write("#include <casadi/mem.h>\n\n")
            f.write(f"/* Exported CasADi functions for {robot_name} */\n")
            for fn_name in ["fk_fun", "fk_ee_fun", "J_fun", "Jdot_fun", "M_fun", "C_fun", "G_fun", "tau_fun", "forward_dyn_fun"]:
                f.write(f"int {fn_name}(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);\n")
            f.write("\n#ifdef __cplusplus\n}\n#endif\n\n")
            f.write(f"#endif /* {guard} */\n")

        if verbose:
            print(f"Generated embedded C code: {c_filepath}")
            print(f"Generated embedded C header: {h_filepath}")

    return results
