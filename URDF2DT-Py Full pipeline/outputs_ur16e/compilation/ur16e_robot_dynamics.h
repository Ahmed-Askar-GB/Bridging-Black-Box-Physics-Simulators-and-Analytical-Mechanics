#ifndef UR16E_ROBOT_DYNAMICS_H
#define UR16E_ROBOT_DYNAMICS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <casadi/mem.h>

/* Exported CasADi functions for ur16e_robot */
int fk_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int fk_ee_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int J_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int Jdot_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int M_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int C_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int G_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int tau_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int forward_dyn_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);

#ifdef __cplusplus
}
#endif

#endif /* UR16E_ROBOT_DYNAMICS_H */
