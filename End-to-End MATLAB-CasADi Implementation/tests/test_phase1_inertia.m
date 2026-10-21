%% test_phase1_inertia.m
% =========================================================================
% UNIT TEST — Phase 1: Inertial Parameter Extraction (P_Ci and I)
% =========================================================================
% Validates that the new functional pipeline:
%   urdf_parse → urdf_build_chain → urdf_extract_dh → urdf_extract_inertia
% produces mass, CoM (P_Ci), and inertia tensor (I) values that are
% NUMERICALLY IDENTICAL to those produced by the legacy class:
%   GenerateDhParams.get_dynamics_parameters()   (My Implementation version)
%
% This cross-validation is the strongest possible test because:
%   (1) It uses the EXACT same URDF as input to both pipelines.
%   (2) The legacy class is the implementation validated in the conference paper.
%   (3) Any error in frame convention, P_Ci formula, or I rotation will
%       produce a numerically detectable discrepancy.
%
% =========================================================================
% TOLERANCE:
%   Mass:         absolute error < 1e-10 kg     (exact double equality)
%   CoM (P_Ci):   absolute error < 1e-7  m      (sub-micrometer)
%   Inertia (I):  absolute error < 1e-10 kg·m²  (near-exact)
%
% =========================================================================
% HOW TO RUN:
%   cd to the "End-to-End MATLAB-CasADi Implementation" folder, then:
%     run tests/test_phase1_inertia.m
%   Or add the folder to the path and call it directly.
%
% PASS criterion: All assertions succeed and "ALL TESTS PASSED" is printed.
% =========================================================================

clear; clc;

% -------------------------------------------------------------------------
% 0. Setup paths
% -------------------------------------------------------------------------
this_dir   = fileparts(mfilename('fullpath'));
parent_dir = fileparts(this_dir);
root_dir   = fileparts(parent_dir);
legacy_dir = fullfile(root_dir, 'New Formulation Eng. Mariam', 'My Implementation');

addpath(fullfile(parent_dir, 'parsing'));
addpath(legacy_dir);

urdf_file = fullfile(root_dir, 'universalUR5.urdf');

fprintf('============================================================\n');
fprintf('  Phase 1 Unit Test — Inertial Parameter Extraction\n');
fprintf('  Robot: %s\n', urdf_file);
fprintf('============================================================\n\n');

% =========================================================================
% STEP 1 — Run the NEW functional pipeline
% =========================================================================
robot   = urdf_parse(urdf_file);
chain   = urdf_build_chain(robot, false);
dh      = urdf_extract_dh(chain, false);
inertia = urdf_extract_inertia(robot, chain, dh, true);   % verbose for display

N = chain.n;

% =========================================================================
% STEP 2 — Run the LEGACY GenerateDhParams pipeline for comparison
% =========================================================================
fprintf('\n--- Running legacy GenerateDhParams for cross-validation ---\n');
legacy_ext = GenerateDhParams(urdf_file);
legacy_ext.verbose = false;
[DHpram_leg, P_Ci_leg, I_leg, m_leg, ~] = legacy_ext.get_dynamics_parameters();
fprintf('  Legacy pipeline complete (%d joints extracted).\n\n', size(DHpram_leg, 1));

% =========================================================================
% TEST A — Mass vector comparison
% =========================================================================
fprintf('--- Test A: Mass vector m comparison ---\n');
TOL_m = 1e-10;   % exact double precision match expected

test_A_pass = true;
m_err = abs(inertia.m - m_leg);
max_m_err = max(m_err);

if max_m_err > TOL_m
    fprintf('  [FAIL] Mass vector mismatch. Max error = %.4e kg\n', max_m_err);
    fprintf('         New m:    '); fprintf('%.6f  ', inertia.m'); fprintf('\n');
    fprintf('         Legacy m: '); fprintf('%.6f  ', m_leg');    fprintf('\n');
    test_A_pass = false;
else
    fprintf('  [PASS] Mass vector matches legacy (max error = %.2e kg).\n', max_m_err);
    fprintf('         m = ['); fprintf('%.4f  ', inertia.m'); fprintf('] kg\n');
end

if test_A_pass
    fprintf('  [PASS] Test A complete.\n\n');
else
    fprintf('  [FAIL] Test A complete.\n\n');
end

% =========================================================================
% TEST B — Center of Mass vector P_Ci comparison
% =========================================================================
% P_Ci(i,:) is the CoM of link i measured FROM the previous DH origin,
% expressed IN the current DH frame orientation.
% See urdf_extract_inertia.m for the full derivation.
% =========================================================================
fprintf('--- Test B: CoM vector P_Ci comparison ---\n');
TOL_p = 1e-7;   % 0.1 micrometer — far below any physical uncertainty

test_B_pass = true;
P_err = abs(inertia.P_Ci - P_Ci_leg);
max_P_err = max(max(P_err));

if max_P_err > TOL_p
    test_B_pass = false;
    fprintf('  [FAIL] P_Ci mismatch. Max element-wise error = %.4e m\n', max_P_err);

    % Print detailed per-link comparison
    fprintf('\n  Per-link P_Ci comparison:\n');
    fprintf('  %-3s | %-30s | %-30s | %-12s\n', '#', 'New P_Ci (m)', 'Legacy P_Ci (m)', 'Max Error');
    fprintf('  %s\n', repmat('-', 1, 82));
    for i = 1:N
        row_err = max(abs(inertia.P_Ci(i,:) - P_Ci_leg(i,:)));
        status = 'PASS'; if row_err > TOL_p, status = 'FAIL'; end
        fprintf('  %-3d | [%8.5f, %8.5f, %8.5f] | [%8.5f, %8.5f, %8.5f] | %.2e  [%s]\n', ...
                i, inertia.P_Ci(i,1), inertia.P_Ci(i,2), inertia.P_Ci(i,3), ...
                P_Ci_leg(i,1), P_Ci_leg(i,2), P_Ci_leg(i,3), row_err, status);
    end
else
    fprintf('  [PASS] P_Ci matches legacy (max element-wise error = %.2e m).\n', max_P_err);

    % Print the P_Ci table even on pass (useful for manual inspection)
    fprintf('\n  P_Ci table (CoM in DH frames):\n');
    fprintf('  %-3s | %-12s | %-12s | %-12s\n', '#', 'x (m)', 'y (m)', 'z (m)');
    fprintf('  %s\n', repmat('-', 1, 46));
    for i = 1:N
        fprintf('  %-3d | %12.6f | %12.6f | %12.6f\n', ...
                i, inertia.P_Ci(i,1), inertia.P_Ci(i,2), inertia.P_Ci(i,3));
    end
end

if test_B_pass
    fprintf('  [PASS] Test B complete.\n\n');
else
    fprintf('  [FAIL] Test B complete.\n\n');
end

% =========================================================================
% TEST C — Inertia tensor I comparison
% =========================================================================
% I(:,:,i) is the 3x3 rotational inertia tensor of link i, expressed in
% the orientation of DH frame {F_i}.  Must be symmetric and PSD.
% =========================================================================
fprintf('--- Test C: Inertia tensor I comparison ---\n');
TOL_I = 1e-9;   % near-exact match (tensors are rotated from same URDF values)

test_C_pass = true;
max_I_err   = 0;

% Use a differently-named variable to avoid shadowing the 3D I_leg array
for i = 1:N
    I_new    = inertia.I(:,:,i);        % new pipeline: slice i of 3D tensor
    I_leg_i  = I_leg(:,:,i);            % legacy pipeline: slice i (renamed to avoid shadowing)
    I_err    = max(max(abs(I_new - I_leg_i)));
    max_I_err = max(max_I_err, I_err);

    if I_err > TOL_I
        fprintf('  [FAIL] Link %d inertia tensor mismatch (max error = %.4e).\n', i, I_err);
        fprintf('         New I(:,:,%d):\n', i);    disp(I_new);
        fprintf('         Legacy I(:,:,%d):\n', i); disp(I_leg_i);
        test_C_pass = false;
    end
end

if test_C_pass
    fprintf('  [PASS] All %d inertia tensors match legacy (max error = %.2e kg·m²).\n', ...
            N, max_I_err);
end

% Additional check: symmetry and positive semi-definiteness of each I tensor
fprintf('\n--- Test C (sub-check): Symmetry and PSD of inertia tensors ---\n');
for i = 1:N
    I_new = inertia.I(:,:,i);

    % Symmetry check: I should equal its own transpose
    sym_err = max(max(abs(I_new - I_new')));
    if sym_err > 1e-14
        fprintf('  [FAIL] Link %d: I is not symmetric (max off-diagonal error = %.2e)\n', i, sym_err);
        test_C_pass = false;
    end

    % PSD check: all eigenvalues must be >= 0
    % (for rigid bodies, I is always at least PSD; strict PD for non-degenerate bodies)
    eigs = eig(I_new);
    if any(eigs < -1e-10)
        fprintf('  [FAIL] Link %d: I is not positive semi-definite (min eigenvalue = %.4e)\n', ...
                i, min(eigs));
        test_C_pass = false;
    end
end

if test_C_pass
    fprintf('  [PASS] All inertia tensors are symmetric and positive semi-definite.\n');
end

if test_C_pass
    fprintf('  [PASS] Test C complete.\n\n');
else
    fprintf('  [FAIL] Test C complete.\n\n');
end

% =========================================================================
% TEST D — DHpram format check (sigma, column order)
% =========================================================================
% The DHpram matrix must match the legacy output exactly.
% Column order: [a, alpha, theta_offset, d, sigma]
% =========================================================================
fprintf('--- Test D: DHpram format cross-validation ---\n');
TOL_dh = 1e-7;

test_D_pass = true;
DHpram_new  = dh.DHpram;
dh_err      = max(max(abs(DHpram_new - DHpram_leg)));

if dh_err > TOL_dh
    fprintf('  [FAIL] DHpram mismatch (max error = %.4e)\n', dh_err);
    fprintf('         New DHpram:\n');    disp(DHpram_new);
    fprintf('         Legacy DHpram:\n'); disp(DHpram_leg);
    test_D_pass = false;
else
    fprintf('  [PASS] DHpram matches legacy output (max error = %.2e).\n', dh_err);
    fprintf('\n  DHpram (DH_dyn format: [a, alpha, theta_offset, d, sigma]):\n');
    fprintf('  %-3s | %-10s | %-10s | %-12s | %-10s | %-5s\n', ...
            '#', 'a (m)', 'alpha', 'theta_off', 'd (m)', 'sigma');
    fprintf('  %s\n', repmat('-', 1, 60));
    for i = 1:N
        fprintf('  %-3d | %10.6f | %10.6f | %12.6f | %10.6f | %5.0f\n', ...
                i, DHpram_new(i,1), DHpram_new(i,2), ...
                DHpram_new(i,3), DHpram_new(i,4), DHpram_new(i,5));
    end
end

if test_D_pass
    fprintf('  [PASS] Test D complete.\n\n');
else
    fprintf('  [FAIL] Test D complete.\n\n');
end

% =========================================================================
% SUMMARY
% =========================================================================
all_passed = test_A_pass && test_B_pass && test_C_pass && test_D_pass;

fprintf('============================================================\n');
if all_passed
    fprintf('  >>> ALL TESTS PASSED <<<\n');
else
    fprintf('  >>> SOME TESTS FAILED — see details above <<<\n');
end
fprintf('============================================================\n');

assert(all_passed, 'Phase 1 inertial parameter unit tests failed. See console output.');
