function [dt, compiled_model, dt_smimport_sensed] = step4_build_models(robot, chain, inertia, fk, jac, dyn, joint_signs, cfg)
% STEP4_BUILD_MODELS  Generate and save triple Simulink digital twin models:
%   1. DH-based Multibody Digital Twin (from dt_build)
%   2. Native raw Simscape Multibody model (from smimport)
%   3. Native smimport model instrumented with EE reference frame & sensors.
%
% =========================================================================
% INPUTS:
%   robot               (struct) Parsed URDF description.
%   chain               (struct) Active kinematic chain.
%   inertia             (struct) Inertial parameter definitions.
%   fk, jac, dyn        (struct) Symbolic kinematics/dynamics objects.
%   joint_signs         [Nx1]    Actuation sign vector.
%   cfg                 (struct) Configuration struct.
%
% OUTPUTS:
%   dt                  (struct) DH Digital Twin model struct from dt_build.
%   compiled_model      (struct) Compiled CasADi analytical model metadata.
%   dt_smimport_sensed  (struct) Instrumented smimport model struct.
% =========================================================================

    fprintf('\n>>> STEP 4: Creating & Saving Triple Simulink Models ...\n');
    n = chain.n;

    % -----------------------------------------------------------------------
    % 4A. Model 1: Multibody Digital Twin from DH Parameters
    % -----------------------------------------------------------------------
    fprintf('  [Model 1/3] Creating DH-based Digital Twin ("%s") ...\n', cfg.dh_model_name);
    if bdIsLoaded(cfg.dh_model_name)
        close_system(cfg.dh_model_name, 0);
    end

    compiled_model = casadi_compile(fk, jac, dyn, robot.name, cfg.output_dir, ...
                                    struct('gen_c', false, 'verbose', false));

    opts_build = struct();
    opts_build.model_name = cfg.dh_model_name;
    opts_build.output_dir = cfg.output_dir;
    opts_build.stop_time  = sprintf('%.2f', cfg.tol.t_dur);
    opts_build.mesh_dir   = cfg.mesh_dir;
    opts_build.verbose    = false;

    dt = dt_build(robot, chain, inertia, opts_build);
    dt = dt_add_ee_frame(dt, chain, struct('model', compiled_model, 'verbose', false, 'save_model', true));
    dt = dt_add_sensors(dt, struct('verbose', false, 'save_model', true));

    % Trajectory excitation: smooth 5th-order polynomial motion across all DOFs
    t_vec = linspace(0, cfg.tol.t_dur, 1001);
    q0 = zeros(n, 1);
    q1 = 0.35 * sin((1:n)' * 1.1);
    [q_traj, dq_traj, ddq_traj] = traj_poly5(q0, q1, cfg.tol.t_dur, t_vec);
    traj_struct = struct('time', t_vec, 'q', q_traj, 'dq', dq_traj, 'ddq', ddq_traj);
    dt = dt_add_trajectory(dt, traj_struct, struct('joint_signs', joint_signs, 'verbose', false, 'save_model', true));

    save_system(cfg.dh_model_name, cfg.dh_model_file);
    fprintf('  [SAVED] Model 1 (DH Multibody) saved to: %s\n', cfg.dh_model_file);

    % -----------------------------------------------------------------------
    % 4B. Model 2: Native Simscape Multibody Model from URDF via smimport
    % -----------------------------------------------------------------------
    fprintf('  [Model 2/3] Checking native smimport model ("%s") ...\n', cfg.smimport_model_name);
    if ~exist(cfg.smimport_model_file, 'file')
        fprintf('  Importing native Simscape model via smimport ...\n');
        orig_dir = pwd;
        cd(cfg.output_dir);
        try
            if bdIsLoaded(cfg.smimport_model_name)
                close_system(cfg.smimport_model_name, 0);
            end
            smimport(cfg.urdf_path, 'ModelName', cfg.smimport_model_name);
            save_system(cfg.smimport_model_name, cfg.smimport_model_file);
            close_system(cfg.smimport_model_name, 0);
            cd(orig_dir);
        catch ex
            cd(orig_dir);
            rethrow(ex);
        end
    end
    fprintf('  [CONFIRMED] Model 2 (Raw smimport) present at: %s\n', cfg.smimport_model_file);

    % -----------------------------------------------------------------------
    % 4C. Model 3: Native smimport Model + EE Reference Frame + Sensors + Trajectory
    % -----------------------------------------------------------------------
    fprintf('  [Model 3/3] Instrumenting smimport model with EE frame, sensors & trajectory ("%s") ...\n', ...
            cfg.smimport_sensed_model_name);
    opts_inst = struct();
    opts_inst.verbose     = false;
    opts_inst.save_model  = true;
    opts_inst.traj        = traj_struct;
    opts_inst.joint_signs = ones(n, 1); % Native URDF joint axes
    dt_smimport_sensed = dt_instrument_smimport(cfg.smimport_model_file, ...
                                                cfg.smimport_sensed_model_file, ...
                                                chain, compiled_model, opts_inst);
    fprintf('  [SAVED] Model 3 (smimport + EE frame + Sensors + Trajectory) saved to: %s\n', ...
            cfg.smimport_sensed_model_file);

end
