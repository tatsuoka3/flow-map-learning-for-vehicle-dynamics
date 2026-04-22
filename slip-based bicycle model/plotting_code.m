clear; clc; close all;

%% settings
dt = 0.01;
tspan = 0:dt:5;
t_ctrl = tspan(:);      % scalar time for plotting controls
eps_rel = 1e-8;

x0 = [1; 1; 1; 1; 0; 0];

true_files = { ...
    'test1_true.mat', ...
    'test2_true.mat', ...
    'test3_true.mat'};

prior_files = { ...
    'test1_prior.mat', ...
    'test2_prior.mat', ...
    'test3_prior.mat'};

pred_files = { ...
    'corrected_pred_last_layer_T=1000_ntest=1_1000_traj.mat', ...
    'corrected_pred_last_layer_T=1000_ntest=2_1000_traj.mat', ...
    'corrected_pred_last_layer_T=1000_ntest=3_1000_traj.mat'};

case_titles = {'Test Case 1','Test Case 2','Test Case 3'};

opts = odeset('RelTol',1e-6,'AbsTol',1e-8);

%% true/prior data generation
for case_id = 1:3
    [t_true, X_true] = ode45(@(t,x) damped_bicycle_true(t, x, case_id), tspan, x0, opts);
    [t_prior, X_prior] = ode45(@(t,x) damped_bicycle_prior(t, x, case_id), tspan, x0, opts);

    X = X_true;
    save(true_files{case_id}, 'X', 't_true');

    X_prior_save = X_prior;
    save(prior_files{case_id}, 'X_prior_save', 't_prior');
end

%% load all
cases = cell(1,3);

for k = 1:3
    %% controls: plot directly from the formulas you specified
    [u_plot, delta_plot] = controls_for_case(t_ctrl, k);

    %% true
    Strue = load(true_files{k});
    if isfield(Strue, 'X')
        X_true = Strue.X;
    else
        X_true = get_numeric_array(Strue);
    end

    %% prior
    Sprior = load(prior_files{k});
    if isfield(Sprior, 'X_prior_save')
        X_prior = Sprior.X_prior_save;
    elseif isfield(Sprior, 'X_prior')
        X_prior = Sprior.X_prior;
    else
        X_prior = get_numeric_array(Sprior);
    end

    %% corrected prediction
    Spred = load(pred_files{k});
    pred_raw = get_numeric_array(Spred);
    pred_row = squeeze_first_sample(pred_raw);
    X_pred = reshape_state_traj(pred_row, 6);

    %% trim to common length
    Nt = min([size(X_true,1), size(X_prior,1), size(X_pred,1)]);
    X_true  = X_true(1:Nt,:);
    X_prior = X_prior(1:Nt,:);
    X_pred  = X_pred(1:Nt,:);
    t = (0:Nt-1)' * dt;

    %% modified relative error
    rel_err = abs(X_true - X_pred) ./ (abs(X_true) + eps_rel);

    %% store
    cases{k}.t_ctrl = t_ctrl;
    cases{k}.u = u_plot;
    cases{k}.delta = delta_plot;
    cases{k}.t = t;
    cases{k}.true = X_true;
    cases{k}.prior = X_prior;
    cases{k}.pred = X_pred;
    cases{k}.relerr = rel_err;
end

%% plot
fig = figure('Color','w','Position',[40 20 1500 1900]);
tiledlayout(5,3,'TileSpacing','compact','Padding','compact');

for k = 1:3
    C = cases{k};

    %% Row 1: control inputs
    nexttile((1-1)*3 + k);
    plot(C.t_ctrl, C.u, 'r-', 'LineWidth', 1.6); hold on;
    plot(C.t_ctrl, C.delta, 'b-', 'LineWidth', 1.6);
    grid on;
    xlabel('Time (s)');
    ylabel('Control Value');
    title(case_titles{k});
    legend('$u$','$\delta$','Interpreter','latex','Location','best');

    %% Row 2: x-y positions
    nexttile((2-1)*3 + k);
    plot(C.prior(:,1), C.prior(:,2), '--', 'Color', [1 0 0], 'LineWidth', 1.2); hold on;
    plot(C.true(:,1),  C.true(:,2),  '-',  'Color', [1 0 0], 'LineWidth', 1.8);
    plot(C.pred(:,1),  C.pred(:,2),  'o', ...
        'Color', [1 0.35 0.35], 'MarkerFaceColor', [1 0.35 0.35], ...
        'MarkerSize', 4);
    grid on;
    xlabel('$x$ position (m)','Interpreter','latex');
    ylabel('$y$ position (m)','Interpreter','latex');
    legend('Prior','True','Pred','Location','best');

    %% Row 3: psi, vy, omega
    nexttile((3-1)*3 + k);
    plot(C.t, C.prior(:,3), '--', 'Color', [0.45 0 0.55], 'LineWidth', 1.2); hold on;
    plot(C.t, C.true(:,3),  '-',  'Color', [0.45 0 0.55], 'LineWidth', 1.8);
    plot(C.t, C.pred(:,3),  'o', ...
        'Color', [0.45 0 0.55], 'MarkerFaceColor', [0.45 0 0.55], ...
        'MarkerSize', 4);

    plot(C.t, C.prior(:,5), '--', 'Color', [1 0.2 0.2], 'LineWidth', 1.2);
    plot(C.t, C.true(:,5),  '-',  'Color', [1 0.2 0.2], 'LineWidth', 1.8);
    plot(C.t, C.pred(:,5),  's', ...
        'Color', [1 0.35 0.35], 'MarkerFaceColor', [1 0.35 0.35], ...
        'MarkerSize', 4);

    plot(C.t, C.prior(:,6), '--', 'Color', [0.60 0.35 0.10], 'LineWidth', 1.2);
    plot(C.t, C.true(:,6),  '-',  'Color', [0.60 0.35 0.10], 'LineWidth', 1.8);
    plot(C.t, C.pred(:,6),  's', ...
        'Color', [0.60 0.35 0.10], 'MarkerFaceColor', [0.60 0.35 0.10], ...
        'MarkerSize', 4);

    grid on;
    xlabel('Time (s)');
    ylabel('$\psi,\, v_y,\, \omega$','Interpreter','latex');
    legend( ...
        'Prior \psi','True \psi','Pred \psi', ...
        'Prior v_y','True v_y','Pred v_y', ...
        'Prior \omega','True \omega','Pred \omega', ...
        'Location','best');

    %% Row 4: vx
    nexttile((4-1)*3 + k);
    plot(C.t, C.prior(:,4), '--', 'Color', [0 0.5 0], 'LineWidth', 1.2); hold on;
    plot(C.t, C.true(:,4),  '-',  'Color', [0 0.5 0], 'LineWidth', 1.8);
    plot(C.t, C.pred(:,4),  's', ...
        'Color', [0 0.5 0], 'MarkerFaceColor', [0 0.5 0], ...
        'MarkerSize', 4);
    grid on;
    xlabel('Time (s)');
    ylabel('$v_x$','Interpreter','latex');
    legend('Prior $v_x$','True $v_x$','Pred $v_x$', ...
        'Interpreter','latex','Location','best');

    %% Row 5: modified relative errors
    nexttile((5-1)*3 + k);
    semilogy(C.t, max(C.relerr(:,1),1e-12), '-', 'Color', [1 0 0], 'LineWidth', 1.5); hold on;
    semilogy(C.t, max(C.relerr(:,2),1e-12), '-', 'Color', [0 0 1], 'LineWidth', 1.5);
    semilogy(C.t, max(C.relerr(:,3),1e-12), '-', 'Color', [0.45 0 0.55], 'LineWidth', 1.5);
    semilogy(C.t, max(C.relerr(:,4),1e-12), '-', 'Color', [0 0.5 0], 'LineWidth', 1.5);
    semilogy(C.t, max(C.relerr(:,5),1e-12), '-', 'Color', [0.8 0.1 0.1], 'LineWidth', 1.5);
    semilogy(C.t, max(C.relerr(:,6),1e-12), '-', 'Color', [0.60 0.35 0.10], 'LineWidth', 1.5);
    grid on;
    xlabel('Time (s)');
    ylabel('Relative Error');
    legend('$x$','$y$','$\psi$','$v_x$','$v_y$','$\omega$', ...
        'Interpreter','latex','Location','best');
    ylim([1e-12, 1e1]);
end

%% helpers
function arr = get_numeric_array(S)
    fns = fieldnames(S);
    arr = [];
    for i = 1:numel(fns)
        val = S.(fns{i});
        if isnumeric(val)
            arr = val;
            return;
        end
    end
    error('No numeric array found in loaded .mat file.');
end

function row = squeeze_first_sample(arr)
    if isvector(arr)
        row = arr(:).';
    else
        row = arr(1,:);
    end
end

function X = reshape_state_traj(row, d)
    row = row(:).';
    n = floor(numel(row)/d);
    row = row(1:n*d);
    X = reshape(row, d, n).';
end

%% model definitions
function dxdt = damped_bicycle_true(t, x, case_id)
    [u, delta] = controls_for_case(t, case_id);

    m  = 2.5;   Iz = 0.015;
    Cf = 2.0;   Cr = 2.0;
    Lf = 0.1;   Lr = 0.05;
    bu = 4.0;   bdelta = 0.4;

    psi = x(3);
    vx  = x(4);
    vy  = x(5);
    w   = x(6);

    Fx  = m * bu * u;
    Fyf = -Cf * ( bdelta*delta - (vy + Lf*w)/vx );
    Fyr = -Cr * ( (vy + Lr*w)/vx );

    dx1 = vx*cos(psi) - vy*sin(psi);
    dx2 = vx*sin(psi) + vy*cos(psi);
    dx3 = w;
    dx4 = Fx/m - w*vy;
    dx5 = (Fyf - Fyr)/m - w*vx;
    dx6 = (Lf*Fyf - Lr*Fyr)/Iz;

    dxdt = [dx1; dx2; dx3; dx4; dx5; dx6];
end

function dxdt = damped_bicycle_prior(t, x, case_id)
    [u, delta] = controls_for_case(t, case_id);

    m  = 2.5;   Iz = 0.015;
    Cf = 2.0;   Cr = 2.0;
    Lf = 0.082; Lr = 0.098;
    bu = 5.0;   bdelta = 0.4;

    psi = x(3);
    vx  = x(4);
    vy  = x(5);
    w   = x(6);

    Fx  = m * bu * u;
    Fyf = -Cf * ( bdelta*delta - (vy + Lf*w)/vx );
    Fyr = -Cr * ( (vy + Lr*w)/vx );

    dx1 = vx*cos(psi) - vy*sin(psi);
    dx2 = vx*sin(psi) + vy*cos(psi);
    dx3 = w;
    dx4 = Fx/m - w*vy;
    dx5 = (Fyf - Fyr)/m - w*vx;
    dx6 = (Lf*Fyf - Lr*Fyr)/Iz;

    dxdt = [dx1; dx2; dx3; dx4; dx5; dx6];
end

function [u, delta] = controls_for_case(t, case_id)
    switch case_id
        case 1
            u     = 0.2 + 0.2*cos(0.3*t);
            delta = 0.5*sin(t);
        case 2
            u     = 0.4*sin(0.5*t).^2;
            delta = 0.02*cos(0.25*t).^2;
        case 3
            u     = 0.4*tanh(0.5*t);
            delta = (0.3*sin(0.5*t))./(0.1*t + 1);
        otherwise
            error('Unknown case_id.');
    end
end