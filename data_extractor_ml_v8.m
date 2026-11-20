%% data_extractor_28_types_v8.m   (for fault_detection_system_v8)
%
% One simulation per fault type (28 runs by default) for the v8 model.
% For every run it
%   1. adds ONE row to fault_detection_features_28_v8.csv
%   2. (optional, SAVE_IMAGES = true) saves one PNG per Scope block:
%        scope_plots/<fault_type>/Scope_Ia_Ib_Ic.png ... (14 files)
%   3. (optional, SAVE_IMAGES = true) saves one 4x4 overview PNG per fault type:
%        overview_plots/<fault_type>_overview.png
%
% CHANGES vs data_extractor_28_types_v6:
%   - Targets fault_detection_system_v8 and sets the v8 parameters on every run:
%       Sensing_ThreePhase:            SourceL, ReclosDelay
%       Breaker_Contactor_Plant_3ph:   FeedbackEnable, BreakerDelay
%   - FEEDBACK = 0 (default) keeps the breaker latch but does NOT open the feeder, so every feature
%     is measured over the whole 1 s fault window, like the v6 extractor (bolted-fault characterisation).
%     FEEDBACK = 1 gives the protected behaviour: the current is interrupted about 45 ms after onset,
%     so the *_postfault_* and *_settled features then describe an interrupted feeder (mostly zeros).
%     For protection numbers use the let-through columns, or fault_timeseries_dump_v8.
%   - mean_phase_diff_postfault_* is a CIRCULAR mean (angle of the mean unit phasor). The v8 Phase
%     channel is wrapped to (-pi, pi], and a plain mean of wrapped angles is wrong near +-pi.
%   - NEW columns:
%       rms_I_neutral_settled     rms of Ia+Ib+Ic in the settled window (about 0 for LL, LLL, balanced faults)
%       I0_over_I2_settled        about 0 for LL, about 1 for LG and single-phase faults
%       letthrough_peak_A, letthrough_I2t_A2s
%                                 onset to contact opening; NaN when the contacts never opened
%       peak_I_fixed_window_A, I2t_fixed_window_A2s
%                                 first LT_WINDOW_S after onset, comparable across all rows
%       Irms_recovery_dev_pct     largest deviation of Irms after reclose from the pre-fault Irms
%   - REMOVED: Ia_sample ... Vc_sample (single instantaneous values that depend on the phase angle at
%     the sampling instant, not on the fault) and breaker_open_delay_s (equals the trip delay plus
%     BREAKER_DELAY, so it carries no extra information).
%   - Scope images are OFF by default (SAVE_IMAGES = false) to save run time.
%   - Note: max_cusum_* are now constants (the CUSUM is capped: 5 for the phase channels, 0.5 for the
%     unbalance channels) and carry no information. They are kept so the column set stays familiar.
%   - Trip delays in v8 include a 5 ms confirmation time (trip_latch), so detection_delay_s_agg is
%     about 5.1 to 9.5 ms.
%
% Needs on the path: fault_type_table.m, severity_range_for_mechanism.m, mechanism_to_code.m
%
% USAGE:
%   >> model_code_v8
%   >> data_extractor_28_types_v8

function data_extractor_ml_v8()

tic;

%% ---- settings ----
mdl = 'fault_detection_system_v8';
onset = 0.4;
faultDuration = 1.0;
stopTime = 2.0;

SOURCE_L = 5e-3;                 % H, source inductance (0 = v6 behaviour)
BREAKER_DELAY = 0.04;            % s, trip command to contacts open
RECLOSE_DELAY = 0.1;             % s, fault clearance to the reclose pulse
FEEDBACK = 0;                    % 0 = breaker does not open the feeder (full fault window), 1 = protected

N_REPEATS = 30;                  % rows per class (28 x 30 = 840 rows). Needs >= 5 for 5-fold CV
ONSET_JITTER = true;             % random fault inception angle (0 to 180 deg) for every repeat after the first
USE_RANDOM_SEVERITY = false;     % first repeat = midpoint of range, later repeats = random draw in range
SETTLE_S = 0.1;                  % skip this long after onset for sequence features
RECOVERY_S = 0.1;                % skip this long after the reclose for the recovery check
LT_WINDOW_S = 0.1;               % fixed window after onset for peak_I_fixed_window_A / I2t_fixed_window_A2s
Z_CLIP = 1000;                   % ohms

SAVE_IMAGES = false;             % true = scope PNGs and overview PNGs (slow)
scopeDir = 'scope_plots';
overviewDir = 'overview_plots';
csvName = 'fault_detection_features_ml_v8.csv';
IMG_RES = 130;

if ~bdIsLoaded(mdl), load_system([mdl '.slx']); end
set_param(mdl, 'StopTime', num2str(stopTime));
if SAVE_IMAGES
    if ~exist(scopeDir, 'dir'), mkdir(scopeDir); end
    if ~exist(overviewDir, 'dir'), mkdir(overviewDir); end
end

%% ---- scope groups: identical to the Scope blocks in Logging_Scopes_3ph ----
scopeGroups = struct( ...
    'name', {'Ia_Ib_Ic','Va_Vb_Vc','Irms','Crest','THD','Phase','Cusum','Residual','Mu0','Trip_Breaker_Fault','GABC','Sequence','Cusum_Unbal','Breaker_Currents'}, ...
    'signals', { ...
        {'Ia','Ib','Ic'}, {'Va','Vb','Vc'}, ...
        {'Irms_A','Irms_B','Irms_C'}, {'Crest_A','Crest_B','Crest_C'}, ...
        {'THD_A','THD_B','THD_C'}, {'Phase_A','Phase_B','Phase_C'}, ...
        {'Cusum_A','Cusum_B','Cusum_C'}, {'Residual_A','Residual_B','Residual_C'}, ...
        {'Mu0_A','Mu0_B','Mu0_C'}, ...
        {'trip','breaker_state','fault_flag','reclose'}, {'G','A','B','C'}, ...
        {'I1_seq','I2_seq','I0_seq','Unbal2','Unbal0','UnbalV0'}, ...
        {'Cusum_U2','Cusum_U0','Cusum_V0'}, ...
        {'Ia_post','Ib_post','Ic_post'} });

faultTypes = fault_type_table();
nTypes = size(faultTypes, 1);
rng(1);
rows = {};
seedCounter = 3001;

if FEEDBACK ~= 0
    fprintf('NOTE: FEEDBACK = 1, features describe the interrupted feeder (see header).\n');
end

for ft = 1:nTypes
    label = faultTypes{ft,1};
    mechanism = faultTypes{ft,2};
    G = faultTypes{ft,3}; A = faultTypes{ft,4}; B = faultTypes{ft,5}; C = faultTypes{ft,6};
    sr = severity_range_for_mechanism(mechanism);

    for rep = 1:N_REPEATS
        if rep == 1 && ~USE_RANDOM_SEVERITY
            severity = mean(sr);
        else
            severity = sr(1) + rand() * (sr(2) - sr(1));
        end
        seed = seedCounter; seedCounter = seedCounter + 1;
        onsetK = onset;
        if ONSET_JITTER && rep > 1
            onsetK = onset + 1e-4 * randi([0 99]);   % 0 to 9.9 ms = half a cycle at 50 Hz, on the 0.1 ms grid
        end

        fprintf('[%2d/%d] %-18s rep %d  mech=%-13s severity=%.3f  seed=%d\n', ft, nTypes, label, rep, mechanism, severity, seed);

        set_param([mdl '/Sensing_ThreePhase'], ...
            'FaultG', onoff(G), 'FaultA', onoff(A), 'FaultB', onoff(B), 'FaultC', onoff(C), ...
            'MechanismCode', num2str(mechanism_to_code(mechanism)), ...
            'Severity', num2str(severity), 'RampDuration', '0', ...
            'FaultOnsetTime', sprintf('%.6f', onsetK), 'FaultDuration', sprintf('%.6f', faultDuration), ...
            'SourceL', sprintf('%g', SOURCE_L), 'ReclosDelay', sprintf('%.6f', RECLOSE_DELAY));
        set_param([mdl '/Signal_Conditioning_3ph'], 'NoiseSeed', num2str(seed));
        set_param([mdl '/Breaker_Contactor_Plant_3ph'], 'FeedbackEnable', num2str(FEEDBACK), ...
            'BreakerDelay', sprintf('%.6f', BREAKER_DELAY));

        simOut = sim(mdl);

        r = extract_features(simOut, label, mechanism, G, A, B, C, severity, onsetK, seed, ...
            faultDuration, SETTLE_S, Z_CLIP, RECOVERY_S, RECLOSE_DELAY, LT_WINDOW_S);
        r.fault_onset_s = onsetK;     % metadata, excluded from the features in training
        rows{end+1} = r; %#ok<AGROW>

        if SAVE_IMAGES && rep == 1       % images for the first repeat of each class only
            safe = matlab.lang.makeValidName(label);
            typeDir = fullfile(scopeDir, safe);
            if ~exist(typeDir, 'dir'), mkdir(typeDir); end
            save_scope_images(simOut, scopeGroups, typeDir, label, onset, faultDuration, IMG_RES);
            save_overview_image(simOut, fullfile(overviewDir, [safe '_overview.png']), label, mechanism, onset, faultDuration, IMG_RES);
        end
    end
end

T = struct2table(vertcat(rows{:}));
writetable(T, csvName);

fprintf('\nWrote %d rows x %d columns to %s\n', height(T), width(T), csvName);
if SAVE_IMAGES
    fprintf('Saved %d scope images in %s/ and %d overviews in %s/\n', nTypes*numel(scopeGroups), scopeDir, nTypes, overviewDir);
end
fprintf('Total time: %.1f s (%.1f min)\n', toc, toc/60);

end

%% =====================================================================
%% Plot helpers
%% =====================================================================
function c = color_for(sigName)
    if any(strcmp(sigName, {'Ia','Va','Irms_A','Crest_A','THD_A','Phase_A','Cusum_A','Residual_A','Mu0_A','A','Ia_post'}))
        c = [0.85 0.1 0.1];
    elseif any(strcmp(sigName, {'Ib','Vb','Irms_B','Crest_B','THD_B','Phase_B','Cusum_B','Residual_B','Mu0_B','B','Ib_post'}))
        c = [0.1 0.6 0.1];
    elseif any(strcmp(sigName, {'Ic','Vc','Irms_C','Crest_C','THD_C','Phase_C','Cusum_C','Residual_C','Mu0_C','C','Ic_post'}))
        c = [0.1 0.2 0.85];
    else
        c = [];
    end
end

function plot_signals(ax, simOut, sigs, onset, faultDuration)
    cyc = lines(7);
    hold(ax, 'on');
    for k = 1:numel(sigs)
        ts = simOut.get(['log_' sigs{k}]);
        t = ts.Time(:);
        y = double(squeeze(ts.Data)); y = y(:);
        col = color_for(sigs{k});
        if isempty(col), col = cyc(mod(k-1, 7) + 1, :); end
        plot(ax, t, y, 'Color', col, 'LineWidth', 1, 'DisplayName', strrep(sigs{k}, '_', '\_'));
    end
    xline(ax, onset, '--r', 'HandleVisibility', 'off');
    xline(ax, onset + faultDuration, '--k', 'HandleVisibility', 'off');
    hold(ax, 'off');
    grid(ax, 'on');
    xlim(ax, [0 t(end)]);
    legend(ax, 'Location', 'eastoutside', 'FontSize', 7);
end

function save_scope_images(simOut, scopeGroups, typeDir, label, onset, faultDuration, res)
    for g = 1:numel(scopeGroups)
        sigs = scopeGroups(g).signals;
        n = numel(sigs);
        fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1000 max(300, 170*n)]);
        for k = 1:n
            ax = subplot(n, 1, k);
            plot_signals(ax, simOut, sigs(k), onset, faultDuration);
            ylabel(ax, strrep(sigs{k}, '_', '\_'));
            if k == 1
                title(ax, sprintf('%s  -  Scope\\_%s', strrep(label, '_', '\_'), strrep(scopeGroups(g).name, '_', '\_')));
            end
            if k == n, xlabel(ax, 'Time (s)'); end
        end
        write_png(fig, fullfile(typeDir, ['Scope_' scopeGroups(g).name '.png']), res);
        close(fig);
    end
end

function save_overview_image(simOut, fileName, label, mechanism, onset, faultDuration, res)
    panels = { ...
        'Scope\_Ia\_Ib\_Ic',           {'Ia','Ib','Ic'}; ...
        'Scope\_Va\_Vb\_Vc',           {'Va','Vb','Vc'}; ...
        'Scope\_Irms',                 {'Irms_A','Irms_B','Irms_C'}; ...
        'Scope\_Crest',                {'Crest_A','Crest_B','Crest_C'}; ...
        'Scope\_THD',                  {'THD_A','THD_B','THD_C'}; ...
        'Scope\_Phase',                {'Phase_A','Phase_B','Phase_C'}; ...
        'Scope\_Cusum',                {'Cusum_A','Cusum_B','Cusum_C'}; ...
        'Scope\_Residual',             {'Residual_A','Residual_B','Residual_C'}; ...
        'Scope\_Mu0',                  {'Mu0_A','Mu0_B','Mu0_C'}; ...
        'Scope\_Trip\_Breaker\_Fault', {'fault_flag','trip','breaker_state','reclose'}; ...
        'Scope\_GABC',                 {'G','A','B','C'}; ...
        'Scope\_Sequence (I1, I2, I0)',            {'I1_seq','I2_seq','I0_seq'}; ...
        'Scope\_Sequence (Unbal2, Unbal0, UnbalV0)', {'Unbal2','Unbal0','UnbalV0'}; ...
        'Scope\_Cusum\_Unbal',         {'Cusum_U2','Cusum_U0','Cusum_V0'}; ...
        'Scope\_Breaker\_Currents (post-breaker)', {'Ia_post','Ib_post','Ic_post'} };

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 2000 1200]);
    for p = 1:size(panels, 1)
        ax = subplot(4, 4, p);
        plot_signals(ax, simOut, panels{p,2}, onset, faultDuration);
        title(ax, panels{p,1}, 'FontSize', 9);
        if any(strcmp(panels{p,2}{1}, {'fault_flag','G'})), ylim(ax, [-0.1 1.2]); end
    end
    sgtitle(sprintf('%s (%s)   red dashed = onset, black dashed = clearance', label, mechanism), ...
        'Interpreter', 'none', 'FontSize', 15, 'FontWeight', 'bold');
    write_png(fig, fileName, res);
    close(fig);
end

function write_png(fig, fileName, res)
    try
        exportgraphics(fig, fileName, 'Resolution', res);
    catch
        set(fig, 'PaperPositionMode', 'auto');
        print(fig, fileName, '-dpng', sprintf('-r%d', res));
    end
end

%% =====================================================================
%% Feature extraction
%% =====================================================================
function row = extract_features(simOut, label, mechanism, G, A, B, C, severity, onset, seed, faultDuration, settleS, zClip, recS, reclDelay, ltWindow)
    DELAY_SENTINEL = 5.0;

    row = struct();
    row.fault_type = label;
    row.mechanism = mechanism;
    row.G = G; row.A = A; row.B = B; row.C = C;
    row.severity = severity;
    row.noise_seed = seed;

    phases = {'A','B','C'};
    aggCusum = -Inf; aggResidual = -Inf; aggIrmsMax = -Inf; aggIrmsMin = Inf;
    preIrms = zeros(1,3); recIrms = zeros(1,3);

    for p = 1:3
        ph = phases{p};
        I = get_ts(simOut, ['log_I' lower(ph)]);
        V = get_ts(simOut, ['log_V' lower(ph)]);
        Irms = get_ts(simOut, ['log_Irms_' ph]);
        Crest = get_ts(simOut, ['log_Crest_' ph]);
        THD = get_ts(simOut, ['log_THD_' ph]);
        Phase = get_ts(simOut, ['log_Phase_' ph]);
        Cusum = get_ts(simOut, ['log_Cusum_' ph]);
        Residual = get_ts(simOut, ['log_Residual_' ph]);

        postMask = I.Time >= onset & I.Time < onset + faultDuration;
        preMask = I.Time >= onset - 0.1 & I.Time < onset;
        recMask = I.Time >= onset + faultDuration + reclDelay + recS;

        row.(['max_I_postfault_' ph]) = max(abs(I.Data(postMask)));
        mean_Irms_post = mean(Irms.Data(postMask));
        row.(['mean_Irms_postfault_' ph]) = mean_Irms_post;
        max_cusum_p = max(Cusum.Data(postMask));
        max_residual_p = max(Residual.Data(postMask));
        row.(['max_cusum_' ph]) = max_cusum_p;
        row.(['max_residual_' ph]) = max_residual_p;
        row.(['mean_crest_postfault_' ph]) = mean(Crest.Data(postMask));
        row.(['mean_thd_postfault_' ph]) = mean(THD.Data(postMask));
        % Phase is wrapped to (-pi, pi] in v8: use the circular mean
        row.(['mean_phase_diff_postfault_' ph]) = angle(mean(exp(1i * double(Phase.Data(postMask)))));
        zRaw = mean(abs(V.Data(postMask))) / max(mean(abs(I.Data(postMask))), 1e-6);
        row.(['implied_Z_postfault_' ph]) = min(zRaw, zClip);

        preIrms(p) = mean(Irms.Data(preMask));
        recIrms(p) = mean(Irms.Data(recMask));

        aggCusum = max(aggCusum, max_cusum_p);
        aggResidual = max(aggResidual, max_residual_p);
        aggIrmsMax = max(aggIrmsMax, mean_Irms_post);
        aggIrmsMin = min(aggIrmsMin, mean_Irms_post);
    end

    row.max_cusum_agg = aggCusum;
    row.max_residual_agg = aggResidual;
    row.mean_Irms_postfault_agg = aggIrmsMax;
    row.min_Irms_postfault_agg = aggIrmsMin;
    row.Irms_imbalance_postfault_agg = aggIrmsMax - aggIrmsMin;

    % Trip: one gang-operated breaker, so a single delay covers all phases.
    trip_ts = get_ts(simOut, 'log_trip');
    tripIdx = find(trip_ts.Data > 0, 1, 'first');
    if isempty(tripIdx)
        row.detection_delay_s_agg = DELAY_SENTINEL;
        row.tripped = 0;
    else
        row.detection_delay_s_agg = trip_ts.Time(tripIdx) - onset;
        row.tripped = 1;
    end

    % Sequence features from the SETTLED part of the fault window.
    seqNames = {'I1_seq','I2_seq','I0_seq','Unbal2','Unbal0','UnbalV0'};
    for s = 1:numel(seqNames)
        ts = get_ts(simOut, ['log_' seqNames{s}]);
        m = ts.Time >= onset + settleS & ts.Time < onset + faultDuration;
        row.(['mean_' seqNames{s} '_settled']) = mean(ts.Data(m));
        row.(['max_' seqNames{s} '_settled']) = max(ts.Data(m));
    end
    row.I0_over_I2_settled = row.mean_I0_seq_settled / max(row.mean_I2_seq_settled, 1e-9);
    cuNames = {'Cusum_U2','Cusum_U0','Cusum_V0'};
    for s = 1:numel(cuNames)
        ts = get_ts(simOut, ['log_' cuNames{s}]);
        m = ts.Time >= onset & ts.Time < onset + faultDuration;
        row.(['max_' cuNames{s}]) = max(ts.Data(m));
    end

    % ---- v8: neutral current, let-through, recovery ----
    tsA = get_ts(simOut, 'log_Ia'); tsB = get_ts(simOut, 'log_Ib'); tsC = get_ts(simOut, 'log_Ic');
    tI = tsA.Time(:);
    iA = double(tsA.Data(:)); iB = double(tsB.Data(:)); iC = double(tsC.Data(:));
    dt = median(diff(tI));

    mSet = tI >= onset + settleS & tI < onset + faultDuration;
    row.rms_I_neutral_settled = sqrt(mean((iA(mSet) + iB(mSet) + iC(mSet)).^2));

    % contacts: first change of breaker_state away from its pre-onset value
    bs = get_ts(simOut, 'log_breaker_state');
    tb = bs.Time(:); yb = double(bs.Data(:));
    kPre = find(tb < onset, 1, 'last');
    kOpen = find(tb >= onset & abs(yb - yb(kPre)) > 0.5, 1, 'first');
    if isempty(kOpen)       % contacts never opened (no trip): no let-through value
        row.letthrough_peak_A = NaN;
        row.letthrough_I2t_A2s = NaN;
    else
        mLT = tI >= onset & tI < min(tb(kOpen), onset + faultDuration);
        row.letthrough_peak_A = max(abs([iA(mLT); iB(mLT); iC(mLT)]));
        row.letthrough_I2t_A2s = dt * sum(iA(mLT).^2 + iB(mLT).^2 + iC(mLT).^2);
    end

    mFW = tI >= onset & tI < onset + ltWindow;
    row.peak_I_fixed_window_A = max(abs([iA(mFW); iB(mFW); iC(mFW)]));
    row.I2t_fixed_window_A2s = dt * sum(iA(mFW).^2 + iB(mFW).^2 + iC(mFW).^2);

    row.Irms_recovery_dev_pct = 100 * max(abs(recIrms ./ preIrms - 1));
end

%% ---- small helpers ----
function s = onoff(flag)
    if flag, s = 'on'; else, s = 'off'; end
end

function ts = get_ts(simOut, varName)
    ts = simOut.get(varName);
end
