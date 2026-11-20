%% fault_timeseries_dump_v8.m   (for fault_detection_system_v8)
%
% Model-validation tool (no ML). For every fault type it runs TWO simulations:
%
%   pass 0  breaker feedback OFF  (FeedbackEnable = 0): the breaker only latches, the plant keeps
%           carrying the fault. Use it to characterise the bolted fault (settled Irms, I1/I2/I0,
%           Phase, THD, ...). Same meaning as the v6 dump.
%   pass 1  breaker feedback ON   (FeedbackEnable = 1, BreakerDelay s after the trip command the
%           contacts open and the line currents go to zero, reclose ReclosDelay s after clearance).
%           Use it to check the protection chain: trip delay, opening time, let-through peak and
%           I2t, interrupted current, reclose, recovery, re-trip.
%
% Both passes use the same noise seed (3000 + fault number), so they are directly comparable.
%
% Outputs (all in fault_analysis/):
%   fault_timeseries_all.csv       all 47 logged signals + I_neutral + V_sum, long format, column
%                                  'feedback' = 0 / 1. Coarse stride plus native resolution around
%                                  onset, clearance and reclose.
%   fault_events_summary.csv       one row per fault AND pass (56 rows) of event / diagnostic numbers
%   fault_onset_sweep.csv          trip delay, breaker opening, let-through vs fault inception angle
%   fault_onset_sweep_summary.csv  one row per swept fault: min / mean / max / spread over the angles
%   fault_timeseries_all.zip       zipped time-series CSV
%
% v8 changes vs the v7 dump: let-through / I2t are NaN when the contacts never open (no more Healthy = 1745),
% two fixed-window columns (peak_I_fixed_window_A, I2t_fixed_window_A2s, window LT_WINDOW_S after
% onset) are comparable across ALL runs, and the onset sweep covers 180 degrees (the waveform repeats
% every half cycle, so 360 degrees gave every row twice).
%
% Needs on the path: fault_detection_system_v8.slx (run model_code_v8 first), fault_type_table.m,
% severity_range_for_mechanism.m, mechanism_to_code.m
% Needs MATLAB R2020a or newer (append-write).
% The plotting helpers of the v6 dump were commented out there and are not carried over.
%
% USAGE:
%   >> model_code_v8
%   >> fault_timeseries_dump_v8

function fault_timeseries_dump_v8()

tic;

%% ---- settings ----
mdl = 'fault_detection_system_v8';
onset = 0.4;
faultDuration = 1.0;
stopTime = 2.0;

SOURCE_L = 5e-3;        % H, source inductance (0 = v6 behaviour)
BREAKER_DELAY = 0.04;   % s from trip command to contacts open (2 cycles at 50 Hz)
RECLOSE_DELAY = 0.1;    % s from fault clearance to the reclose pulse
PASSES = [0 1];         % 0 = breaker feedback off, 1 = on
CSV_PASSES = [0 1];     % which passes go into fault_timeseries_all.csv

SEVERITY_POS = 0.5;     % 0 = low end, 0.5 = midpoint, 1 = high end of each mechanism's range
ONLY = {};              % e.g. {'Healthy','LL_AB','LLG_AB'} to run a few faults; {} = all 28
SETTLE_S = 0.1;         % skip this long after onset for "settled" averages (open-loop pass only)
RECOVERY_S = 0.1;       % skip this long after the load is reconnected for the "recovery" average
Z_CLIP = 1000;          % ohms

% ---- CSV size control ----
% Keeps every COARSE_STRIDE-th sample, plus ALL native samples in three windows (onset, clearance,
% reclose). With two passes this is about 2400 rows x 56 runs = 134k rows, roughly 45 MB; set
% CSV_PASSES = 1 to halve it. Events and the sweep always use the full native data.
COARSE_STRIDE = 20;               % 20 x 1e-4 s = 2 ms
FINE_ONSET = [-0.010 0.060];      % relative to onset (covers trip + 40 ms breaker delay)
FINE_CLEAR = [-0.005 0.030];      % relative to clearance
FINE_RECLOSE = [-0.005 0.030];    % relative to the reclose pulse
CSV_SIGNIFICANT_DIGITS = 5;
MAKE_ZIP = true;

% ---- onset-angle sweep (finding 1 of the v6 analysis) ----
DO_ONSET_SWEEP = true;
SWEEP_FAULTS = {'LG_A','LL_AB','LLL','Overcurrent_A','Overvoltage_A','OpenCircuit_A'};   % {} = all 28
SWEEP_SPAN_DEG = 180;   % the plant is odd-symmetric, so 180 deg (half a cycle) already covers every inception angle
N_ANGLES = 10;          % onset moved in N_ANGLES equal steps over SWEEP_SPAN_DEG (18 deg = 1 ms for 10).
                        % Keep 10 ms * SWEEP_SPAN_DEG/180 / N_ANGLES a multiple of 0.1 ms (5, 10, 20, 25, 50 all work)
LT_WINDOW_S = 0.1;      % fixed window after onset for peak_I_fixed_window_A and I2t_fixed_window_A2s
SWEEP_FD = 0.15;        % shorter fault window for the sweep runs, s
SWEEP_RECLOSE = 0.05;   % reclose delay for the sweep runs, s

outDir = 'fault_analysis';
csvTS = fullfile(outDir, 'fault_timeseries_all.csv');
csvEvents = fullfile(outDir, 'fault_events_summary.csv');
csvSweep = fullfile(outDir, 'fault_onset_sweep.csv');
csvSweepSum = fullfile(outDir, 'fault_onset_sweep_summary.csv');

if ~bdIsLoaded(mdl), load_system([mdl '.slx']); end
set_param(mdl, 'StopTime', num2str(stopTime));
if ~exist(outDir, 'dir'), mkdir(outDir); end
if exist(csvTS, 'file'), delete(csvTS); end
clr = onset + faultDuration;
tRecl = clr + RECLOSE_DELAY;
fineWin = [onset + FINE_ONSET(1), onset + FINE_ONSET(2); ...
           clr + FINE_CLEAR(1), clr + FINE_CLEAR(2); ...
           tRecl + FINE_RECLOSE(1), tRecl + FINE_RECLOSE(2)];

%% ---- the 47 logged signals (same names as the To-Workspace blocks: log_<name>) ----
rawNames = {'Ia','Ib','Ic','Va','Vb','Vc', ...
    'Irms_A','Irms_B','Irms_C','Crest_A','Crest_B','Crest_C', ...
    'THD_A','THD_B','THD_C','Phase_A','Phase_B','Phase_C', ...
    'Cusum_A','Cusum_B','Cusum_C','Residual_A','Residual_B','Residual_C', ...
    'Mu0_A','Mu0_B','Mu0_C', ...
    'trip','breaker_state','fault_flag','G','A','B','C', ...
    'I1_seq','I2_seq','I0_seq','Unbal2','Unbal0','UnbalV0', ...
    'Cusum_U2','Cusum_U0','Cusum_V0', ...
    'Ia_post','Ib_post','Ic_post','reclose'};
assert(numel(rawNames) == 47, 'Expected 47 logged signals, got %d', numel(rawNames));
derivedNames = {'I_neutral','V_sum'};
allNames = [rawNames derivedNames];
discreteNames = {'trip','breaker_state','fault_flag','G','A','B','C','reclose'};

faultTypes = fault_type_table();
nTypes = size(faultTypes, 1);
evRows = {};
firstWrite = true;

for ft = 1:nTypes
    label = faultTypes{ft,1};
    mechanism = faultTypes{ft,2};
    G = faultTypes{ft,3}; A = faultTypes{ft,4}; B = faultTypes{ft,5}; C = faultTypes{ft,6};
    seed = 3000 + ft;

    if ~isempty(ONLY) && ~any(strcmp(label, ONLY)), continue; end

    sr = severity_range_for_mechanism(mechanism);
    severity = sr(1) + SEVERITY_POS * (sr(2) - sr(1));

    for fb = PASSES
        fprintf('[%2d/%d] %-18s feedback=%d mech=%-13s severity=%.3f seed=%d\n', ft, nTypes, label, fb, mechanism, severity, seed);

        configure_run(mdl, G, A, B, C, mechanism, severity, onset, faultDuration, SOURCE_L, RECLOSE_DELAY, seed, fb, BREAKER_DELAY);
        simOut = sim(mdl);

        S = collect_signals(simOut, rawNames, discreteNames);
        S.I_neutral = S.Ia + S.Ib + S.Ic;
        S.V_sum = S.Va + S.Vb + S.Vc;
        nS = numel(S.time_s);

        % CSV grid: coarse everywhere, native resolution around onset, clearance and reclose
        tn = S.time_s;
        keep = false(nS, 1);
        keep(1:COARSE_STRIDE:nS) = true;
        keep(nS) = true;
        for w = 1:size(fineWin, 1)
            keep = keep | (tn >= fineWin(w,1) & tn <= fineWin(w,2));
        end
        idx = find(keep);

        ev = compute_events(S, label, mechanism, G, A, B, C, severity, seed, ft, fb, ...
            onset, faultDuration, SETTLE_S, RECOVERY_S, COARSE_STRIDE, Z_CLIP, LT_WINDOW_S);
        evRows{end+1} = ev; %#ok<AGROW>

        fprintf('        samples=%d (dt=%.2g s) -> %d csv rows | trip %s | contacts open %s | peak let-through %.1f A | I2t %.0f A^2 s | recovery dev %.2f %%\n', ...
            nS, ev.native_dt_s, numel(idx), fmt_ms(ev.trip_delay_ms), fmt_ms(ev.breaker_open_delay_ms), ...
            ev.letthrough_peak_A, ev.letthrough_I2t_A2s, ev.Irms_recovery_dev_pct);
        if fb == 1 && ev.interrupt_ok == 0
            fprintf('        WARNING: current after the contacts opened is %.3g A (should be 0)\n', ev.I_after_open_max_A);
        end

        if ismember(fb, CSV_PASSES)
            write_ts_csv(S, allNames, idx, ft, fb, label, mechanism, severity, seed, G, A, B, C, csvTS, firstWrite, CSV_SIGNIFICANT_DIGITS);
            firstWrite = false;
        end
        clear S simOut
    end
end

T = struct2table(vertcat(evRows{:}));
writetable(T, csvEvents);

info = dir(csvTS);
fprintf('\nWrote %s (%.1f MB) and %s (%d rows x %d columns)\n', csvTS, info.bytes/1e6, csvEvents, height(T), width(T));
if MAKE_ZIP
    zipName = fullfile(outDir, 'fault_timeseries_all.zip');
    zip(zipName, 'fault_timeseries_all.csv', outDir);
    zi = dir(zipName);
    fprintf('Zipped to %s (%.1f MB)\n', zipName, zi.bytes/1e6);
end

%% ---- onset-angle sweep ----
if DO_ONSET_SWEEP
    run_onset_sweep(mdl, faultTypes, SWEEP_FAULTS, SEVERITY_POS, N_ANGLES, SWEEP_SPAN_DEG, SWEEP_FD, SWEEP_RECLOSE, ...
        BREAKER_DELAY, SOURCE_L, onset, rawNames, discreteNames, csvSweep, csvSweepSum);
end

fprintf('Total time: %.1f s (%.1f min)\n', toc, toc/60);

end

%% =====================================================================
%% Model configuration
%% =====================================================================
function configure_run(mdl, G, A, B, C, mechanism, severity, onset, faultDuration, sourceL, reclDelay, seed, fb, breakerDelay)
    set_param([mdl '/Sensing_ThreePhase'], ...
        'FaultG', onoff(G), 'FaultA', onoff(A), 'FaultB', onoff(B), 'FaultC', onoff(C), ...
        'MechanismCode', num2str(mechanism_to_code(mechanism)), ...
        'Severity', num2str(severity), 'RampDuration', '0', ...
        'FaultOnsetTime', sprintf('%.6f', onset), 'FaultDuration', sprintf('%.6f', faultDuration), ...
        'SourceL', sprintf('%g', sourceL), 'ReclosDelay', sprintf('%.6f', reclDelay));
    set_param([mdl '/Signal_Conditioning_3ph'], 'NoiseSeed', num2str(seed));
    set_param([mdl '/Breaker_Contactor_Plant_3ph'], 'FeedbackEnable', num2str(fb), 'BreakerDelay', sprintf('%.6f', breakerDelay));
end

%% =====================================================================
%% Data collection
%% =====================================================================
function S = collect_signals(simOut, rawNames, discreteNames)
    ref = simOut.get('log_Ia');
    tn = ref.Time(:);
    S = struct();
    S.time_s = tn;
    for k = 1:numel(rawNames)
        nm = rawNames{k};
        ts = simOut.get(['log_' nm]);
        tt = ts.Time(:);
        y = double(squeeze(ts.Data));
        if numel(y) ~= numel(tt)
            error('log_%s has %d data values for %d time stamps (not a scalar signal).', nm, numel(y), numel(tt));
        end
        y = y(:);
        if numel(tt) == numel(tn) && max(abs(tt - tn)) < 1e-12
            S.(nm) = y;
        else
            [tt, iu] = unique(tt);
            y = y(iu);
            if ismember(nm, discreteNames), meth = 'previous'; else, meth = 'linear'; end
            v = interp1(tt, y, tn, meth);
            S.(nm) = fillmissing(v, 'nearest');
        end
    end
end

function write_ts_csv(S, names, idx, id, fb, label, mechanism, sev, seed, G, A, B, C, fileName, firstWrite, nSig)
    n = numel(idx);
    M = zeros(n, numel(names));
    for k = 1:numel(names), M(:,k) = S.(names{k})(idx); end
    M = round(M, nSig, 'significant');
    t = round(S.time_s(idx), 9);
    Tm = table(repmat(id,n,1), repmat(fb,n,1), repmat(string(label),n,1), repmat(string(mechanism),n,1), ...
        repmat(sev,n,1), repmat(seed,n,1), repmat(G,n,1), repmat(A,n,1), repmat(B,n,1), repmat(C,n,1), t, ...
        'VariableNames', {'fault_id','feedback','fault_type','mechanism','severity','noise_seed','cfg_G','cfg_A','cfg_B','cfg_C','time_s'});
    T = [Tm, array2table(M, 'VariableNames', names)];
    if firstWrite
        writetable(T, fileName, 'WriteMode', 'overwrite');
    else
        writetable(T, fileName, 'WriteMode', 'append', 'WriteVariableNames', false);
    end
end

%% =====================================================================
%% Events and diagnostics (native resolution)
%% =====================================================================
function ev = compute_events(S, label, mechanism, G, A, B, C, severity, seed, id, fb, onset, fd, settleS, recS, decim, zClip, ltWindow)
    tn = S.time_s;
    dt = median(diff(tn));
    clr = onset + fd;
    ph = {'A','B','C'};

    ev = struct();
    ev.fault_id = id; ev.fault_type = label; ev.mechanism = mechanism;
    ev.breaker_feedback = fb;
    ev.cfg_G = G; ev.cfg_A = A; ev.cfg_B = B; ev.cfg_C = C;
    ev.severity = severity; ev.noise_seed = seed;
    ev.native_samples = numel(tn);
    ev.native_dt_s = dt;
    ev.csv_decimation = decim;

    mPre = tn >= onset - 0.1 & tn < onset;
    mFault = tn >= onset & tn < clr;
    ipre = find(tn < onset, 1, 'last');

    % trip command
    it = find(S.trip > 0, 1, 'first');
    ev.tripped = double(~isempty(it));
    if isempty(it)
        ev.trip_time_s = NaN; ev.trip_delay_ms = NaN; ev.trip_before_onset = 0;
    else
        ev.trip_time_s = tn(it); ev.trip_delay_ms = (tn(it) - onset) * 1e3;
        ev.trip_before_onset = double(tn(it) < onset);
    end
    ev.tripped_at_end = S.trip(end);

    % breaker contacts: first change away from the pre-onset value
    b0 = S.breaker_state(ipre);
    ib = find(tn >= onset & abs(S.breaker_state - b0) > 0.5, 1, 'first');
    ev.breaker_state_pre = b0;
    if isempty(ib)
        ev.breaker_open_time_s = NaN; ev.breaker_open_delay_ms = NaN; ev.breaker_open_minus_trip_ms = NaN;
    else
        ev.breaker_open_time_s = tn(ib); ev.breaker_open_delay_ms = (tn(ib) - onset) * 1e3;
        ev.breaker_open_minus_trip_ms = (tn(ib) - ev.trip_time_s) * 1e3;
    end

    % reclose pulse
    ir = find(S.reclose > 0.5, 1, 'first');
    if isempty(ir), tRec = NaN; else, tRec = tn(ir); end
    ev.reclose_time_s = tRec;
    if isnan(tRec), ev.retrip_after_reclose = NaN; else, ev.retrip_after_reclose = double(any(S.trip(tn >= tRec + 0.005) > 0)); end

    % fault flag (injector ground truth)
    iff = find(S.fault_flag > 0.5 & tn >= onset, 1, 'first');
    if isempty(iff), ev.fault_flag_delay_ms = NaN; else, ev.fault_flag_delay_ms = (tn(iff) - onset) * 1e3; end
    ev.fault_flag_before_onset = double(any(S.fault_flag(tn < onset) > 0.5));

    % windows. "Settled" fault window ends at clearance, or at contact opening in the closed-loop
    % pass (so it is empty there: use pass 0 for settled fault features).
    tEnd = clr;
    if fb == 1 && ~isempty(ib), tEnd = min(clr, tn(ib)); end
    mSet = tn >= onset + settleS & tn < tEnd;
    if fb == 1 && ~isnan(tRec), tRec0 = tRec; else, tRec0 = clr; end
    mRec = tn >= tRec0 + recS;
    if fb == 1 && ~isempty(ib), mAT = tn >= tn(ib) + 0.05 & tn < clr; else, mAT = false(size(tn)); end

    % let-through: everything between onset and contact opening (or clearance if never opened)
    % NaN when the contacts never opened (no trip), so Healthy is not mixed with interrupted faults.
    if isempty(ib)
        ev.letthrough_peak_A = NaN; ev.letthrough_I2t_A2s = NaN;
    else
        mLT = tn >= onset & tn < min(tn(ib), clr);
        ev.letthrough_peak_A = max([wmaxabs(S.Ia, mLT), wmaxabs(S.Ib, mLT), wmaxabs(S.Ic, mLT)]);
        ev.letthrough_I2t_A2s = dt * (wsumsq(S.Ia, mLT) + wsumsq(S.Ib, mLT) + wsumsq(S.Ic, mLT));
    end
    % Fixed window after onset, comparable across ALL runs: breaker-off pass = unprotected fault,
    % breaker-on pass = what the protection actually lets through in the same window.
    mFW = tn >= onset & tn < onset + ltWindow;
    ev.peak_I_fixed_window_A = max([wmaxabs(S.Ia, mFW), wmaxabs(S.Ib, mFW), wmaxabs(S.Ic, mFW)]);
    ev.I2t_fixed_window_A2s = dt * (wsumsq(S.Ia, mFW) + wsumsq(S.Ib, mFW) + wsumsq(S.Ic, mFW));

    % current after the contacts opened (closed loop only): must be exactly zero
    if fb == 1 && ~isempty(ib)
        tStop = tRec; if isnan(tStop), tStop = tn(end) + 1; end
        mAO = tn >= tn(ib) + 2*dt & tn < tStop;
        ev.I_after_open_max_A = max([wmaxabs(S.Ia, mAO), wmaxabs(S.Ib, mAO), wmaxabs(S.Ic, mAO)]);
        ev.interrupt_ok = double(ev.I_after_open_max_A < 1e-9);
    else
        ev.I_after_open_max_A = NaN; ev.interrupt_ok = NaN;
    end

    % per phase
    for p = 1:3
        q = ph{p};
        I = S.(['I' lower(q)]); V = S.(['V' lower(q)]); R = S.(['Irms_' q]);
        ev.(['Irms_pre_' q]) = wmean(R, mPre);
        ev.(['Irms_fault_' q]) = wmean(R, mSet);
        ev.(['Irms_ratio_' q]) = ev.(['Irms_fault_' q]) / ev.(['Irms_pre_' q]);
        ev.(['Irms_after_open_' q]) = wmean(R, mAT);
        ev.(['Irms_recovery_' q]) = wmean(R, mRec);
        [pk, tpk] = wpeak(I, tn, mFault);
        ev.(['peak_I_' q]) = pk;
        ev.(['peak_I_' q '_t_s']) = tpk;
        ev.(['Ipk_pre_' q]) = wmaxabs(I, mPre);
        ev.(['Vpk_pre_' q]) = wmaxabs(V, mPre);
        ev.(['Vpk_fault_' q]) = wmaxabs(V, mSet);
        zpre = wmean(abs(V), mPre) / max(wmean(abs(I), mPre), 1e-6);
        zflt = wmean(abs(V), mSet) / max(wmean(abs(I), mSet), 1e-6);
        ev.(['implied_Z_pre_' q]) = min(zpre, zClip);
        ev.(['implied_Z_fault_' q]) = min(zflt, zClip);
        ev.(['max_cusum_' q]) = wmax(S.(['Cusum_' q]), mFault);
        ev.(['max_residual_' q]) = wmax(S.(['Residual_' q]), mFault);
        ev.(['mean_crest_fault_' q]) = wmean(S.(['Crest_' q]), mSet);
        ev.(['mean_thd_fault_' q]) = wmean(S.(['THD_' q]), mSet);
        ev.(['mean_phase_fault_' q]) = wmean(S.(['Phase_' q]), mSet);
    end
    devs = abs([ev.Irms_recovery_A / ev.Irms_pre_A, ev.Irms_recovery_B / ev.Irms_pre_B, ev.Irms_recovery_C / ev.Irms_pre_C] - 1) * 100;
    ev.Irms_recovery_dev_pct = max(devs);

    % sequence components and neutral current (settled window: open-loop pass only)
    ev.I1_pre = wmean(S.I1_seq, mPre);
    ev.I1_settled = wmean(S.I1_seq, mSet);
    ev.I2_settled = wmean(S.I2_seq, mSet);
    ev.I0_settled = wmean(S.I0_seq, mSet);
    ev.I0_over_I2_settled = ev.I0_settled / max(ev.I2_settled, 1e-9);
    ev.Unbal2_settled = wmean(S.Unbal2, mSet);
    ev.Unbal0_settled = wmean(S.Unbal0, mSet);
    ev.UnbalV0_settled = wmean(S.UnbalV0, mSet);
    ev.max_Cusum_U2 = wmax(S.Cusum_U2, mFault);
    ev.max_Cusum_U0 = wmax(S.Cusum_U0, mFault);
    ev.max_Cusum_V0 = wmax(S.Cusum_V0, mFault);
    ev.rms_I_neutral_pre = wrms(S.I_neutral, mPre);
    ev.rms_I_neutral_settled = wrms(S.I_neutral, mSet);
    ev.rms_V_sum_pre = wrms(S.V_sum, mPre);
    ev.rms_V_sum_settled = wrms(S.V_sum, mSet);
end

function m = wmean(x, mask)
    if ~any(mask), m = NaN; else, m = mean(x(mask)); end
end
function m = wmax(x, mask)
    if ~any(mask), m = NaN; else, m = max(x(mask)); end
end
function m = wmaxabs(x, mask)
    if ~any(mask), m = NaN; else, m = max(abs(x(mask))); end
end
function m = wrms(x, mask)
    if ~any(mask), m = NaN; else, m = sqrt(mean(x(mask).^2)); end
end
function m = wsumsq(x, mask)
    if ~any(mask), m = NaN; else, m = sum(x(mask).^2); end
end
function [pk, tpk] = wpeak(x, t, mask)
    if ~any(mask), pk = NaN; tpk = NaN; return; end
    xi = x(mask); ti = t(mask);
    [~, k] = max(abs(xi));
    pk = xi(k); tpk = ti(k);
end

function s = fmt_ms(x)
    if isnan(x), s = 'none'; else, s = sprintf('%.3f ms', x); end
end

%% =====================================================================
%% Onset-angle sweep: same fault, onset moved over half a cycle (180 deg)
%% =====================================================================
function run_onset_sweep(mdl, faultTypes, sweepNames, sevPos, nAngles, spanDeg, sweepFd, sweepRecl, breakerDelay, sourceL, onset0, rawNames, discreteNames, csvSweep, csvSweepSum)
    simNames = {'Ia','Ib','Ic','trip','breaker_state','reclose'};
    nTypes = size(faultTypes, 1);
    rows = {};
    for ft = 1:nTypes
        label = faultTypes{ft,1};
        if ~isempty(sweepNames) && ~any(strcmp(label, sweepNames)), continue; end
        mechanism = faultTypes{ft,2};
        G = faultTypes{ft,3}; A = faultTypes{ft,4}; B = faultTypes{ft,5}; C = faultTypes{ft,6};
        seed = 3000 + ft;
        sr = severity_range_for_mechanism(mechanism);
        severity = sr(1) + sevPos * (sr(2) - sr(1));

        for k = 1:nAngles
            onsetK = onset0 + (k - 1) * (spanDeg / 360 * 0.02) / nAngles;   % 50 Hz: 360 deg = 20 ms
            angleDeg = mod(360 * 50 * (onsetK - onset0), 360);              % relative to the rising zero crossing of Va
            fprintf('[sweep] %-16s angle %6.1f deg (onset %.5f s)\n', label, angleDeg, onsetK);

            configure_run(mdl, G, A, B, C, mechanism, severity, onsetK, sweepFd, sourceL, sweepRecl, seed, 1, breakerDelay);
            set_param(mdl, 'StopTime', sprintf('%.6f', onsetK + sweepFd + sweepRecl + 0.15));
            simOut = sim(mdl);
            S = collect_signals(simOut, simNames, discreteNames);
            tn = S.time_s; dt = median(diff(tn));

            r = struct();
            r.fault_type = label; r.mechanism = mechanism; r.severity = severity;
            r.onset_s = onsetK; r.inception_angle_deg = angleDeg;

            it = find(S.trip > 0, 1, 'first');
            if isempty(it) || tn(it) < onsetK
                r.trip_delay_ms = NaN;
            else
                r.trip_delay_ms = (tn(it) - onsetK) * 1e3;
            end
            r.trip_before_onset = double(~isempty(it) && tn(it) < onsetK);

            ib = find(tn >= onsetK & S.breaker_state > 0.5, 1, 'first');
            tOpen = onsetK + sweepFd;
            if isempty(ib)
                r.breaker_open_delay_ms = NaN;
            else
                r.breaker_open_delay_ms = (tn(ib) - onsetK) * 1e3;
                tOpen = min(tn(ib), tOpen);
            end
            mLT = tn >= onsetK & tn < tOpen;
            if isempty(ib)      % contacts never opened: no let-through value
                r.letthrough_peak_A = NaN; r.letthrough_I2t_A2s = NaN;
            else
                r.letthrough_peak_A = max([wmaxabs(S.Ia, mLT), wmaxabs(S.Ib, mLT), wmaxabs(S.Ic, mLT)]);
                r.letthrough_I2t_A2s = dt * (wsumsq(S.Ia, mLT) + wsumsq(S.Ib, mLT) + wsumsq(S.Ic, mLT));
            end
            r.tripped_at_end = S.trip(end);
            rows{end+1} = r; %#ok<AGROW>
            clear S simOut
        end
    end

    T = struct2table(vertcat(rows{:}));
    writetable(T, csvSweep);

    names = unique(T.fault_type, 'stable');
    Srows = cell(numel(names), 1);
    for i = 1:numel(names)
        m = strcmp(T.fault_type, names{i});
        d = T.trip_delay_ms(m);
        lp = T.letthrough_peak_A(m);
        s = struct();
        s.fault_type = names{i};
        s.n_angles = nnz(m);
        s.trip_delay_min_ms = min(d); s.trip_delay_mean_ms = mean(d, 'omitnan'); s.trip_delay_max_ms = max(d);
        s.trip_delay_spread_ms = max(d) - min(d);
        s.letthrough_peak_min_A = min(lp); s.letthrough_peak_max_A = max(lp);   % min / max ignore NaN
        s.n_no_trip = nnz(isnan(d));
        Srows{i} = s;
    end
    TS = struct2table(vertcat(Srows{:}));
    writetable(TS, csvSweepSum);
    fprintf('Wrote %s (%d rows) and %s (%d rows)\n', csvSweep, height(T), csvSweepSum, height(TS));
end

%% ---- small helper ----
function s = onoff(flag)
    if flag, s = 'on'; else, s = 'off'; end
end
