% Sweeps all 28 fault-type rows (see fault_type_table.m: 1 Healthy + 11
% ShortCircuit location combos + 4 mechanisms x 4 phase-selections) x
% onset times x severity/noise-seed repeats, runs
% fault_detection_system_v3, and writes one CSV row per run with THREE
% tiers of features (per-phase, aggregate, raw-Kaggle-style).
%
% *** UPDATED: severity is now drawn per-mechanism from
% severity_range_for_mechanism.m's [low,high] ranges (the exact ranges
% already validated on the single-phase v10 pipeline - see that file's
% header), using v10's proven rand()-within-range-per-repeat pattern,
% instead of one fixed [2,10,50,200] grid applied to every fault type
% regardless of mechanism. That fixed grid was never physically valid for
% Overvoltage/Undervoltage (V-scale factors, not Ohms) or OpenCircuit
% (needs R > 300 Ohm to trigger the decay branch at all) - it would have
% silently produced nonsense runs for those mechanisms. ***

function data_extractor_code_v3()

tic;   % time the full sweep, reported at the end after the CSV is written

mdl = 'fault_detection_system_v3_updated';
if ~bdIsLoaded(mdl)
    load_system([mdl '.slx']);
end

faultTypes = fault_type_table();   % from fault_type_table.m - 28 rows
onsetTimes = [0.3, 0.4, 0.5, 0.6, 0.7];
faultDuration = 1.0;
% *** GROWN (post-ML-review): bumped 2 -> 5. Unlike Healthy (dummy
% severity, so repeats there would just duplicate rows - see
% onsetTimesHealthy above), every non-Healthy mechanism now genuinely
% varies its physics with each independent severity draw post-fix, so
% more reps here means more real training diversity across the severity
% range, not duplicate rows. 27 non-Healthy types x 5 onsets x 5 reps =
% 675 runs (up from 270) - re-run time scales accordingly, budget for it. ***
nRepeats = 5;   % independent (severity, noise seed) draws per (type, onset)

% *** FIX (post-sweep audit): Healthy previously got only 5 rows total
% (one per onset in onsetTimes, repsForType forced to 1) - thin relative
% to every other fault_type's 10, and arguably the worst class to
% undertrain since a Healthy miss is a missed real-fault-vs-no-fault call.
% Bumping repsForType instead would NOT have helped: severity is a dummy
% (severity_range_for_mechanism('Healthy') = [1,1], so severity always
% evaluates to exactly 1) and NoiseSeed has no physical effect (no noise
% source wired in), so repeats at a FIXED onset would just be bit-
% identical duplicate rows - not real new information. Onset time is the
% only thing that actually changes a Healthy row's feature values (it
% shifts how much of the startup transient falls in each pre/post
% window - see the mean_I_prefault_* discussion). So Healthy gets its own
% denser onset sweep instead of extra reps at the same 5 onsets. ***
onsetTimesHealthy = linspace(0.2, 1.2, 15);   % 15 distinct onsets -> 15 genuinely-distinct Healthy rows

rng(1);   % seed severity draws for a reproducible sweep, same as v10
seedCounter = 2001;   % distinct from v8-v10's 1000-1999 range, so a
                       % combined single-phase+3-phase seed audit never collides

set_param(mdl, 'StopTime', num2str(max([onsetTimes, onsetTimesHealthy]) + faultDuration + 0.5));

rows = {};

for ft = 1:size(faultTypes,1)
    label = faultTypes{ft,1};
    mechanism = faultTypes{ft,2};
    G = faultTypes{ft,3}; A = faultTypes{ft,4}; B = faultTypes{ft,5}; C = faultTypes{ft,6};
    isHealthy = strcmp(mechanism, 'Healthy');
    severityRange = severity_range_for_mechanism(mechanism);
    repsForType = nRepeats;
    if isHealthy
        repsForType = 1;   % still 1 - see onsetTimesHealthy fix above for why reps don't help Healthy
    end

    if isHealthy
        thisOnsetTimes = onsetTimesHealthy;
    else
        thisOnsetTimes = onsetTimes;
    end

    for onset = thisOnsetTimes
        for rep = 1:repsForType
            seed = seedCounter; seedCounter = seedCounter + 1;
            severity = severityRange(1) + rand() * (severityRange(2) - severityRange(1));

            % FIX: 'Mechanism' no longer exists as a mask param (see
            % v11_model_code.m header) - convert the label to its numeric
            % MechanismCode via mechanism_to_code.m before set_param, so
            % the value actually reaches Fault_Injector's inline
            % switch/case instead of a derivation step that used to
            % silently fall back to Healthy.
            set_param([mdl '/Sensing_ThreePhase'], ...
                'FaultG', onoff(G), 'FaultA', onoff(A), 'FaultB', onoff(B), 'FaultC', onoff(C), ...
                'MechanismCode', num2str(mechanism_to_code(mechanism)), 'Severity', num2str(severity), 'RampDuration', '0', ...
                'FaultOnsetTime', num2str(onset), 'FaultDuration', num2str(faultDuration), ...
                'NoiseSeed', num2str(seed));

            simOut = sim(mdl);

            row = extract_features(simOut, label, mechanism, G, A, B, C, severity, onset, seed, faultDuration);
            rows{end+1} = row; %#ok<AGROW>
        end
    end
end

T = vertcat(rows{:});
writetable(struct2table(T), 'fault_detection_features_v3.csv');
fprintf('Wrote %d rows to fault_detection_features_v3.csv\n', numel(rows));

elapsed = toc;
fprintf('Total sweep time: %.1f s (%.1f min)\n', elapsed, elapsed/60);

end

function s = onoff(flag)
    if flag, s = 'on'; else, s = 'off'; end
end

function row = extract_features(simOut, label, mechanism, G, A, B, C, severity, onset, seed, faultDuration)
    DELAY_SENTINEL = 5.0;

    row = struct();
    row.fault_type = label;
    row.mechanism = mechanism;
    row.G = G; row.A = A; row.B = B; row.C = C;   % Kaggle-native label columns
    row.severity = severity;
    row.fault_onset_s = onset;
    row.noise_seed = seed;

    phases = {'A','B','C'};
    % FIX (post-sweep audit): aggDelay used to start at -Inf, which
    % isnan() never catches, so fillna() silently skipped it and
    % never-tripped rows (e.g. Healthy) ended up with
    % detection_delay_s_agg == -Inf instead of the DELAY_SENTINEL every
    % other never-tripped column correctly falls back to. Starting from
    % NaN instead fixes this: MATLAB's max() ignores NaN against a real
    % number (so a genuine trip on any phase still updates aggDelay
    % normally via max_or_nan below), and if every phase stays NaN,
    % aggDelay stays NaN too, so fillna() now catches it correctly.
    aggCusum = -Inf; aggResidual = -Inf; aggIrmsMax = -Inf; aggIrmsMin = Inf; aggDelay = NaN; aggBreakerDelay = -Inf;

    for p = 1:3
        ph = phases{p};
        I = get_ts(simOut, ['log_I' lower(ph)]);  % Ia/Ib/Ic
        V = get_ts(simOut, ['log_V' lower(ph)]);
        Irms = get_ts(simOut, ['log_Irms_' ph]);
        Crest = get_ts(simOut, ['log_Crest_' ph]);
        THD = get_ts(simOut, ['log_THD_' ph]);
        Phase = get_ts(simOut, ['log_Phase_' ph]);
        Cusum = get_ts(simOut, ['log_Cusum_' ph]);
        Residual = get_ts(simOut, ['log_Residual_' ph]);

        preMask = I.Time < onset;
        % FIX: was I.Time >= onset (unbounded, ran to end-of-sim
        % including the post-clearance recovery tail), diluting every
        % mean-based postfault feature. Now bounded to the fault-active
        % window, matching v9/v10.
        postMask = I.Time >= onset & I.Time < onset + faultDuration;

        mean_I_pre = mean(I.Data(preMask));
        std_I_pre = std(I.Data(preMask));
        max_I_post = max(abs(I.Data(postMask)));
        mean_Irms_post = mean(Irms.Data(postMask));
        max_cusum_p = max(Cusum.Data(postMask));
        max_residual_p = max(Residual.Data(postMask));
        mean_crest_post = mean(Crest.Data(postMask));
        mean_thd_post = mean(THD.Data(postMask));
        mean_phase_post = mean(Phase.Data(postMask));
        implied_Z = mean(abs(V.Data(postMask))) / max(mean(abs(I.Data(postMask))), 1e-6);

        % Per-phase tripped/detection-delay: this model trips ALL phases
        % together (gang-operated breaker, see v11_model_code.m), so
        % detection_delay is really a per-SIMULATION quantity, but we
        % still record it per-phase using that phase's own cusum-flag
        % crossing time, in case a future per-pole breaker variant needs it.
        trip_ts = get_ts(simOut, 'log_trip');
        tripIdx = find(trip_ts.Data > 0, 1, 'first');
        if isempty(tripIdx)
            delay = NaN;
        else
            delay = trip_ts.Time(tripIdx) - onset;
        end

        row.(['mean_I_prefault_' ph]) = mean_I_pre;
        row.(['std_I_prefault_' ph]) = std_I_pre;
        row.(['max_I_postfault_' ph]) = max_I_post;
        row.(['mean_Irms_postfault_' ph]) = mean_Irms_post;
        row.(['max_cusum_' ph]) = max_cusum_p;
        row.(['max_residual_' ph]) = max_residual_p;
        row.(['mean_crest_postfault_' ph]) = mean_crest_post;
        row.(['mean_thd_postfault_' ph]) = mean_thd_post;
        row.(['mean_phase_diff_postfault_' ph]) = mean_phase_post;
        row.(['implied_Z_postfault_' ph]) = implied_Z;
        row.(['detection_delay_s_' ph]) = fillna(delay, DELAY_SENTINEL);

        aggCusum = max(aggCusum, max_cusum_p);
        aggResidual = max(aggResidual, max_residual_p);
        aggIrmsMax = max(aggIrmsMax, mean_Irms_post);
        aggIrmsMin = min(aggIrmsMin, mean_Irms_post);
        aggDelay = max_or_nan(aggDelay, delay);
    end

    % Extract the new zero-sequence current feature
    I0_rms = get_ts(simOut, 'log_I0_rms');
    
    % Use postMask defined by the last phase (since onset/duration are global per run)
    row.mean_I0_rms_postfault = mean(I0_rms.Data(postMask));

    % Aggregate tier - same names as v8's single-phase features, so a
    % classifier trained on this subset is directly column-comparable to
    % the v8/v10 single-phase feature set (not identical distribution,
    % obviously, but same schema - useful for any future cross-model work).
    row.max_cusum_agg = aggCusum;
    row.max_residual_agg = aggResidual;
    row.mean_Irms_postfault_agg = aggIrmsMax;
    % FIX (post-CSV audit): max() alone is blind to single-phase faults
    % that LOWER current (OpenCircuit_A/B/C, Undervoltage_A/B/C) - the two
    % untouched phases sitting at healthy ~28.8A dominate the max, so
    % mean_Irms_postfault_agg reads identical to Healthy for those fault
    % types even though the affected phase's own _A/_B/_C column shows the
    % drop clearly. min_Irms_postfault_agg and the imbalance feature below
    % close that gap without touching the existing column, so anything
    % already trained against the old schema still works.
    row.min_Irms_postfault_agg = aggIrmsMin;
    row.Irms_imbalance_postfault_agg = aggIrmsMax - aggIrmsMin;   % near-zero for Healthy/balanced 3-phase faults, large for ANY single-phase anomaly regardless of direction
    row.detection_delay_s_agg = fillna(aggDelay, DELAY_SENTINEL);

    breaker_ts = get_ts(simOut, 'log_breaker_state');
    bIdx = find(breaker_ts.Data > 0, 1, 'first');
    if isempty(bIdx)
        row.breaker_open_delay_s = DELAY_SENTINEL;
        row.tripped = 0;
    else
        row.breaker_open_delay_s = breaker_ts.Time(bIdx) - onset;
        row.tripped = 1;
    end

    % Raw Kaggle-style channels: a handful of instantaneous samples
    % during the fault window, NOT full time series (keeps row width
    % sane) - closest honest analog to a single Kaggle classData.csv row.
    Ia = get_ts(simOut, 'log_Ia'); Ib = get_ts(simOut, 'log_Ib'); Ic = get_ts(simOut, 'log_Ic');
    Va = get_ts(simOut, 'log_Va'); Vb = get_ts(simOut, 'log_Vb'); Vc = get_ts(simOut, 'log_Vc');
    sampleIdx = find(Ia.Time >= onset + 0.05, 1, 'first');
    if ~isempty(sampleIdx)
        row.Ia_sample = Ia.Data(sampleIdx); row.Ib_sample = Ib.Data(sampleIdx); row.Ic_sample = Ic.Data(sampleIdx);
        row.Va_sample = Va.Data(sampleIdx); row.Vb_sample = Vb.Data(sampleIdx); row.Vc_sample = Vc.Data(sampleIdx);
    else
        row.Ia_sample = NaN; row.Ib_sample = NaN; row.Ic_sample = NaN;
        row.Va_sample = NaN; row.Vb_sample = NaN; row.Vc_sample = NaN;
    end
end

function ts = get_ts(simOut, varName)
    ts = simOut.get(varName);
end

function v = fillna(v, sentinel)
    if isnan(v), v = sentinel; end
end

function v = max_or_nan(current, candidate)
    if isnan(candidate), v = current; else, v = max(current, candidate); end
end