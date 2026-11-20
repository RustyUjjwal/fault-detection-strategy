%% sensitivity_sweep_v8.m   (for fault_detection_system_v8)
%
% Produces the data for Section 5.11 of the report:
%   Figure 5.21  detection delay against fault severity (and the smallest severity that is still detected)
%   Figure 5.22  healthy-run CUSUM maximum for different noise seeds (and delay spread for a mild fault)
%
% Outputs (in fault_analysis/):
%   sensitivity_severity.csv   one row per (fault, severity): tripped, trip delay, detection delay
%   sensitivity_noise.csv      one row per (case, seed): max CUSUM, max residual, tripped, delay
%   fig_5_21.png, fig_5_22.png quick MATLAB versions of the two figures
%
% Detection delay = trip delay minus the 5 ms trip confirmation (CONFIRM_SAMPLES in trip_latch).
% Breaker feedback is off, so only the detector is exercised.
%
% Needs on the path: fault_type_table.m, mechanism_to_code.m (same as the other scripts).
%
% USAGE:
%   >> model_code_v8
%   >> sensitivity_sweep_v8

function sensitivity_sweep_v8()

tic;

%% ---- settings ----
mdl = 'fault_detection_system_v8';
onset = 0.4; faultDuration = 0.3; stopSev = 0.6;     % short runs: the trip happens within about 10 ms
stopNoise = 2.0;                                     % healthy noise runs use the full 2 s
CONFIRM_MS = 5.0;                                    % trip confirmation time in trip_latch
SOURCE_L = 5e-3; BREAKER_DELAY = 0.04; RECLOSE_DELAY = 0.1;
outDir = 'fault_analysis';
IMG_RES = 130;

% severity grids (the report ranges are in severity_range_for_mechanism.m; the grids extend towards
% the healthy value so that the smallest detectable severity can be found)
sweeps = struct( ...
    'fault',    {'LG_A',  'Overcurrent_A', 'Overvoltage_A', 'Undervoltage_A', 'OpenCircuit_A'}, ...
    'label',    {'LG fault: fault resistance (ohm)', 'Overcurrent: load resistance (ohm)', 'Overvoltage: voltage multiplier', 'Undervoltage: voltage multiplier', 'Open circuit: series resistance (ohm)'}, ...
    'values',   { [0.3 0.55 0.8 1 2 5 10 20 50], [1.5 2 2.5 3 3.5 4 4.5 4.8 4.9 4.95], ...
                  [1.005 1.01 1.02 1.03 1.05 1.1 1.15 1.3 1.5], [0.4 0.6 0.75 0.9 0.95 0.97 0.98 0.99 0.995], ...
                  [6 10 20 50 100 500 2750 5000] });

% noise sweep: healthy run and one mild fault, 30 seeds each
seeds = 5001:5030;
mildFault = 'Overcurrent_A'; mildSeverity = 4.5;     % 4.5 ohm load: a 5 percent current rise

if ~bdIsLoaded(mdl), load_system([mdl '.slx']); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
faultTypes = fault_type_table();

%% ---- 1. severity sweep ----
sevRows = {};
set_param(mdl, 'StopTime', num2str(stopSev));
for s = 1:numel(sweeps)
    ft = find(strcmp(faultTypes(:,1), sweeps(s).fault), 1);
    for sev = sweeps(s).values
        fprintf('severity sweep: %-16s severity=%g\n', sweeps(s).fault, sev);
        simOut = run_case(mdl, faultTypes(ft,:), sev, 3001, onset, faultDuration, SOURCE_L, BREAKER_DELAY, RECLOSE_DELAY);
        trip = simOut.get('log_trip');
        k = find(trip.Data > 0, 1, 'first');
        r = struct('fault', sweeps(s).fault, 'severity', sev);
        if isempty(k)
            r.tripped = 0; r.trip_delay_ms = NaN; r.detection_delay_ms = NaN;
        else
            r.tripped = 1; r.trip_delay_ms = (trip.Time(k) - onset) * 1e3; r.detection_delay_ms = r.trip_delay_ms - CONFIRM_MS;
        end
        sevRows{end+1} = r; %#ok<AGROW>
    end
end
TS = struct2table(vertcat(sevRows{:}));
writetable(TS, fullfile(outDir, 'sensitivity_severity.csv'));

%% ---- 2. noise sweep ----
set_param(mdl, 'StopTime', num2str(stopNoise));
noiseRows = {};
ftH = find(strcmp(faultTypes(:,1), 'Healthy'), 1);
ftM = find(strcmp(faultTypes(:,1), mildFault), 1);
for seed = seeds
    fprintf('noise sweep: seed %d\n', seed);
    % healthy
    simOut = run_case(mdl, faultTypes(ftH,:), 1, seed, onset, faultDuration, SOURCE_L, BREAKER_DELAY, RECLOSE_DELAY);
    r = struct('case', 'Healthy', 'seed', seed);
    r.max_cusum = max([max(simOut.get('log_Cusum_A').Data), max(simOut.get('log_Cusum_B').Data), max(simOut.get('log_Cusum_C').Data)]);
    r.max_residual = max([max(simOut.get('log_Residual_A').Data), max(simOut.get('log_Residual_B').Data), max(simOut.get('log_Residual_C').Data)]);
    r.tripped = double(any(simOut.get('log_trip').Data > 0)); r.detection_delay_ms = NaN;
    noiseRows{end+1} = r; %#ok<AGROW>
    % mild fault
    simOut = run_case(mdl, faultTypes(ftM,:), mildSeverity, seed, onset, faultDuration, SOURCE_L, BREAKER_DELAY, RECLOSE_DELAY);
    trip = simOut.get('log_trip'); k = find(trip.Data > 0 & trip.Time >= onset, 1, 'first');
    r = struct('case', mildFault, 'seed', seed);
    m = simOut.get('log_Cusum_A').Time >= onset;
    r.max_cusum = max(simOut.get('log_Cusum_A').Data(m)); r.max_residual = max(simOut.get('log_Residual_A').Data(m));
    if isempty(k), r.tripped = 0; r.detection_delay_ms = NaN;
    else, r.tripped = 1; r.detection_delay_ms = (trip.Time(k) - onset) * 1e3 - CONFIRM_MS; end
    noiseRows{end+1} = r; %#ok<AGROW>
end
TN = struct2table(vertcat(noiseRows{:}));
writetable(TN, fullfile(outDir, 'sensitivity_noise.csv'));

%% ---- 3. quick figures ----
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1500 700]);
for s = 1:numel(sweeps)
    ax = subplot(2, 3, s); hold(ax, 'on'); grid(ax, 'on');
    m = strcmp(TS.fault, sweeps(s).fault);
    sev = TS.severity(m); dly = TS.detection_delay_ms(m); trp = TS.tripped(m) > 0;
    plot(ax, sev(trp), dly(trp), 'o-');
    if any(~trp), plot(ax, sev(~trp), zeros(1, nnz(~trp)), 'rx', 'MarkerSize', 9, 'LineWidth', 1.5); end
    xlabel(ax, sweeps(s).label); ylabel(ax, 'Detection delay (ms)'); title(ax, strrep(sweeps(s).fault, '_', '\_'));
    if any(~trp), legend(ax, {'detected', 'not detected'}, 'Location', 'best'); end
end
sgtitle('Detection delay against fault severity (red x = not detected)');
write_png(fig, fullfile(outDir, 'fig_5_21.png'), IMG_RES); close(fig);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1300 450]);
mH = strcmp(TN.case, 'Healthy'); mF = strcmp(TN.case, mildFault);
ax = subplot(1, 2, 1); plot(ax, TN.seed(mH), TN.max_cusum(mH), 'o-'); hold(ax, 'on'); yline(ax, 0.5, '--k', 'threshold h = 0.5'); grid(ax, 'on');
xlabel(ax, 'Noise seed'); ylabel(ax, 'Largest CUSUM value in the healthy run'); title(ax, sprintf('Healthy run (%d false trips in %d seeds)', sum(TN.tripped(mH)), nnz(mH)));
ax = subplot(1, 2, 2); plot(ax, TN.seed(mF), TN.detection_delay_ms(mF), 's-'); grid(ax, 'on');
xlabel(ax, 'Noise seed'); ylabel(ax, 'Detection delay (ms)'); title(ax, sprintf('%s, %.2g ohm: delay for different noise seeds', strrep(mildFault, '_', '\_'), mildSeverity));
write_png(fig, fullfile(outDir, 'fig_5_22.png'), IMG_RES); close(fig);

fprintf('\nWrote %s, %s and two figures in %s/\n', 'sensitivity_severity.csv', 'sensitivity_noise.csv', outDir);
fprintf('Total time: %.1f s (%.1f min)\n', toc, toc/60);

end

%% =====================================================================
function simOut = run_case(mdl, ftRow, severity, seed, onset, fd, sourceL, bDelay, rDelay)
    G = ftRow{3}; A = ftRow{4}; B = ftRow{5}; C = ftRow{6};
    set_param([mdl '/Sensing_ThreePhase'], ...
        'FaultG', onoff(G), 'FaultA', onoff(A), 'FaultB', onoff(B), 'FaultC', onoff(C), ...
        'MechanismCode', num2str(mechanism_to_code(ftRow{2})), 'Severity', num2str(severity, 8), 'RampDuration', '0', ...
        'FaultOnsetTime', sprintf('%.6f', onset), 'FaultDuration', sprintf('%.6f', fd), ...
        'SourceL', sprintf('%g', sourceL), 'ReclosDelay', sprintf('%.6f', rDelay));
    set_param([mdl '/Signal_Conditioning_3ph'], 'NoiseSeed', num2str(seed));
    set_param([mdl '/Breaker_Contactor_Plant_3ph'], 'FeedbackEnable', '0', 'BreakerDelay', sprintf('%.6f', bDelay));
    simOut = sim(mdl);
end

function s = onoff(flag)
    if flag, s = 'on'; else, s = 'off'; end
end

function write_png(fig, fileName, res)
    try
        exportgraphics(fig, fileName, 'Resolution', res);
    catch
        set(fig, 'PaperPositionMode', 'auto');
        print(fig, fileName, '-dpng', sprintf('-r%d', res));
    end
end
