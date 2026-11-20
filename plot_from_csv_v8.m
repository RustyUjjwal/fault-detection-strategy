%% plot_from_csv_v8.m
% Plots the fault timelines from fault_timeseries_all.csv written by fault_timeseries_dump_v8.
% Each figure has the line currents, the rms currents, the CUSUM statistics and the trip / breaker
% state, for the whole run (left) and zoomed on the fault start (right).
%
% USAGE:
%   >> plot_from_csv_v8                                   % defaults below
%   >> plot_from_csv_v8('fault_timeseries_all.csv', {'LG_A','LLL'}, 1)
% Needs MATLAB R2020a or newer (tiledlayout, xline).

function plot_from_csv_v8(csvFile, faults, fb)
if nargin < 1, csvFile = fullfile('fault_analysis', 'fault_timeseries_all.csv'); end
if nargin < 2, faults = {'LG_A','LL_AB','LLL','Overcurrent_A'}; end
if nargin < 3, fb = 1; end                  % 1 = breaker acts on the plant, 0 = bolted fault

T = readtable(csvFile);
T.fault_type = string(T.fault_type);
for k = 1:numel(faults)
    timeline(T, faults{k}, fb);
end
mechanisms(T);
end

function timeline(T, name, fb)
H = 0.5; CAP = 5.0;                         % CUSUM threshold and cap of the current channels
ZOOM = [0.395 0.465];
col = [0.165 0.471 0.839; 0.922 0.408 0.204; 0.106 0.686 0.478];
g = T(T.fault_type == string(name) & T.feedback == fb, :);
if isempty(g), warning('No rows for %s with feedback %d', name, fb); return; end
t = g.time_s;
onset = first_time(t, g.fault_flag > 0.5);          % the injector's own fault flag = true onset
tTrip = first_time(t, g.trip > 0.5);
tOpen = first_time(t, g.breaker_state > 0.5);
tRec = first_time(t, g.reclose > 0.5);
iAct = find(g.fault_flag > 0.5);
if isempty(iAct), clear_t = NaN; else, clear_t = t(iAct(end)); end

fig = figure('Color', 'w', 'Position', [100 80 1100 900]);
tl = tiledlayout(4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
phs = 'ABC';
rowNames = {'line current (A)', 'Irms (A)', 'CUSUM statistic', ''};
xl = {[0 t(end)], ZOOM};
for r = 1:4
    for j = 1:2
        ax = nexttile(tl, (r-1)*2 + j); hold(ax, 'on'); grid(ax, 'on');
        m = t >= xl{j}(1) & t <= xl{j}(2);
        for p = 1:3
            ph = phs(p);
            switch r
                case 1, y = g.(['I' lower(ph)]);
                case 2, y = g.(['Irms_' ph]);
                case 3, y = g.(['Cusum_' ph]);
                otherwise, y = [];
            end
            if r <= 3, plot(ax, t(m), y(m), 'Color', col(p,:), 'LineWidth', 1.2, 'DisplayName', ['phase ' ph]); end
        end
        if r == 3
            yline(ax, H, '--k'); yline(ax, CAP, ':', 'Color', [0.5 0.5 0.5]); ylim(ax, [-0.3 CAP*1.18]);
        end
        if r == 4
            area(ax, t(m), 0.7 * double(g.trip(m) > 0.5), 'FaceColor', [0.89 0.29 0.28], 'EdgeColor', 'none', 'DisplayName', 'trip command');
            area(ax, t(m), 0.7 * double(g.breaker_state(m) > 0.5) + 1.5, 1.5, 'FaceColor', [0.38 0.31 0.84], 'EdgeColor', 'none', 'DisplayName', 'breaker open');
            ylim(ax, [-0.1 2.4]); yticks(ax, []); xlabel(ax, 'time (s)');
        end
        for xv = [onset clear_t tRec]
            if ~isnan(xv), xline(ax, xv, ':', 'Color', [0.5 0.5 0.5]); end
        end
        xlim(ax, xl{j});
        if j == 1 && r <= 3, ylabel(ax, rowNames{r}); end
        if r == 1 && j == 1, legend(ax, 'Location', 'northeast'); title(ax, 'whole run'); end
        if r == 1 && j == 2, title(ax, sprintf('zoom %.3f to %.3f s', ZOOM(1), ZOOM(2))); end
        if r == 4 && j == 1, legend(ax, 'Location', 'west'); end
    end
end
txt = sprintf('%s, breaker feedback %d', strrep(name, '_', '\_'), fb);
if ~isnan(tTrip)
    txt = sprintf('%s   |   trip +%.1f ms', txt, (tTrip - onset)*1e3);
    if ~isnan(tOpen), txt = sprintf('%s, contacts open +%.1f ms', txt, (tOpen - onset)*1e3); end
end
txt = sprintf('%s, peak |I| %.0f A', txt, max(abs([g.Ia; g.Ib; g.Ic])));
title(tl, txt, 'FontWeight', 'bold');
exportgraphics(fig, sprintf('timeline_%s_fb%d.png', name, fb), 'Resolution', 160);
end

function mechanisms(T)
cases = {'Healthy','healthy','--',[0.33 0.33 0.33]; 'Overcurrent_A','overcurrent','-',[0.165 0.471 0.839]; ...
         'Overvoltage_A','overvoltage','-',[0.922 0.408 0.204]; 'Undervoltage_A','undervoltage','-',[0.106 0.686 0.478]; ...
         'OpenCircuit_A','open circuit','-',[0.93 0.63 0]};
fig = figure('Color', 'w', 'Position', [100 100 800 420]); ax = axes(fig); hold(ax, 'on'); grid(ax, 'on');
for k = 1:size(cases, 1)
    g = T(T.fault_type == string(cases{k,1}) & T.feedback == 0, :);
    if ~isempty(g), plot(ax, g.time_s, g.Irms_A, cases{k,3}, 'Color', cases{k,4}, 'LineWidth', 1.6, 'DisplayName', cases{k,2}); end
end
xlabel(ax, 'time (s)'); ylabel(ax, 'Irms of phase A (A)'); title(ax, 'Phase A rms current, breaker feedback off');
legend(ax, 'Location', 'south', 'NumColumns', 5);
exportgraphics(fig, 'mechanisms.png', 'Resolution', 160);
end

function x = first_time(t, mask)
i = find(mask, 1, 'first');
if isempty(i), x = NaN; else, x = t(i); end
end
