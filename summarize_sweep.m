function R = summarize_sweep(folder, p)
% Loops over every run csv in a folder -> Table 5.4 (delays), Fig 5.18 (bar), Fig 5.19 (detector heat map)
files = dir(fullfile(folder,'*.csv')); files = files(~strcmp({files.name},'delay_table.csv'));
R = table();
for k = 1:numel(files)
    f = fullfile(folder, files(k).name); T = readtable(f); t = T.Time;
    ev = plot_fault_run(f, p, fullfile(folder,'figs'));
    cus = max([T.Cusum_A T.Cusum_B T.Cusum_C],[],2);
    obs = max(abs([T.Residual_A T.Residual_B T.Residual_C]),[],2);
    sq  = max([T.Cusum_U2 T.Cusum_U0 T.Cusum_V0],[],2);
    oc  = max([T.Irms_A./T.Mu0_A T.Irms_B./T.Mu0_B T.Irms_C./T.Mu0_C],[],2);
    on = ev.onset; post = t >= on;                      % only look after fault onset
    row.Fault = string(erase(files(k).name,'.csv')); row.Onset_s = on;
    row.CUSUM_ms = d(t, cus>p.h, post, on); row.Observer_ms = d(t, obs>p.rmax, post, on);
    row.SeqCUSUM_ms = d(t, sq>p.hs, post, on); row.Overcurrent_ms = d(t, oc>p.koc, post, on);
    row.Trip_ms = 1e3*(ev.trip-on); row.FalseTrip_before_onset = any(T.trip(t<on)>0.5);
    R = [R; struct2table(row)]; %#ok<AGROW>
end
writetable(R, fullfile(folder,'delay_table.csv'));
% Fig 5.18: detection (trip) delay by fault type
f = figure('Color','w','Position',[100 100 950 450]); bar(categorical(R.Fault), R.Trip_ms); ylabel('Trip delay after onset (ms)'); grid on; xtickangle(45);
exportgraphics(f, fullfile(folder,'Fig_5_18_delay.png'),'Resolution',300);
% Fig 5.19: detector coverage heat map
M = ~isnan(R{:,{'CUSUM_ms','Observer_ms','SeqCUSUM_ms','Overcurrent_ms'}});
f = figure('Color','w','Position',[100 100 700 600]); imagesc(M); colormap([.9 .9 .9; .2 .6 .3]);
set(gca,'XTick',1:4,'XTickLabel',{'CUSUM','Observer','Seq CUSUM','Overcurrent'},'YTick',1:height(R),'YTickLabel',R.Fault);
title('Detected (green) / missed (grey)'); exportgraphics(f, fullfile(folder,'Fig_5_19_coverage.png'),'Resolution',300);
end

function ms = d(t, cond, post, on)
i = find(cond & post, 1); if isempty(i), ms = NaN; else, ms = 1e3*(t(i)-on); end
end
