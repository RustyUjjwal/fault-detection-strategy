function ev = plot_fault_run(csvFile, p, outDir)
% Annotated plots for ONE fault run (Figs 5.1 to 5.17 style).
% p = struct('h',..,'rmax',..,'hs',..,'koc',..)  <- take these from detection_3ph
if nargin < 3, outDir = 'figs'; end
if ~exist(outDir,'dir'), mkdir(outDir); end
T = readtable(csvFile); t = T.Time; [~,nm] = fileparts(csvFile);

cus  = max([T.Cusum_A T.Cusum_B T.Cusum_C],[],2);
ev.onset = first(t, T.fault_flag > 0.5);
ev.cusum = first(t, cus > p.h);
ev.trip  = first(t, T.trip > 0.5);
ev.open  = first(t, T.breaker_state ~= T.breaker_state(1));
ev.delay_cusum_ms = 1e3*(ev.cusum - ev.onset);
ev.delay_trip_ms  = 1e3*(ev.trip  - ev.onset);

% ---- Fig A: currents, Irms+mu0, CUSUM, trip/breaker ----
f = figure('Position',[100 100 900 950],'Color','w'); tl = tiledlayout(4,1,'TileSpacing','compact');
ax = nexttile; plot(t,T.Ia,t,T.Ib,t,T.Ic); ylabel('Current (A)'); legend('Ia','Ib','Ic','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
ax = nexttile; plot(t,T.Irms_A,t,T.Irms_B,t,T.Irms_C); hold on; plot(t,T.Mu0_A,'k--'); ylabel('Irms (A)'); legend('A','B','C','mu0','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
ax = nexttile; plot(t,T.Cusum_A,t,T.Cusum_B,t,T.Cusum_C); yline(p.h,'k--','h'); ylabel('CUSUM'); legend('A','B','C','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
ax = nexttile; stairs(t,T.trip,'LineWidth',1.2); hold on; stairs(t,T.breaker_state); stairs(t,T.fault_flag,':'); ylim([-.2 1.3]);
ylabel('State'); xlabel('Time (s)'); legend('trip','breaker\_state','fault\_flag','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
title(tl, sprintf('%s: onset %.4f s, CUSUM %.1f ms, trip %.1f ms', strrep(nm,'_','\_'), ev.onset, ev.delay_cusum_ms, ev.delay_trip_ms));
save_fig(f, outDir, [nm '_A_currents_cusum_trip']);

% ---- Fig B: voltages, THD, crest ----
f = figure('Position',[100 100 900 760],'Color','w'); tiledlayout(3,1,'TileSpacing','compact');
ax = nexttile; plot(t,T.Va,t,T.Vb,t,T.Vc); ylabel('Voltage (V)'); legend('Va','Vb','Vc','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
ax = nexttile; plot(t,T.THD_A,t,T.THD_B,t,T.THD_C); ylabel('THD'); legend('A','B','C','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
ax = nexttile; plot(t,T.Crest_A,t,T.Crest_B,t,T.Crest_C); yline(sqrt(2),'k--'); ylabel('Crest'); xlabel('Time (s)'); legend('A','B','C','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
save_fig(f, outDir, [nm '_B_voltage_thd_crest']);

% ---- Fig C: sequence components ----
f = figure('Position',[100 100 900 950],'Color','w'); tiledlayout(4,1,'TileSpacing','compact');
ax = nexttile; plot(t,T.I1_seq,t,T.I2_seq,t,T.I0_seq); ylabel('Seq. current'); legend('I1','I2','I0','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
ax = nexttile; plot(t,T.Unbal2,t,T.Unbal0,t,T.UnbalV0); ylabel('Unbalance'); legend('Unbal2','Unbal0','UnbalV0','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
ax = nexttile; plot(t,T.Cusum_U2,t,T.Cusum_U0,t,T.Cusum_V0); yline(p.hs,'k--','h_s'); ylabel('CUSUM unbal.'); legend('U2','U0','V0','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
ax = nexttile; plot(t,T.Phase_A,t,T.Phase_B,t,T.Phase_C); ylabel('Phase'); xlabel('Time (s)'); legend('A','B','C','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
save_fig(f, outDir, [nm '_C_sequence']);

% ---- Fig D: observer residual ----
f = figure('Position',[100 100 900 420],'Color','w'); ax = axes; plot(t,T.Residual_A,t,T.Residual_B,t,T.Residual_C); hold on
yline(p.rmax,'k--','r_{max}'); yline(-p.rmax,'k--'); ylabel('Residual'); xlabel('Time (s)'); legend('A','B','C','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
save_fig(f, outDir, [nm '_D_residual']);

% ---- Fig E: breaker plant, pre vs post currents ----
f = figure('Position',[100 100 900 620],'Color','w'); tiledlayout(2,1,'TileSpacing','compact');
ax = nexttile; plot(t,T.Ia,t,T.Ib,t,T.Ic); ylabel('Ia, Ib, Ic (A)'); ann(ax,ev,t,T.fault_flag);
ax = nexttile; plot(t,T.Ia_post,t,T.Ib_post,t,T.Ic_post); hold on; stairs(t,T.reclose*max(abs(T.Ia_post)),'k:'); ylabel('post-breaker (A)'); xlabel('Time (s)');
legend('Ia\_post','Ib\_post','Ic\_post','reclose','Location','eastoutside'); ann(ax,ev,t,T.fault_flag);
save_fig(f, outDir, [nm '_E_breaker']);
end

function tt = first(t, mask), i = find(mask,1); if isempty(i), tt = NaN; else, tt = t(i); end, end

function ann(ax, ev, t, ff)
% shade fault window, mark onset / CUSUM detection / trip / breaker opening
hold(ax,'on'); yl = ylim(ax); grid(ax,'on');
on = ff > 0.5; if any(on)
    i0 = find(on,1); i1 = find(on,1,'last');
    patch(ax,[t(i0) t(i1) t(i1) t(i0)],[yl(1) yl(1) yl(2) yl(2)],[1 .85 .85],'FaceAlpha',.35,'EdgeColor','none','HandleVisibility','off');
end
m = {ev.onset,'onset','r'; ev.cusum,'CUSUM','b'; ev.trip,'trip','k'; ev.open,'breaker open',[0 .5 0]};
for k = 1:size(m,1)
    if ~isnan(m{k,1}), xline(ax,m{k,1},'--','Color',m{k,3},'Label',m{k,2},'LabelVerticalAlignment','bottom','HandleVisibility','off'); end
end
ylim(ax,yl); uistack(findobj(ax,'Type','patch'),'bottom');
end

function save_fig(f, d, n), exportgraphics(f, fullfile(d,[n '.png']), 'Resolution', 300); close(f); end
