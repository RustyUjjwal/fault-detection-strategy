function export_run_to_csv(out, csvFile)
% Save every logged signal of one simulation into ONE csv (Time + all columns).
% Usage:  out = sim('fault_detection_system_v6');  export_run_to_csv(out,'runs/LG_A.csv')
names = {'Ia','Ib','Ic','Va','Vb','Vc', ...
 'Irms_A','Irms_B','Irms_C','Crest_A','Crest_B','Crest_C','THD_A','THD_B','THD_C', ...
 'Phase_A','Phase_B','Phase_C','Cusum_A','Cusum_B','Cusum_C','Residual_A','Residual_B','Residual_C', ...
 'Mu0_A','Mu0_B','Mu0_C','trip','breaker_state','fault_flag','G','A','B','C', ...
 'I1_seq','I2_seq','I0_seq','Unbal2','Unbal0','UnbalV0','Cusum_U2','Cusum_U0','Cusum_V0', ...
 'Ia_post','Ib_post','Ic_post','reclose'};
t = out.tout(:);
T = table(t, 'VariableNames', {'Time'});
for k = 1:numel(names)
    x = out.(['log_' names{k}]);              % name used in the To Workspace blocks
    if isa(x,'timeseries'), v = x.Data(:); else, v = x(:); end
    if numel(v) ~= numel(t), v = interp1(linspace(0,1,numel(v)), v, linspace(0,1,numel(t)))'; end
    T.(names{k}) = v;
end
[d,~,~] = fileparts(csvFile); if ~isempty(d) && ~exist(d,'dir'), mkdir(d); end
writetable(T, csvFile);
end
