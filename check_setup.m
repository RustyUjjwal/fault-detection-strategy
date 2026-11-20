function check_setup(model)
% Run this ONCE before plot_fault_run.  Usage:  check_setup('fault_detection_system_v6')
% It prints (1) solver settings, (2) constant blocks, (3) threshold lines from every
% MATLAB Function block, (4) which log_ variables exist vs what export_run_to_csv expects,
% (5) what breaker_state and fault_flag actually look like.
load_system(model);

fprintf('=== 1. Solver settings ===\n');
fprintf('Solver %s | FixedStep %s | StopTime %s\n', get_param(model,'Solver'), ...
    get_param(model,'FixedStep'), get_param(model,'StopTime'));

fprintf('\n=== 2. Constant blocks (fault settings, delays, noise seed) ===\n');
cb = find_system(model,'LookUnderMasks','all','BlockType','Constant');
for i = 1:numel(cb), fprintf('%-75s %s\n', cb{i}, get_param(cb{i},'Value')); end

fprintf('\n=== 3. Threshold-like lines inside MATLAB Function blocks ===\n');
pat = '(thr|thresh|limit|drift|\<h\>|\<k\>|rmax|koc|settle|warmup|alpha|gain|\<L\>|=\s*[0-9.]+e?-?[0-9]*;)';
rt = sfroot; charts = rt.find('-isa','Stateflow.EMChart');
for c = 1:numel(charts)
    if ~startsWith(charts(c).Path, model), continue; end
    L = splitlines(string(charts(c).Script));
    keep = ~cellfun('isempty', regexpi(cellstr(L), pat)) & ~startsWith(strtrim(L),'%');
    fprintf('\n[%s]\n', charts(c).Path); fprintf('%s\n', L(keep));
end

fprintf('\n=== 4. Logged variables ===\n');
expected = {'Ia','Ib','Ic','Va','Vb','Vc','Irms_A','Irms_B','Irms_C','Crest_A','Crest_B','Crest_C', ...
 'THD_A','THD_B','THD_C','Phase_A','Phase_B','Phase_C','Cusum_A','Cusum_B','Cusum_C', ...
 'Residual_A','Residual_B','Residual_C','Mu0_A','Mu0_B','Mu0_C','trip','breaker_state','fault_flag', ...
 'G','A','B','C','I1_seq','I2_seq','I0_seq','Unbal2','Unbal0','UnbalV0','Cusum_U2','Cusum_U0','Cusum_V0', ...
 'Ia_post','Ib_post','Ic_post','reclose'};
out = sim(model);
try, vars = out.who; catch, vars = properties(out); end
logs = vars(startsWith(vars,'log_'));
fprintf('%d log_ variables found, %d expected by export_run_to_csv\n', numel(logs), numel(expected));
fprintf('Expected but NOT in model : %s\n', strjoin(setdiff(strcat('log_',expected), logs), ', '));
fprintf('In model but not expected : %s\n', strjoin(setdiff(logs, strcat('log_',expected)), ', '));

fprintf('\n=== 5. breaker_state / fault_flag / trip behaviour ===\n');
t = out.tout;
for nm = {'log_breaker_state','log_fault_flag','log_trip'}
    if ~ismember(nm{1}, vars), continue; end
    x = out.(nm{1}); if isa(x,'timeseries'), x = x.Data; end, x = x(:);
    ch = find(diff(x) ~= 0);
    fprintf('%s: values {%s}, changes at t = %s s\n', nm{1}, num2str(unique(round(x))'), ...
        num2str(t(min(ch+1, numel(t)))', '%.4f '));
end
end
