%% severity_range_for_mechanism.m
%
% Severity draw range per mechanism - copied verbatim from v10's
% severityRangeByType (build_fault_detection_model.m / v10_data_extractor_code.m),
% the exact ranges already validated on the single-phase pipeline (final
% v10 classifiers hit 1.000 accuracy against data generated with these
% ranges). ShortCircuit's range is reused as-is for all 11 location-based
% ShortCircuit rows in fault_type_table.m.
%
% Split into its own file for the same reason as fault_type_table.m - see
% that file's header.
function range = severity_range_for_mechanism(mechanism)
    switch mechanism
        case 'Healthy'
            range = [1     1];      % ignored - dummy value, matches v8/v10
        case 'ShortCircuit'
            range = [0.3   0.8];    % target R (Ohm), nominal was 0.5
        case 'Overcurrent'
            range = [1.5   3.5];    % target R (Ohm), nominal was 2.5
        case 'Overvoltage'
            range = [1.15  1.5];    % V scale factor, nominal was 1.3
        case 'Undervoltage'
            range = [0.4   0.75];   % V scale factor, nominal was 0.6
        case 'OpenCircuit'
            range = [500   5000];   % target R (Ohm) - must stay > R_STABLE_MAX (300)
                                      % to trigger the decay branch, same as v10
        otherwise
            error('severity_range_for_mechanism:unknown', 'Unknown mechanism "%s"', mechanism);
    end
end
