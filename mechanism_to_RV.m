%% mechanism_to_RV.m
%
% Mechanism + Severity -> (R_target, V_scale). Literal port of v8's
% mask.Initialization switch/case (build_fault_detection_model.m,
% Sensing_Simulated mask). Kept as its own callable file, on purpose, so
% BOTH the Simulink mask (v11_model_code.m's Sensing_ThreePhase
% mask.Initialization calls this directly, same base/mask workspace
% MATLAB code can call any function on the path) and this script's own
% physics-verification test call the exact same logic - no risk of the
% mask's inline copy drifting from what actually got tested.
function [R_target, V_scale] = mechanism_to_RV(mechanism, severity)
    R_normal = 5;
    switch mechanism
        case 'Healthy'
            R_target = R_normal; V_scale = 1;
        case {'ShortCircuit', 'Overcurrent'}
            R_target = severity; V_scale = 1;
        case {'Overvoltage', 'Undervoltage'}
            R_target = R_normal; V_scale = severity;
        case 'OpenCircuit'
            R_target = severity; V_scale = 1;   % must exceed R_STABLE_MAX (300) to actually
                                                  % trigger the decay branch - see
                                                  % severity_range_for_mechanism.m
        otherwise
            error('mechanism_to_RV:unknown', 'Unknown mechanism "%s"', mechanism);
    end
end
