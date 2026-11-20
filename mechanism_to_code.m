%% mechanism_to_code.m
%
% Mechanism label (string) -> MechanismCode (numeric), for the extractor
% to hand the model a plain number instead of a string. Companion to
% mechanism_to_RV.m, which still holds the authoritative R/V mapping used
% by fault_injector_body()'s inlined switch/case - keep the two in sync
% if you ever add a mechanism.
%
% *** ADDED (post-v11 CSV audit): the model used to take Mechanism as a
% string mask popup and derive R_target/V_scale from it inside
% mask.Initialization - that derivation step is what silently broke (see
% v11_model_code.m header for the full diagnosis). Fault_Injector now
% resolves R_target/V_scale itself from a numeric MechanismCode, so the
% extractor needs this function to convert the human-readable label from
% fault_type_table.m into that number before calling set_param. ***
function code = mechanism_to_code(mechanism)
    switch mechanism
        case 'Healthy'
            code = 1;
        case 'ShortCircuit'
            code = 2;
        case 'Overcurrent'
            code = 3;
        case 'Overvoltage'
            code = 4;
        case 'Undervoltage'
            code = 5;
        case 'OpenCircuit'
            code = 6;
        otherwise
            error('mechanism_to_code:unknown', 'Unknown mechanism "%s"', mechanism);
    end
end
