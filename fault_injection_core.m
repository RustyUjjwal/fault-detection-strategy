%% fault_injection_core.m
%
% CORE FAULT-INJECTION PHYSICS for the three-phase model, shown standalone
% BEFORE the Simulink build script, same reason as before: worth
% reviewing/testing this piece in isolation first.
%
% *** NOTE ON THIS FILE'S PURPOSE: fault_injection_3ph() below is meant to
% become the body of the Fault_Injector MATLAB Function block inside
% Sensing_ThreePhase (see fault_injector_body() in v11_model_code.m for
% the actual embedded copy) - it is NOT meant to be called directly from
% another script. If you need the fault-type list or severity ranges in a
% script, use fault_type_table.m / severity_range_for_mechanism.m /
% mechanism_to_RV.m instead - those are separate, real, externally-
% callable files (see their headers for why they had to be split out). ***
%
% *** SYNC FIX (post-ML-review): this file had drifted out of sync with
% the real model - it still showed the old R_target/V_scale-as-input
% signature from before the MechanismCode fix (see v11_model_code.m
% header for that fix's full history). Brought back in sync: takes
% MechanismCode+Severity directly now, resolving R_target/V_scale inline
% via the same switch/case fault_injector_body() uses, so this reference
% file and the deployed block body can't silently diverge again. ***
%
% TWO INDEPENDENT AXES, exactly as agreed:
%   1. WHICH PHASES ARE INVOLVED - the existing G/A/B/C mask (Kaggle-style,
%      1 = involved), used to pick WHICH of Ia/Ib/Ic get the mechanism's
%      physics applied this run. Unaffected phases stay perfectly healthy.
%   2. FAULT MECHANISM - which physical phenomenon acts on each involved
%      phase: ShortCircuit/Overcurrent (target resistance), Overvoltage/
%      Undervoltage (source voltage scale), OpenCircuit (decay to zero).
%
% TERMINOLOGY NOTE (for your report): LG/LL/LLG are UNSYMMETRICAL faults,
% only LLL/LLLG are SYMMETRICAL (all three phases affected identically).
% All five are still correctly grouped under the ShortCircuit MECHANISM
% in fault_type_table.m - "short-circuit-category" and "symmetrical" are
% not the same claim, don't conflate them in the writeup.
%
% *** GROUND-FAULT PHYSICS ADDED (post-ML-review, confusion-matrix audit):
% ground involvement used to flow straight through to the logged G column
% with NO effect on per-phase R/V - meaning LLG_AB and LL_AB were bit-
% identical simulations under two labels, which a classifier can never
% tell apart (confirmed empirically: every multiclass confusion across
% four models, on two separate sweeps, was exactly these ground-ambiguous
% pairs; see ml_report.md). Fixed with a physically-motivated, minimal
% addition: when a ShortCircuit fault involves ground, model the ground
% return as a SECOND current path in parallel with the phase-to-phase
% fault path (R_ground = 2 Ohm, an assumed neutral-grounding-resistor
% value - real distribution systems commonly use a 1-10 Ohm NGR). Two
% resistances in parallel are always LOWER than either alone
% (R_eff = R_target*R_ground/(R_target+R_ground)), so a grounded fault now
% draws genuinely more current than the otherwise-identical ungrounded
% fault - LLG_AB and LL_AB are no longer the same simulation. Gated to
% MechanismCode==2 (ShortCircuit) only, since G never varies for the other
% four mechanisms in fault_type_table.m (they have no G=1 rows at all), so
% this cannot change any Overcurrent/Overvoltage/Undervoltage/OpenCircuit
% run. RE-RUN verify_physics.m after this change - it numerically checks
% mechanism behavior and does not yet know about this addition. ***

function [Ia, Ib, Ic, Va, Vb, Vc, fault_flag, G, A, B, C] = fault_injection_core( ...
    VaIn, VbIn, VcIn, t, ...
    FaultG, FaultA, FaultB, FaultC, ...   % 0/1 phase-involvement mask, per sweep run
    MechanismCode, Severity, ...          % 1=Healthy..6=OpenCircuit - see mechanism_to_code.m
    RampDuration, FaultOnsetTime, FaultDuration)

    persistent Ia_s Ib_s Ic_s faultSamples initialized
    if isempty(initialized)
        Ia_s = 0; Ib_s = 0; Ic_s = 0; faultSamples = 0; initialized = true;
    end

    Ts = 1e-4;              % MUST match model FixedStep - see v8's same warning
    L = 0.02;                % H, same per-phase inductance as v8's single circuit
    R_normal = 5;             % Ohm, same healthy baseline as v8
    R_STABLE_MAX = 300;       % same forward-Euler stability ceiling as v8's RL_Plant -
                               % above this, R_target is a decay-branch open-circuit
                               % request, not a numerically-valid resistance to plug
                               % into the normal recursion (see v8's changelog on this)

    % Resolve MechanismCode+Severity into R_target/V_scale - same mapping
    % as mechanism_to_RV.m, mirrored inline (see v11_model_code.m header
    % for why this lives inline rather than calling out to that function).
    switch MechanismCode
        case 1
            R_target = R_normal; V_scale = 1;
        case {2, 3}
            R_target = Severity; V_scale = 1;
        case {4, 5}
            R_target = R_normal; V_scale = Severity;
        case 6
            R_target = Severity; V_scale = 1;
        otherwise
            R_target = R_normal; V_scale = 1;
    end

    % GROUND-FAULT PHYSICS: ground return acts as a second current path in
    % parallel with the phase-to-phase fault path, only meaningful for
    % ShortCircuit (the only mechanism where G varies - see header). Two
    % resistances in parallel are always <= the smaller one, so a grounded
    % fault draws MORE current than the otherwise-identical ungrounded
    % fault - this is what makes LLG_AB genuinely different from LL_AB now.
    R_ground = 2;   % Ohm, assumed neutral-grounding-resistor value
    if MechanismCode == 2 && FaultG > 0.5
        R_target = (R_target * R_ground) / (R_target + R_ground);
    end

    fault_active = (t >= FaultOnsetTime) && (t < FaultOnsetTime + FaultDuration);
    if fault_active
        faultSamples = faultSamples + 1;
    else
        faultSamples = 0;
    end
    rampSamples = max(1, round(RampDuration / Ts));
    frac = min(1, faultSamples / rampSamples);   % same ramp-in as v8; RampDuration=0 -> frac=1 instantly

    [Ia_s, Va] = update_phase(Ia_s, VaIn, FaultA, fault_active, frac, R_target, V_scale, R_normal, R_STABLE_MAX, Ts, L);
    [Ib_s, Vb] = update_phase(Ib_s, VbIn, FaultB, fault_active, frac, R_target, V_scale, R_normal, R_STABLE_MAX, Ts, L);
    [Ic_s, Vc] = update_phase(Ic_s, VcIn, FaultC, fault_active, frac, R_target, V_scale, R_normal, R_STABLE_MAX, Ts, L);

    Ia = Ia_s; Ib = Ib_s; Ic = Ic_s;

    fault_flag = fault_active;
    % Pass the mask params straight through to logging so every simulated
    % row can be labeled EXACTLY like a Kaggle row - same column names,
    % same 0/1 convention.
    G = double(FaultG); A = double(FaultA); B = double(FaultB); C = double(FaultC);
end

% Single-phase RL update - literally v8's RL_Plant body, gated by whether
% THIS phase is flagged as involved this run. Kept as a local helper so
% fault_injection_3ph can call it three times here for readability; the
% embedded MATLAB Function block body in v11_model_code.m inlines the
% same logic three times instead (Simulink Function blocks can't call
% local helpers the way a normal .m file can), so if you edit the physics
% here, mirror the edit into fault_injector_body() too.
function [I_next, V_eff] = update_phase(I_prev, V_in, faultFlag, fault_active, frac, R_target, V_scale, R_normal, R_STABLE_MAX, Ts, L)
    if fault_active && faultFlag > 0.5 && R_target > R_STABLE_MAX
        % Open-circuit path: exact discrete decay of a source-free RL
        % circuit, exp(-Ts*R/L) - unconditionally stable for any R, unlike
        % forward-Euler below. Source voltage is unaffected by an open
        % circuit, so V_eff is just passed through.
        R_eff = R_normal + frac * (R_target - R_normal);
        I_next = I_prev * exp(-Ts * R_eff / L);
        V_eff = V_in;
    elseif fault_active && faultFlag > 0.5
        R = R_normal + frac * (R_target - R_normal);
        V_eff = V_in * (1 + frac * (V_scale - 1));
        I_next = I_prev + Ts * (V_eff - R * I_prev) / L;
    else
        V_eff = V_in;
        I_next = I_prev + Ts * (V_eff - R_normal * I_prev) / L;
    end
end
