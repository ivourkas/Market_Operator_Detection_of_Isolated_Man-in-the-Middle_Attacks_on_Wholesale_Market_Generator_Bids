function spaOPFsmlF_oodStress(casef, fmx, its, scenarioName, loadProfileCsv)
% spaOPFsmlF_oodStress
% Generate out-of-distribution coordinated-attack OPF cases.
%
% This file is intentionally separate from spaOPFsmlF.m. The original
% single-generator/single-parameter data generator remains untouched; this
% stress-test generator is only for the revision experiment where a model
% trained on the original single-point data is tested on unseen coordinated
% attack patterns.
%
% Inputs
%   casef          MATPOWER case name or path.
%   fmx            Load-randomization factor, clipped to [0, 0.2].
%   its            Number of load blocks to solve.
%   scenarioName   One of:
%                    'two_generators_same_parameter'
%                    'one_generator_two_parameters'
%   loadProfileCsv Optional CSV of original held-out N300 normal load profiles.
%
% Output
%   A CSV file with the same OPF-result layout as spaOPFsmlF.m, plus extra
%   metadata rows documenting the coordinated stress-test scenario.

scenarioName = validatestring( ...
    scenarioName, ...
    {'two_generators_same_parameter', 'one_generator_two_parameters'}, ...
    mfilename, ...
    'scenarioName');

fmx = max(0, min(0.2, fmx));

optoptns = optimoptions('linprog', 'Display', 'none');

runStamp = strcat(num2str(date), num2str(hour(datetime('now'))));
runStamp = strcat(runStamp, num2str(minute(datetime('now'))));
runStamp = strcat(runStamp, num2str(round(second(datetime('now')))));

mpc = loadcase(casef);
mpc = ext2int(mpc);
YBUS = full(makeYbus(mpc));
NB = size(mpc.bus, 1);
NG = size(mpc.gen, 1);

Pg = zeros(1, NB);
Pd = mpc.bus(:, 3);
for i = 1:NG
    Pg(1, mpc.gen(i, 1)) = mpc.gen(i, 9);
end
Pg = Pg / mpc.baseMVA;
Pd = -Pd / mpc.baseMVA;

useMatchedLoadProfiles = nargin >= 5 && ~isempty(loadProfileCsv);
if useMatchedLoadProfiles
    [matchedProfiles, profileIndexBySeason] = readMatchedLoadProfiles(loadProfileCsv);
    availableLoadBlocks = size(profileIndexBySeason, 1);
    its = min(its, availableLoadBlocks);
    if its < 1
        error('At least one matched load block is required.');
    end
    loadModeLabel = 'MatchedHeldout';
else
    matchedProfiles = table();
    profileIndexBySeason = [];
    loadModeLabel = 'RandomLoad';
end

scenarioCode = oodScenarioCode(scenarioName);
fln = strcat( ...
    runStamp, ...
    '_OOD_', scenarioName, ...
    '_', loadModeLabel, ...
    '_N', num2str(NB), ...
    '_G', num2str(NG), ...
    '.csv');

fprintf('OOD stress-test generator initialized.\n');
fprintf('  Case: %s\n', casef);
fprintf('  Scenario: %s (%d)\n', scenarioName, scenarioCode);
fprintf('  Load mode: %s\n', loadModeLabel);
if useMatchedLoadProfiles
    fprintf('  Matched load profile CSV: %s\n', loadProfileCsv);
    fprintf('  Matched four-season blocks available: %d\n', availableLoadBlocks);
else
    fprintf('  Load randomization fmx: %.3f\n', fmx);
end
fprintf('  Iterations: %d\n', its);
fprintf('  Output: %s\n', fln);

if ~useMatchedLoadProfiles
    mx = (1 - fmx) * min(sum(Pg), sum(abs(Pd)));

    Win = mx * [0.625, 0.675, 0.665, 0.655, 0.655, 0.675, 0.75, 0.8, 0.70, 0.65, 0.64, 0.63, 0.62, 0.6, 0.6, 0.64, 0.8, 0.85, 0.875, 0.85, 0.825, 0.775, 0.7, 0.65];
    Spr = mx * [0.675, 0.7, 0.685, 0.68, 0.675, 0.67, 0.725, 0.775, 0.775, 0.7, 0.675, 0.65, 0.6, 0.575, 0.55, 0.55, 0.575, 0.6, 0.7, 0.775, 0.775, 0.75, 0.725, 0.7];
    Sum = mx * [0.75, 0.65, 0.6, 0.55, 0.525, 0.515, 0.5, 0.525, 0.625, 0.775, 0.875, 0.925, 0.92, 0.875, 0.92, 0.92, 0.92, 0.975, 1, 0.975, 0.95, 0.95, 0.92, 0.875];
    Aut = mx * [0.625, 0.525, 0.475, 0.45, 0.425, 0.425, 0.45, 0.5, 0.6, 0.7, 0.75, 0.775, 0.77, 0.765, 0.76, 0.755, 0.745, 0.745, 0.755, 0.8, 0.875, 0.925, 0.915, 0.75];
end

pr2f = [];

for iters = 1:its
    if useMatchedLoadProfiles
        [TotalLoad, minhrLd, profileMetadata] = buildMatchedSeasonalLoads( ...
            matchedProfiles, profileIndexBySeason, iters, Pd);
    else
        [TotalLoad, minhrLd] = drawSeasonalLoads(Pd, fmx, Win, Spr, Sum, Aut);
        profileMetadata = emptyProfileMetadata();
    end
    [BigAeqTot, BigbeqTot, BiglbTot, BigubTot, ElX] = buildSeasonOpfMatrices( ...
        YBUS, TotalLoad, Pg, mpc, NG);
    [BigA, Bigb, RampRate, nzg] = buildRampConstraints(mpc, Pg, ElX);
    [Bigf, mxgc, mngc] = buildCostVector(mpc, ElX);
    [BigxTot, fvalTot] = solveSeasonOpfs( ...
        Bigf, BigA, Bigb, BigAeqTot, BigbeqTot, BiglbTot, BigubTot, optoptns);
    attackPlan = drawCoordinatedAttackPlan(scenarioName, scenarioCode, nzg);
    [BigAAtt, BigbAtt, BiglbTotAtt, BigubTotAtt, BigfAtt, attackPlan] = applyCoordinatedAttack( ...
        attackPlan, BigA, Bigb, BiglbTot, BigubTot, Bigf, RampRate, Pg, mpc, ElX, minhrLd, mxgc, mngc, nzg);
    [BigxTotATT, fvalTotATT] = solveSeasonOpfs( ...
        BigfAtt, BigAAtt, BigbAtt, BigAeqTot, BigbeqTot, BiglbTotAtt, BigubTotAtt, optoptns);

    fprintf('  Prepared load scenario %d/%d, minimum hourly load %.4f p.u.\n', ...
        iters, its, minhrLd);
    fprintf('    Normal seasonal objective values: %.4f %.4f %.4f %.4f\n', fvalTot);
    fprintf('    Attack seasonal objective values: %.4f %.4f %.4f %.4f\n', fvalTotATT);
    printAttackPlan(attackPlan);

    pr2f = appendStressRows( ...
        pr2f, ...
        BigxTot, ...
        fvalTot, ...
        BigxTotATT, ...
        fvalTotATT, ...
        attackPlan, ...
        profileMetadata);
end

dlmwrite(fln, full(pr2f), '-append');
fprintf('Saved OOD stress-test data: %s\n', fln);

end

function [TotalLoad, minhrLd] = drawSeasonalLoads(Pd, fmx, Win, Spr, Sum, Aut)
% Draw one stochastic 96-hour load block using the original seasonal model.

fc = fmx * rand;

WinMax = Win + Win * fc;
WinMin = Win - Win * fc;
SprMax = Spr + Spr * fc;
SprMin = Spr - Spr * fc;
SumMax = Sum + Sum * fc;
SumMin = Sum - Sum * fc;
AutMax = Aut + Aut * fc;
AutMin = Aut - Aut * fc;

Winter = (WinMax - WinMin) .* rand(1, 24) + WinMin;
Spring = (SprMax - SprMin) .* rand(1, 24) + SprMin;
Summer = (SumMax - SumMin) .* rand(1, 24) + SumMin;
Autumn = (AutMax - AutMin) .* rand(1, 24) + AutMin;

WinLoad = Winter' * Pd' / abs(sum(Pd));
SprLoad = Spring' * Pd' / abs(sum(Pd));
SumLoad = Summer' * Pd' / abs(sum(Pd));
AutLoad = Autumn' * Pd' / abs(sum(Pd));

TotalLoad = sparse([WinLoad; SprLoad; SumLoad; AutLoad]);
minhrLd = full(min(abs(sum(TotalLoad, 2))));

end

function [TotalLoad, minhrLd, profileMetadata] = buildMatchedSeasonalLoads( ...
    matchedProfiles, profileIndexBySeason, loadBlock, Pd)
% Build one 96-hour load block from original held-out normal profiles.

loadColumns = hourlyLoadColumnNames();
seasonLoadBlocks = [];
profileMetadata = emptyProfileMetadata();

for season = 1:4
    profileRow = profileIndexBySeason(loadBlock, season);
    hourlyTotalLoad = table2array(matchedProfiles(profileRow, loadColumns));
    seasonLoad = hourlyTotalLoad' * Pd' / abs(sum(Pd));
    seasonLoadBlocks = [seasonLoadBlocks; seasonLoad];
    profileMetadata.matched_profile_id(season) = matchedProfiles.matched_profile_id(profileRow);
    profileMetadata.matched_raw_row_index(season) = matchedProfiles.raw_row_index(profileRow);
end

TotalLoad = sparse(seasonLoadBlocks);
minhrLd = full(min(abs(sum(TotalLoad, 2))));

end

function profileMetadata = emptyProfileMetadata()
% Metadata placeholders for random-load mode and matched-load initialization.

profileMetadata = struct( ...
    'matched_profile_id', zeros(1, 4), ...
    'matched_raw_row_index', zeros(1, 4));

end

function [BigAeqTot, BigbeqTot, BiglbTot, BigubTot, ElX] = buildSeasonOpfMatrices( ...
    YBUS, TotalLoad, Pg, mpc, NG)
% Build the four 24-hour DC-OPF blocks used for the seasonal solves.

NB = size(mpc.bus, 1);
ElX = 2 * length(find(triu(YBUS, 1))) + 2 * NB;

BigAeqTot = [];
BigbeqTot = [];
BiglbTot = [];
BigubTot = [];

for season = 1:4
    firstHour = 24 * (season - 1) + 1;
    Load = TotalLoad(firstHour:firstHour + 23, :);

    BigAeq = [];
    Bigbeq = [];
    Biglb = [];
    Bigub = [];

    for hour = 1:24
        [Aeq, beq, lb, ub] = lnzdOPF_fe(YBUS, Load(hour, :), Pg);

        for g = 1:NG
            lb(mpc.gen(g, 1)) = mpc.gen(g, 10) / mpc.baseMVA;
        end

        BigAeq = sparse(blkdiag(BigAeq, Aeq));
        Bigbeq = sparse([Bigbeq; beq]);
        Biglb = sparse([Biglb; lb]);
        Bigub = sparse([Bigub; ub]);
    end

    BigAeqTot = sparse([BigAeqTot, BigAeq]);
    BigbeqTot = sparse([BigbeqTot, Bigbeq]);
    BiglbTot = sparse([BiglbTot, Biglb]);
    BigubTot = sparse([BigubTot, Bigub]);
end

end

function [BigA, Bigb, RampRate, nzg] = buildRampConstraints(mpc, Pg, ElX)
% Create 24-hour ramp constraints using the original data-generator rule.

genlabel = transpose(1:size(mpc.gen, 1));
gen = [genlabel, mpc.gen(:, 9)];
gen = gen(find(gen(:, 2)), :);
gen = sortrows(gen, 2);

len = length(mpc.gen(:, 9));
smallGenerators = gen(1:round(size(gen, 1) / 2), 1);

RampRate = zeros(1, len);
RampRate(smallGenerators) = 0.5;
RampRate(find(RampRate == 0)) = 0.3;

nzg = find(mpc.gen(:, 9))';
BigA = zeros(1, 24 * ElX);
Bigb = 0;
cntr = 1;

for i = nzg
    for hour = 1:23
        BigA(cntr, (ElX * (hour - 1) + mpc.gen(i, 1))) = 1;
        BigA(cntr, (ElX * hour + mpc.gen(i, 1))) = -1;
        Bigb(1, cntr) = RampRate(1, i) * Pg(1, mpc.gen(i, 1));
        cntr = cntr + 1;

        BigA(cntr, (ElX * (hour - 1) + mpc.gen(i, 1))) = -1;
        BigA(cntr, (ElX * hour + mpc.gen(i, 1))) = 1;
        Bigb(1, cntr) = RampRate(1, i) * Pg(1, mpc.gen(i, 1));
        cntr = cntr + 1;
    end
end

BigA = sparse(BigA);

end

function [Bigf, mxgc, mngc] = buildCostVector(mpc, ElX)
% Build the repeated 24-hour linear cost vector.

NG = size(mpc.gen, 1);
f = zeros(1, ElX);
for i = 1:NG
    f(1, mpc.gen(i, 1)) = mpc.gencost(i, 4 + mpc.gencost(i, 4) - 1);
end

mxgc = max(f) * 1.2;
mngc = min(nonzeros(f)) * 0.5;
Bigf = repmat(f, 1, 24);

end

function [BigxTot, fvalTot] = solveSeasonOpfs( ...
    Bigf, BigA, Bigb, BigAeqTot, BigbeqTot, BiglbTot, BigubTot, optoptns)
% Solve one 24-hour OPF for each season and preserve infeasibility flags.

BigxTot = [];
fvalTot = [];
seasonVariables = size(BigAeqTot, 2) / 4;

for season = 1:4
    firstColumn = seasonVariables * (season - 1) + 1;
    lastColumn = firstColumn + seasonVariables - 1;

    [Bigx, fval] = linprog( ...
        Bigf, ...
        BigA, ...
        Bigb, ...
        BigAeqTot(:, firstColumn:lastColumn), ...
        BigbeqTot(:, season), ...
        BiglbTot(:, season), ...
        BigubTot(:, season), ...
        optoptns);

    if isempty(Bigx)
        Bigx = -100000 * ones(seasonVariables, 1);
        fval = -100000;
    end

    BigxTot = [BigxTot, Bigx];
    fvalTot = [fvalTot, fval];
end

end

function [BigAAtt, BigbAtt, BiglbTotAtt, BigubTotAtt, BigfAtt, attackPlan] = applyCoordinatedAttack( ...
    attackPlan, BigA, Bigb, BiglbTot, BigubTot, Bigf, RampRate, Pg, mpc, ElX, minhrLd, mxgc, mngc, nzg)
% Apply the two planned perturbations to copied OPF matrices.

BigAAtt = BigA;
BigbAtt = Bigb;
BiglbTotAtt = BiglbTot;
BigubTotAtt = BigubTot;
BigfAtt = Bigf;

[BigAAtt, BigbAtt, BiglbTotAtt, BigubTotAtt, BigfAtt, primaryAlpha] = applySingleAttack( ...
    attackPlan.primary_attack_type, ...
    attackPlan.primary_generator, ...
    BigAAtt, ...
    BigbAtt, ...
    BiglbTotAtt, ...
    BigubTotAtt, ...
    BigfAtt, ...
    RampRate, ...
    Pg, ...
    mpc, ...
    ElX, ...
    minhrLd, ...
    mxgc, ...
    mngc, ...
    nzg);

[BigAAtt, BigbAtt, BiglbTotAtt, BigubTotAtt, BigfAtt, secondaryAlpha] = applySingleAttack( ...
    attackPlan.secondary_attack_type, ...
    attackPlan.secondary_generator, ...
    BigAAtt, ...
    BigbAtt, ...
    BiglbTotAtt, ...
    BigubTotAtt, ...
    BigfAtt, ...
    RampRate, ...
    Pg, ...
    mpc, ...
    ElX, ...
    minhrLd, ...
    mxgc, ...
    mngc, ...
    nzg);

attackPlan.primary_alpha = primaryAlpha;
attackPlan.secondary_alpha = secondaryAlpha;

end

function pr2f = appendStressRows( ...
    pr2f, BigxTot, fvalTot, BigxTotATT, fvalTotATT, attackPlan, profileMetadata)
% Store matched normal and attacked seasonal OPF outcomes with OOD metadata.

seasonLabels = 1:4;
normalRows = [ ...
    BigxTot; ...
    fvalTot; ...
    seasonLabels; ...
    zeros(1, 4); ...
    zeros(1, 4); ...
    attackPlan.scenario_code * ones(1, 4); ...
    zeros(1, 4); ...
    zeros(1, 4); ...
    zeros(1, 4); ...
    zeros(1, 4); ...
    profileMetadata.matched_profile_id; ...
    profileMetadata.matched_raw_row_index];

attackedRows = [ ...
    BigxTotATT; ...
    fvalTotATT; ...
    seasonLabels; ...
    attackPlan.primary_attack_type * ones(1, 4); ...
    attackPlan.primary_generator * ones(1, 4); ...
    attackPlan.scenario_code * ones(1, 4); ...
    attackPlan.secondary_attack_type * ones(1, 4); ...
    attackPlan.secondary_generator * ones(1, 4); ...
    attackPlan.primary_alpha * ones(1, 4); ...
    attackPlan.secondary_alpha * ones(1, 4); ...
    profileMetadata.matched_profile_id; ...
    profileMetadata.matched_raw_row_index];

pr2f = sparse([pr2f, normalRows, attackedRows]);

end

function [BigA, Bigb, BiglbTot, BigubTot, Bigf, alpha] = applySingleAttack( ...
    attackType, attgen, BigA, Bigb, BiglbTot, BigubTot, Bigf, RampRate, Pg, mpc, ElX, minhrLd, mxgc, mngc, nzg)
% Apply one original attack primitive to one generator.

switch attackType
    case 1
        alpha = 0.01 + 0.99 * rand;
        attackedRampRate = RampRate(1, attgen) * alpha;
        generatorPosition = find(nzg == attgen);

        firstConstraint = 23 * 2 * (generatorPosition - 1) + 1;
        lastConstraint = firstConstraint + (23 * 2) - 1;
        for row = firstConstraint:lastConstraint
            Bigb(1, row) = attackedRampRate * Pg(1, mpc.gen(attgen, 1));
        end

    case 2
        alpha = 0.01 + 0.99 * rand;
        for hour = 1:24
            row = ((hour - 1) * ElX) + mpc.gen(attgen, 1);
            BigubTot(row, :) = BigubTot(row, :) * alpha;
        end

    case 3
        alpha = 0.01 + 0.99 * rand;
        for hour = 1:24
            row = ((hour - 1) * ElX) + mpc.gen(attgen, 1);
            BiglbTot(row, :) = min(BigubTot(row, :) * alpha, 0.95 * minhrLd);
        end

    case 4
        alpha = mngc + (mxgc - mngc) * rand;
        for hour = 1:24
            col = ((hour - 1) * ElX) + mpc.gen(attgen, 1);
            Bigf(1, col) = alpha;
        end

    otherwise
        error('Unknown attack type: %d', attackType);
end

end

function attackPlan = drawCoordinatedAttackPlan(scenarioName, scenarioCode, nzg)
% Draw the coordinated attack structure without changing OPF matrices yet.

attackTypes = 1:4;

switch scenarioName
    case 'two_generators_same_parameter'
        chosenGenerators = nzg(randperm(length(nzg), 2));
        sharedAttackType = attackTypes(randi(length(attackTypes)));

        attackPlan = struct( ...
            'scenario_name', scenarioName, ...
            'scenario_code', scenarioCode, ...
            'primary_generator', chosenGenerators(1), ...
            'secondary_generator', chosenGenerators(2), ...
            'primary_attack_type', sharedAttackType, ...
            'secondary_attack_type', sharedAttackType);

    case 'one_generator_two_parameters'
        chosenGenerator = nzg(randi(length(nzg)));
        chosenAttackTypes = attackTypes(randperm(length(attackTypes), 2));
        chosenAttackTypes = orderSameGeneratorAttackTypes(chosenAttackTypes);

        attackPlan = struct( ...
            'scenario_name', scenarioName, ...
            'scenario_code', scenarioCode, ...
            'primary_generator', chosenGenerator, ...
            'secondary_generator', chosenGenerator, ...
            'primary_attack_type', chosenAttackTypes(1), ...
            'secondary_attack_type', chosenAttackTypes(2));

    otherwise
        error('Unknown OOD scenario: %s', scenarioName);
end

end

function orderedAttackTypes = orderSameGeneratorAttackTypes(attackTypes)
% Keep paired capacity-limit attacks feasible when both limits are selected.

orderedAttackTypes = attackTypes;
if all(ismember([2, 3], attackTypes))
    orderedAttackTypes = [2, 3];
end

end

function printAttackPlan(attackPlan)
% Print the planned coordinated attack in human-readable form.

fprintf('    OOD plan: %s | G%d:%s + G%d:%s\n', ...
    attackPlan.scenario_name, ...
    attackPlan.primary_generator, ...
    attackTypeName(attackPlan.primary_attack_type), ...
    attackPlan.secondary_generator, ...
    attackTypeName(attackPlan.secondary_attack_type));

end

function name = attackTypeName(attackType)
% Human-readable attack-type names matching the original generator.

switch attackType
    case 1
        name = 'ramp_rate';
    case 2
        name = 'upper_limit';
    case 3
        name = 'lower_limit';
    case 4
        name = 'economic_bid';
    otherwise
        error('Unknown attack type: %d', attackType);
end

end

function code = oodScenarioCode(scenarioName)
% Stable numeric code for downstream CSV parsing.

switch scenarioName
    case 'two_generators_same_parameter'
        code = 101;
    case 'one_generator_two_parameters'
        code = 102;
    otherwise
        error('Unknown OOD scenario: %s', scenarioName);
end

end

function [matchedProfiles, profileIndexBySeason] = readMatchedLoadProfiles(loadProfileCsv)
% Read held-out N300 normal load profiles and group them by season.

if ~isfile(loadProfileCsv)
    error('Matched load profile CSV not found: %s', loadProfileCsv);
end

matchedProfiles = readtable(loadProfileCsv, 'VariableNamingRule', 'preserve');
loadColumns = hourlyLoadColumnNames();
requiredColumns = [{'matched_profile_id', 'raw_row_index', 'original_ssn'}, loadColumns];

missingColumns = setdiff(requiredColumns, matchedProfiles.Properties.VariableNames);
if ~isempty(missingColumns)
    error('Matched load profile CSV is missing required columns: %s', strjoin(missingColumns, ', '));
end

if height(matchedProfiles) == 0
    error('Matched load profile CSV is empty.');
end

if numel(unique(matchedProfiles.matched_profile_id)) ~= height(matchedProfiles)
    error('Matched load profile IDs are not unique.');
end

if numel(unique(matchedProfiles.raw_row_index)) ~= height(matchedProfiles)
    error('Matched raw row indices are not unique.');
end

seasonLabels = matchedProfiles.original_ssn;
if any(~ismember(seasonLabels, 1:4))
    error('Matched load profiles must use original_ssn values 1, 2, 3, and 4.');
end

loadMatrix = table2array(matchedProfiles(:, loadColumns));
if any(~isfinite(loadMatrix), 'all')
    error('Matched load profile CSV contains NaN or infinite hourly loads.');
end

if any(loadMatrix <= 0, 'all')
    error('Matched load profile CSV contains non-positive hourly loads.');
end

rowsBySeason = cell(1, 4);
seasonCounts = zeros(1, 4);
for season = 1:4
    seasonRows = find(seasonLabels == season);
    seasonRows = sortProfileRows(matchedProfiles, seasonRows);
    rowsBySeason{season} = seasonRows;
    seasonCounts(season) = numel(seasonRows);
end

if any(seasonCounts == 0) || numel(unique(seasonCounts)) ~= 1
    error('Matched load profiles must contain the same number of rows for each season.');
end

profileIndexBySeason = zeros(seasonCounts(1), 4);
for season = 1:4
    profileIndexBySeason(:, season) = rowsBySeason{season};
end

end

function sortedRows = sortProfileRows(matchedProfiles, rowIndices)
% Keep matched profile rows in the same temporal order as the original test set.

if ismember('temporal_index', matchedProfiles.Properties.VariableNames)
    [~, order] = sort(matchedProfiles.temporal_index(rowIndices));
elseif ismember('matched_profile_id', matchedProfiles.Properties.VariableNames)
    [~, order] = sort(matchedProfiles.matched_profile_id(rowIndices));
else
    order = 1:numel(rowIndices);
end

sortedRows = rowIndices(order);

end

function loadColumns = hourlyLoadColumnNames()
% Column names used by scripts/export_n300_heldout_normal_loads.py.

loadColumns = cell(1, 24);
for hour = 1:24
    loadColumns{hour} = sprintf('load_h%02d', hour);
end

end
