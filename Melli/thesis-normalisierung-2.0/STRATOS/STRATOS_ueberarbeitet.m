% ========================================================================
% STRATOS_all_in_one.m
% End-to-end: Phantom -> Projektionen -> LGS-Reko -> Auswertung
% - Arbeitet mit XCAT-Dateien: phantom_XX_gt.par_act_1.bin / _atn_1.bin
% - Projektor akzeptiert ACT entweder als Pfad (string) ODER als 3D-Array
%
% ÜBERARBEITET: alles außer betrachtete Organe hat KEINE Aktivität!
% ========================================================================

clear; close all; clc;

%% =================== USER CONFIG ===================
% --- XCAT / Output ---
par_template = 'C:\BerndsDaten\XCAT\XCAT_Phantom\general.samp.par';
output_dir   = 'C:\BerndsDaten\XCAT\XCAT_Phantom\Melanie_Ender';
xcat_bin     = 'C:\BerndsDaten\XCAT\XCAT_Phantom\dxcat2';

% --- Kollimator / Volumendimensionen ---
collmat = "C:\Users\Admin_mbd\Documents\MATLAB\Melanie_Ender\Bachelorthesis_PaulWenzel_GammakameraSim\Kollimatoren\hexagonal\LEAP_Kernel.mat";
dims    = [256,256,650];      % [nx, ny, nz] (achte auf 650 vs. 651 nach deiner PAR)

% --- Voxelgeometrie (nur für Dämpfung/PSF in diesem Script NICHT benutzt) ---
phys.dx_cm = 0.2; phys.dy_cm = 0.2; phys.dz_cm = 0.2;

% --- Kernel-Parameter (wie gehabt) ---
kernel.z0_slices = 29;
kernel.dz_cm     = phys.dz_cm;

% --- Scatter-Parameter ---
use_scatter = 1;    % 0 = aus, 1 = an
sigma       = 2.0;  % <- zurück auf deinen „guten“ Wert

% --- RRMSE-ROI (Pixel pro Rand; 0 = Full-FOV) ---
roi_border_px = 16;

% --- organ_ids.txt (Pfad anpassen) ---
organ_ids_file = 'C:\BerndsDaten\XCAT\XCAT_Phantom\organ_ids.txt';

% --- Organgruppen & Aktivitätsbereiche (dein Bestand) ---
organ_groups_params = {
    'liver', {'liver_activity'};
    'kidney', {'r_kidney_cortex_activity','l_kidney_cortex_activity','r_kidney_medulla_activity','l_kidney_medulla_activity'};
    'prostate', {'prostate_activity'};
    'small_intest', {'sm_intest_activity'};
    'spleen', {'spleen_activity'};
    'others', {'brain_activity','laryngopharynx_activity','eye_activity','lens_activity', 'myoLV_act', 'myoRV_act', 'myoLA_act', 'myoRA_act', ...
        'body_activity','bldplLV_act', 'bldplRV_act', 'bldplLA_act', 'bldplRA_act', ...
        'coronary_art_activity', 'coronary_vein_activity', 'valve_thickness', ...
        'skin_activity', 'rbreast_activity', 'lbreast_activity', 'muscle_activity', ...
        'sinus_activity', 'gall_bladder_activity', 'r_lung_activity', 'l_lung_activity', ...
        'esophagus_activity', 'esophagus_cont_activity', 'larynx_activity', 'st_wall_activity', ...
        'st_cnts_activity', 'pancreas_activity', 'adrenal_activity', 'r_renal_pelvis_activity', ...
        'l_renal_pelvis_activity', 'rib_activity', 'cortical_bone_activity', 'spine_activity', ...
        'spinal_cord_activity', 'bone_marrow_activity', 'art_activity', 'vein_activity', ...
        'bladder_activity', 'asc_li_activity', 'trans_li_activity', 'desc_li_activity', ...
        'rectum_activity', 'sem_activity', 'vas_def_activity', 'test_activity', 'penis_activity', ...
        'epididymus_activity', 'ejac_duct_activity', 'pericardium_activity', 'cartilage_activity', ...
        'intest_air_activity', 'ureter_activity', 'urethra_activity', 'lymph_activity', ...
        'lymph_abnormal_activity', 'trach_bronch_activity', 'airway_activity', 'uterus_activity', ...
        'vagina_activity', 'right_ovary_activity', 'left_ovary_activity', 'fallopian_tubes_activity', ...
        'parathyroid_activity', 'thyroid_activity', 'thymus_activity', 'salivary_activity', ...
        'pituitary_activity', 'lesn_activity', 'Corpus_Callosum_act', 'Caudate_act', ...
        'Internal_capsule_act', 'Putamen_act', 'Globus_pallidus_act', 'Thalamus_act', ...
        'Fornix_act', 'Anterior_commissure_act', 'Amygdala_act', 'Hippocampus_act', ...
        'Lateral_ventricle_act', 'Third_ventricle_act', 'Fourth_ventricle_act', 'Cerebral_aqueduct_act', ...
        'Mamillary_bodies_act', 'Cerebral_peduncles_act', 'Superior_colliculus_act', ...
        'Inferior_colliculus_act', 'Pineal_gland_act', 'Periacquaductal_grey_outer_act', ...
        'Periacquaductal_grey_act', 'Pons_act', 'Superior_cerebellar_peduncle_act', ...
        'Middle_cerebellar_peduncle_act', 'Substantia_nigra_act', 'Medulla_act', ...
        'Medullary_pyramids_act', 'Inferior_olive_act', 'Tegmentum_of_midbrain_act', ...
        'Midbrain_act', 'cerebellum_act', 'white_matter_act', 'grey_matter_act'};};

activity_ranges = struct( ...
    'liver',      [3, 7], ...
    'kidney',     [10, 35], ...
    'prostate',   [35, 50], ...
    'small_intest',[3, 5], ...
    'spleen',     [3, 5], ...
    'others',     [0, 0] ...
);

%% =================== HAUPTLAUF ===================

% --- Ordner anlegen & nächsten freien Index ermitteln ---
if ~exist(output_dir,'dir'); mkdir(output_dir); end
[next_idx, next_name] = next_phantom_idx(output_dir);
fprintf('Nächster freier Phantom-Index ist %d (=%s)\n', next_idx, next_name);

N = 1;
phantoms    = cell(N,1);
results_all = cell(N,1);

% --- Erzeugen & Simulieren: Indizes next_idx .. next_idx+N-1 ---
for k = 1:N
    ii = next_idx + (k-1);   % tatsächliche Phantom-Nummer auf der Platte
    phantoms{k}    = create_phantom(ii, output_dir, par_template, xcat_bin, organ_groups_params, activity_ranges);
    results_all{k} = simulate_phantom(phantoms{k}, collmat, dims, organ_ids_file, phys, kernel, use_scatter, sigma);
end

% --- Auswertung/Plots/Speichern (unverändert, aber über k laufen) ---
for k = 1:N
    phantom = phantoms{k};
    res     = results_all{k};

    figure('Renderer','painters','Name',['Phantom ' phantom.Name ' Projektionen'],'NumberTitle','off');
    subplot(2,2,1); imagesc(res.proj_AP);     axis image; colorbar; title('GT AP');
    subplot(2,2,2); imagesc(res.proj_AP_rec); axis image; colorbar; title('Rekonstruktion AP');
    subplot(2,2,3); imagesc(res.proj_PA);     axis image; colorbar; title('GT PA');
    subplot(2,2,4); imagesc(res.proj_PA_rec); axis image; colorbar; title('Rekonstruktion PA');

    [rrmse_ap, alpha_ap] = rrmse_alpha(res.proj_AP, res.proj_AP_rec, roi_border_px);
    [rrmse_pa, alpha_pa] = rrmse_alpha(res.proj_PA, res.proj_PA_rec, roi_border_px);

    fprintf('\n== %s ==\n', phantom.Name);
    fprintf('RRMSE (AP) = %.2f %%  (alpha = %.3f)\n', 100*rrmse_ap, alpha_ap);
    fprintf('RRMSE (PA) = %.2f %%  (alpha = %.3f)\n', 100*rrmse_pa, alpha_pa);

    if isfield(res,'true_x') && isfield(res,'x_est')
        figure('Renderer','painters','Name',['Phantom ' phantom.Name ' Organ Activities'],'NumberTitle','off');
        bar([res.true_x(:), res.x_est(:)]);
        legend('Ground Truth','Reconstruction');
        if isfield(res,'organs'), xticks(1:length(res.organs)); xticklabels(res.organs); end
        ylabel('Activity'); title(['Organ activities: GT vs Reconstruction for ' phantom.Name]);
    end

    % --- Speichern der Projektionen als .mat ---
    save_path = fullfile(phantom.Dir, sprintf('%s_projections.mat', phantom.Name));
    proj_AP      = res.proj_AP;
    proj_PA      = res.proj_PA;
    proj_AP_rec  = res.proj_AP_rec;
    proj_PA_rec  = res.proj_PA_rec;
    save(save_path, 'proj_AP', 'proj_PA', 'proj_AP_rec', 'proj_PA_rec', '-v7.3');
    fprintf('Projektionen gespeichert in:\n  %s\n', save_path);
end


%% =================== FUNKTIONEN ===================

% -------------------- simulate_phantom -------------
function results = simulate_phantom(phantom_info, collmat, dims, organ_ids_file, phys, kernel, use_scatter, sigma)
    % Dateinamen
    fnameATN   = phantom_info.GTParFile + "_atn_1.bin";
    fnameACTgt = phantom_info.GTParFile + "_act_1.bin";

    % --- 1) GT-Projektionen ---
    [proj_AP, proj_PA] = Gammakamera_modified( ...
        fnameATN, fnameACTgt, "frontal", ...
        use_scatter, 1, 1, collmat, ...
        dims(1), dims(2), dims(3), sigma, false, ...
        phys, kernel);

    % >>> Patch #1: identische Orientierung wie bei den A-Spalten <<<
    proj_AP = rot90(proj_AP); proj_AP = flipud(proj_AP); proj_AP = proj_AP';
    proj_PA = rot90(proj_PA); proj_PA = flipud(proj_PA); proj_PA = proj_PA';

    % Messvektor b durch Zeilenvektorisierung und Hintereinanderschalten
    % (PA an AP hängen)
    b = [proj_AP(:); proj_PA(:)];

    % --- 2) Organ-IDs & Gruppen ---
    fnameACTmask = phantom_info.MaskParFile + "_act_1.bin";
    act_data_ids = readRawDataCubeFromDisk(fnameACTmask, 0, 0, dims, 'float32', 'n');           % liest das ID-Volumen (Maskenlauf), in dem jedes Voxel eine Organ-ID hat

    orgID = readlines(organ_ids_file);
    names = strings(length(orgID),1);
    ids   = zeros(length(orgID),1);
    for i = 1:length(orgID)
        line = strtrim(orgID(i));
        if isempty(line), continue; end
        parts = split(line,'=');
        if numel(parts)~=2, warning('Zeile %d unlesbar: %s', i, line); continue; end
        names(i) = strtrim(parts(1));
        ids(i)   = str2double(strtrim(parts(2)));
    end

    liver_ids       = ids(contains(names, 'liver'));
    kidney_ids      = ids(contains(names, 'kidney'));
    prostate_ids    = ids(contains(names, 'prostate'));
    smallintest_ids = ids(contains(names, 'small_intest'));
    spleen_ids      = ids(contains(names, 'spleen'));
    all_ids         = unique(ids);
    main_ids        = [liver_ids; kidney_ids; prostate_ids; smallintest_ids; spleen_ids];
    rest_ids        = setdiff(all_ids, main_ids);

    organ_groups = {
        'liver',      {'liver_activity'},        liver_ids;
        'kidney',     {'r_kidney_cortex_activity','l_kidney_cortex_activity','r_kidney_medulla_activity','l_kidney_medulla_activity'}, kidney_ids;
        'prostate',   {'prostate_activity'},     prostate_ids;
        'smallintest',{'sm_intest_activity'},    smallintest_ids;
        'spleen',     {'spleen_activity'},       spleen_ids;
        'others',     phantom_info.Activities(end).Parameter, rest_ids;
    };

    comp=1; atn=1; coll=1; activity_per_voxel=1;                            % Maskenwert pro Voxe ist 1 --> A skaliert die Projektione pro Einheitsaktivität

    % --- 3) Dummy-Projektion (Größencheck) ---
    dummy = zeros(dims,'single');
    [proj_AP_dummy, proj_PA_dummy] = Gammakamera_modified( ...
        fnameATN, dummy, "frontal", comp, atn, coll, collmat, ...
        dims(1), dims(2), dims(3), sigma, false, phys, kernel);
    % gleiche Orientierung
    proj_AP_dummy = rot90(proj_AP_dummy); proj_AP_dummy = flipud(proj_AP_dummy); proj_AP_dummy = proj_AP_dummy';
    proj_PA_dummy = rot90(proj_PA_dummy); proj_PA_dummy = flipud(proj_PA_dummy); proj_PA_dummy = proj_PA_dummy';

    b_len = numel(proj_AP_dummy) + numel(proj_PA_dummy);
    N_org = length(organ_groups);
    A = zeros(b_len, N_org, 'double');

    % --- 4) A-Spalten ---
    for i = 1:N_org
        group_ids = organ_groups{i,3};
        if iscell(group_ids), group_ids = cell2mat(group_ids); end                      % holt ID-Liste der Gruppe --> normalisiert auf num. Vektor

        mask_group = zeros(size(act_data_ids), 'single');                               % Binärmaske (1 in Voxeln der Gruppe, sonst 0)
        for j = 1:length(group_ids)
            mask_group(round(act_data_ids) == group_ids(j)) = activity_per_voxel;
        end

        [proj_APi, proj_PAi] = Gammakamera_modified( ...                                % projiziert nur diese Gruppe (bei Einheitsaktivität pro Voxel)
            fnameATN, mask_group, "frontal", comp, atn, coll, collmat, ...
            dims(1), dims(2), dims(3), sigma, false, phys, kernel);

        % gleiche Orientierung wie b (Patch)
        proj_APi = rot90(proj_APi); proj_APi = flipud(proj_APi); proj_APi = proj_APi';
        proj_PAi = rot90(proj_PAi); proj_PAi = flipud(proj_PAi); proj_PAi = proj_PAi';

        A(:,i) = [proj_APi(:); proj_PAi(:)];
    end

    % --- 5) NNLS (ohne Regularisierung) — Patch #2 ---
    lambda = 0.0; scalingFactor = 1e5;                                                  % scalingFactor verbessert die Kondition (rein numerisch), weil b und A oft kleine Werte haben
    A_reg = [A*scalingFactor; sqrt(lambda)*eye(N_org)];
    b_reg = [b*scalingFactor; zeros(N_org,1)];
    x_est = lsqnonneg(A_reg, b_reg);

    % --- Optional: LS-Untergrenze als Sanity-Check ---
    x_ls   = A \ b;
    b_hat  = A * x_ls;
    rrmse_lb = norm(b - b_hat) / norm(b);
    fprintf('LS-Untergrenze (ohne NNLS/λ): %.2f %%\n', 100*rrmse_lb);

    % --- 6) GT-Aktivitäten (wie gehabt, Reihenfolgeannahme) ---
    true_x = zeros(N_org,1);
    organs = cell(N_org,1);
    for i = 1:N_org
        true_x(i) = mean([phantom_info.Activities(i).Activity]);
        organs{i} = organ_groups{i,1};
    end

    % --- 7) Reprojektion ---
    b_est = A * x_est;
    proj_AP_rec = reshape(b_est(1:numel(proj_AP)), size(proj_AP));
    proj_PA_rec = reshape(b_est(numel(proj_AP)+1:end), size(proj_PA));

    % --- 8) Fehlermaße ---
    rmse  = sqrt(mean((b - b_est).^2));
    rrmse = norm(b - b_est)/norm(b);
    fprintf('Phantom %s: RMSE = %.6e, RRMSE = %.4f\n', phantom_info.Name, rmse, rrmse);

    % --- 9) Ergebnisse ---
    results = struct();
    results.x_est = x_est;
    results.true_x = true_x;
    results.organs = organs;
    results.rmse = rmse;
    results.rrmse = rrmse;
    results.proj_AP = proj_AP;
    results.proj_PA = proj_PA;
    results.proj_AP_rec = proj_AP_rec;
    results.proj_PA_rec = proj_PA_rec;
end

% -------------------- Gammakamera_modified -------------

function [proj_AP, proj_PA] = Gammakamera_modified(fnameATN,fnameACT_or_vol,view,comp,atn,coll,collmat,dim1,dim2,dim3,sigma, saveFigures, phys, kernel)
% Vorwärtsprojektor
if nargin < 12, saveFigures = false; end

% ATN
if ~isfile(fnameATN), error('ATN-Datei fehlt: %s', string(fnameATN)); end
atn_data = load_volume_flexible(fnameATN, [dim1,dim2,dim3]);

% ACT (Datei ODER Volumen)
if ischar(fnameACT_or_vol) || isstring(fnameACT_or_vol)
    if ~isfile(fnameACT_or_vol), error('ACT-Datei fehlt: %s', string(fnameACT_or_vol)); end
    act_data = load_volume_flexible(fnameACT_or_vol, [dim1,dim2,dim3]);
else
    act_data = fnameACT_or_vol;
    assert(all(size(act_data) == [dim1,dim2,dim3]), 'ACT-Volumen hat falsche Dimensionen.');
end

% Kernel
S = load(collmat,'kernel_mat'); kernel_mat = S.kernel_mat;

if view == "frontal"
    act_ap = rot90(permute(act_data,[1 3 2]));  atn_ap = rot90(permute(atn_data,[1 3 2]));
    act_pa = flip(act_ap,3);                     atn_pa = flip(atn_ap,3);

    proj_AP = processView_phys(act_ap, atn_ap, "AP", comp, atn, coll, kernel_mat, sigma, phys, kernel, saveFigures);
    proj_PA = processView_phys(act_pa, atn_pa, "PA", comp, atn, coll, kernel_mat, sigma, phys, kernel, saveFigures);
elseif view == "sagittal"
    act_sag = rot90(permute(act_data,[2 3 1]));
    atn_sag = rot90(permute(atn_data,[2 3 1]));
    proj_AP = processView_phys(act_sag, atn_sag, "Sagittal", comp, atn, coll, kernel_mat, sigma, phys, kernel, saveFigures);
    proj_PA = [];
else
    error('Unsupported view: %s', string(view));
end
end

% -------------------- processView_phys -------------

function total_proj = processView_phys(act_data, atn_data, label, comp, atn, coll, kernel_mat, sigma, phys, kernel, saveFigures)
% Pipeline: (1) Scatter (2D je Slice, falls an) (2) Attenuation (3) Collimator (ab z=30) (4) Summe

% (1) Scatter (2D je Slice; belässt z-Struktur)
% 3D würde entlang z zwischen Schichten mischen  und damit die
% Tiefenunschärfe doppelt zählen, die später durch z-Integration der
% Attenuation und tiefenabhängiger PSF ohnehin schon entsteht
if comp == 1
    act_scattered = zeros(size(act_data),'like',act_data);
    for z = 1:size(act_data,3)
        act_scattered(:,:,z) = imgaussfilt(act_data(:,:,z), sigma);
    end
else
    act_scattered = act_data;
end

% (2) Dämpfung
% exponentielle Dämpfung entlang der Projektionsachse (z)
if atn == 1
    cumatn = cumsum(atn_data, 3);
    mat_atn = act_scattered .* exp(-cumatn);
else
    mat_atn = act_scattered;
end

% (3) Collimator (ab z=30, zz=z-29)
if coll == 1
    Z = size(mat_atn,3);
    mat_coll = zeros(size(mat_atn),'like',mat_atn);
    for z = 30:Z
        zz = z - 29;
        zz = min(zz, size(kernel_mat,3));
        K = kernel_mat(:,:,zz);
        mat_coll(:,:,z) = conv2(mat_atn(:,:,z), K, 'same');
    end
else
    mat_coll = mat_atn;
end

% (4) Projektion (z-Summation)
total_proj = sum(mat_coll, 3);

if saveFigures
    f = figure('Name',['All Effects - ' char(label)],'NumberTitle','off');
    imagesc(total_proj); axis image; colorbar;
end
end

% -------------------- load_volume_flexible -------------

function vol = load_volume_flexible(src, dims)
% .bin wird als float32/nativ gelesen
src = string(src);
if endsWith(lower(src), ".bin")
    vol = readRawDataCubeFromDisk(src, 0, 0, dims, 'float32', 'n');
else
    vol = readRawDataCubeFromDisk(src, 0, 0, dims);
end
end

% -------------------- rrmse_alpha -------------

function [rrmse, alpha] = rrmse_alpha(gt, rec, roi_border_px)
if nargin < 3, roi_border_px = 0; end
if roi_border_px > 0
    gt  = central_roi(gt, roi_border_px);
    rec = central_roi(rec, roi_border_px);
end
gtv  = double(gt(:));
recv = double(rec(:));
alpha = (recv' * gtv) / max(recv' * recv, eps);
rrmse = norm(alpha*recv - gtv) / max(norm(gtv), eps);
end

% -------------------- central_roi -------------

function A = central_roi(A, b)
if b > 0 && all(size(A) > 2*b)
    A = A(1+b:end-b, 1+b:end-b);
end
end


% -------------------- create_phantom -------------

function phantom_info = create_phantom(i, output_dir, par_template, xcat_bin, organ_groups_params, activity_ranges)
    phantom_name = sprintf('phantom_%02d', i);

    % Unterordner pro Phantom
    phantom_dir = fullfile(output_dir, phantom_name);
    if ~exist(phantom_dir, 'dir'); mkdir(phantom_dir); end

    parfile = fullfile(phantom_dir, phantom_name + ".par");

    template = readlines(par_template);
    new_par  = template;

    % Output filename (Basisname)
    output_line = sprintf('output_filename = %s/%s', phantom_dir, phantom_name);
    new_par(end+1) = output_line;

    % Height/Weight
    height = 150 + rand()*40;
    idx = contains(new_par,'height');
    if any(idx), new_par(idx) = string(sprintf('height = %.1f', height)); end

    weight = 50 + rand()*70;
    idx = contains(new_par,'weight');
    if any(idx), new_par(idx) = string(sprintf('weight = %.1f', weight)); end

    % Gender (erstmal nur Männer, weil Prostata-Anwendungsfall)
    gender_val = 0;
    idx = contains(new_par,'gender');
    if any(idx), new_par(idx) = string(sprintf('gender = %d', gender_val));
    else,        new_par(end+1) = string(sprintf('gender = %d', gender_val)); end

    % Organ file
    idx_organ_file = contains(new_par,'organ_file');
    if gender_val == 0
        organ_file_str = 'organ_file = vmale50.nrb';
    else
        organ_file_str = 'organ_file = vfemale50.nrb';
    end
    if any(idx_organ_file), new_par(idx_organ_file) = organ_file_str;
    else,                   new_par(end+1)          = organ_file_str;
    end

    % --- Energie-Default für GT/Maske: 208 keV ---
    energy_val_gt_mask = 208;
    idx = contains(new_par,'energy');
    if any(idx), new_par(idx) = string(sprintf('energy = %.1f', energy_val_gt_mask));
    else,        new_par(end+1) = string(sprintf('energy = %.1f', energy_val_gt_mask));
    end

    % Attenuation-Tabelle (auf die gefixte Datei zeigen)
    attn_table_path = fullfile(fileparts(par_template), 'atten_table_fixed.dat');
    attn_table_path = strrep(attn_table_path,'\','/');
    idx_atten = contains(new_par,'atten_table_filename');
    attn_line = sprintf('atten_table_filename= %s   # for attenuation data calculation', attn_table_path);
    if any(idx_atten), new_par(idx_atten) = attn_line;
    else,              new_par(end+1)     = attn_line;
    end

    % optionale Organ-Volumina
    organ_volumes = struct();
    organ_volumes.vol_prostate     = 15   + (60-15)*rand();
    organ_volumes.vol_salivary     = 20   + (60-20)*rand();
    organ_volumes.vol_rkidney      = 100  + (200-100)*rand();
    organ_volumes.vol_lkidney      = 100  + (200-100)*rand();
    organ_volumes.vol_liver        = 1000 + (2200-1000)*rand();
    organ_volumes.vol_spleen       = 100  + (350-100)*rand();
    organ_volumes.vol_small_intest = 400  + (1200-400)*rand();
    organ_volumes.vol_bladder      = 50   + (500-50)*rand();
    organ_fields = fieldnames(organ_volumes);
    for k = 1:length(organ_fields)
        idx = contains(new_par, organ_fields{k});
        if any(idx)
            new_par(idx) = string(sprintf('%s = %.1f', organ_fields{k}, organ_volumes.(organ_fields{k})));
        end
    end

    % Start/Endslice (achte später auf dims Z!)
    idx = contains(new_par,'startslice');
    if any(idx), new_par(idx) = "startslice = 550"; end
    idx = contains(new_par,'endslice');
    if any(idx), new_par(idx) = "endslice = 1200"; end

    % ================== DEFAULT-FLAGS für GT/Maske (beide an), Averages aus ==================
    % act_phan_each = 1, atten_phan_each = 1, act_phan_ave = 0, atten_phan_ave = 0
    idx = contains(new_par,'act_phan_each');
    if any(idx), new_par(idx) = "act_phan_each = 1";
    else,        new_par(end+1) = "act_phan_each = 1"; end

    idx = contains(new_par,'atten_phan_each');
    if any(idx), new_par(idx) = "atten_phan_each = 1";
    else,        new_par(end+1) = "atten_phan_each = 1"; end

    idx = contains(new_par,'act_phan_ave');
    if any(idx), new_par(idx) = "act_phan_ave = 0";
    else,        new_par(end+1) = "act_phan_ave = 0"; end

    idx = contains(new_par,'atten_phan_ave');
    if any(idx), new_par(idx) = "atten_phan_ave = 0";
    else,        new_par(end+1) = "atten_phan_ave = 0"; end

    % ---------------- CT/NURBS parfile (Energie = 70 keV, NUR NURBS/ATN speichern) ----------------
    ct_parfile  = fullfile(phantom_dir, phantom_name + "_ct.par");
    ct_par = new_par;

    % NURBS speichern
    nurbs_val = 1;
    idx = contains(ct_par,'nurbs_save');
    if any(idx), ct_par(idx) = string(sprintf('nurbs_save = %d',nurbs_val));
    else,        ct_par(end+1) = string(sprintf('nurbs_save = %d',nurbs_val)); end

    % Energie für CT explizit auf 70 setzen
    energy_val_ct = 70;
    idx = contains(ct_par,'energy');
    if any(idx), ct_par(idx) = string(sprintf('energy = %.1f', energy_val_ct));
    else,        ct_par(end+1) = string(sprintf('energy = %.1f', energy_val_ct)); end

    % >>> FLAGS für CT: nur ATN-Einzel-Phantom, keine ACT-Datei, keine Averages <<<
    idx = contains(ct_par,'act_phan_each');
    if any(idx), ct_par(idx) = "act_phan_each = 0";
    else,        ct_par(end+1) = "act_phan_each = 0"; end

    idx = contains(ct_par,'atten_phan_each');
    if any(idx), ct_par(idx) = "atten_phan_each = 1";
    else,        ct_par(end+1) = "atten_phan_each = 1"; end

    idx = contains(ct_par,'act_phan_ave');
    if any(idx), ct_par(idx) = "act_phan_ave = 0";
    else,        ct_par(end+1) = "act_phan_ave = 0"; end

    idx = contains(ct_par,'atten_phan_ave');
    if any(idx), ct_par(idx) = "atten_phan_ave = 0";
    else,        ct_par(end+1) = "atten_phan_ave = 0"; end

    writelines(ct_par, ct_parfile);

    % ---------------- Aktivitäten pro Gruppe (wie gehabt) ----------------
    phantom_activities = struct('Name',{},'Parameter',{},'Activity',{});
    for g = 1:size(organ_groups_params,1)
        group_name   = organ_groups_params{g,1};
        params       = organ_groups_params{g,2};
        range        = activity_ranges.(group_name);
        activity_val = range(1) + (range(2)-range(1))*rand;

        phantom_activities(g).Name      = group_name;
        phantom_activities(g).Parameter = params;
        phantom_activities(g).Activity  = activity_val;

        for p = 1:length(params)
            idx_param = contains(new_par, params{p});
            if any(idx_param)
                new_par(idx_param) = string(sprintf('%s = %.3f', params{p}, activity_val));
            end
        end
    end

    % ---------------- Masken-Parfile (IDs kommen aus *_act_1.bin) ----------------
    mask_parfile = fullfile(phantom_dir, phantom_name + "_mask.par");
    mask_par     = new_par;
    idx = contains(mask_par,'color_code');
    if any(idx), mask_par(idx) = "color_code = 1"; end

    % >>> FLAGS für MASKE: ACT ja, ATN nein, Averages aus <<<
    idx = contains(mask_par,'act_phan_each');
    if any(idx), mask_par(idx) = "act_phan_each = 1";
    else,        mask_par(end+1) = "act_phan_each = 1"; end

    idx = contains(mask_par,'atten_phan_each');
    if any(idx), mask_par(idx) = "atten_phan_each = 0";
    else,        mask_par(end+1) = "atten_phan_each = 0"; end

    idx = contains(mask_par,'act_phan_ave');
    if any(idx), mask_par(idx) = "act_phan_ave = 0";
    else,        mask_par(end+1) = "act_phan_ave = 0"; end

    idx = contains(mask_par,'atten_phan_ave');
    if any(idx), mask_par(idx) = "atten_phan_ave = 0";
    else,        mask_par(end+1) = "atten_phan_ave = 0"; end

    writelines(mask_par, mask_parfile);

    % ---------------- Ground-Truth-Parfile (beide an, Averages aus) ----------------
    gt_parfile = fullfile(phantom_dir, phantom_name + "_gt.par");
    gt_par     = new_par;
    idx = contains(gt_par,'color_code');
    if any(idx), gt_par(idx) = "color_code = 0"; end

    % (nicht zwingend nötig, da oben in new_par schon so gesetzt; hier zur Klarheit)
    idx = contains(gt_par,'act_phan_each');
    if any(idx), gt_par(idx) = "act_phan_each = 1";
    else,        gt_par(end+1) = "act_phan_each = 1"; end

    idx = contains(gt_par,'atten_phan_each');
    if any(idx), gt_par(idx) = "atten_phan_each = 1";
    else,        gt_par(end+1) = "atten_phan_each = 1"; end

    idx = contains(gt_par,'act_phan_ave');
    if any(idx), gt_par(idx) = "act_phan_ave = 0";
    else,        gt_par(end+1) = "act_phan_ave = 0"; end

    idx = contains(gt_par,'atten_phan_ave');
    if any(idx), gt_par(idx) = "atten_phan_ave = 0";
    else,        gt_par(end+1) = "atten_phan_ave = 0"; end

    writelines(gt_par, gt_parfile);

    % XCAT ausführen
    cd(fileparts(xcat_bin));
    system(sprintf('%s %s %s', xcat_bin, mask_parfile, mask_parfile));  % Masken: ACT only
    system(sprintf('%s %s %s', xcat_bin, gt_parfile,   gt_parfile));    % GT: ACT+ATN
    system(sprintf('%s %s %s', xcat_bin, ct_parfile,   ct_parfile));    % CT: ATN only (70 keV)

    % Optional: CT-Projector
    ctproj_bin = 'C:\BerndsDaten\XCAT\CT_Projector\ct_project';
    ct_input   = fullfile(phantom_dir,  phantom_name + "_ct.par_1.nrb");
    ct_output  = fullfile(phantom_dir, 'ct', phantom_name + "_CT");
    if exist(ctproj_bin, 'file')
        system(sprintf('"%s" "%s" "%s"', ctproj_bin, ct_input, ct_output));
        fprintf('CT Simulation fertig für %s\n', phantom_name);
    end

    phantom_info = struct();
    phantom_info.Name        = phantom_name;
    phantom_info.Dir         = phantom_dir;
    phantom_info.Height      = height;
    phantom_info.Weight      = weight;
    phantom_info.Gender      = gender_val;
    phantom_info.OrganFile   = organ_file_str;
    phantom_info.ParFile     = parfile;
    phantom_info.MaskParFile = mask_parfile;
    phantom_info.GTParFile   = gt_parfile;
    phantom_info.Activities  = phantom_activities;

    % (optional) kleines Manifest als .mat
    save(fullfile(phantom_dir, 'manifest.mat'), '-struct', 'phantom_info');

    fprintf('Phantom %d erstellt: %s (Gender=%d)\n', i, phantom_name, gender_val);
end


function [next_idx, next_name] = next_phantom_idx(output_dir)
% Findet den nächsten freien phantom_##-Ordnernamen in output_dir.

    if ~exist(output_dir,'dir')
        next_idx  = 1;
        next_name = sprintf('phantom_%02d', 1);
        return;
    end

    d = dir(fullfile(output_dir, 'phantom_*'));
    nums = [];
    for k = 1:numel(d)
        if d(k).isdir
            tok = regexp(d(k).name, '^phantom_(\d+)$', 'tokens', 'once');
            if ~isempty(tok)
                nums(end+1) = str2double(tok{1}); %#ok<AGROW>
            end
        end
    end

    if isempty(nums)
        next_idx = 1;
    else
        next_idx = max(nums) + 1;
    end
    next_name = sprintf('phantom_%02d', next_idx);
end
