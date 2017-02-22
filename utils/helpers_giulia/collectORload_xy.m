function [Xtr, Ytr, Xval, Yval] = collectORload_xy(XYtr_dirpath, XYval_dirpath, save_Xtr, save_Xval, ext, XYtr_regpath, XYval_regpath, fc_dir)
    
% XYtr_dirpath: percorso di salvataggio training
% XYval_dirpath: percorso di salvataggio validation
% save_Xtr: flag to save training matrix
% save_Xval: flag to save validation matrix
% ext: estensione del file da salvare (biunario o mat)
% XYtr_regpath: percorso dei file di registro di training
% XYval_regpath: percorso dei file di registro di validation
% fc_dir: full path agli scores (e.g. /data/giulia/ICUBWORLD_ULTIMATE/RGBD_experiments/caffenet/scores/fc6)
 
%% TRAIN

if ~exist(XYtr_dirpath, 'dir')
    
    % collect X and Y and (optionally) save them

    fid = fopen(XYtr_regpath);
    input_registry = textscan(fid, '%s %d');
    fclose(fid);
    Ytr_idx = input_registry{2} + 1; % the saved Y is 0-indexed
    nclasses = max(Ytr_idx);
    ntr = numel(Ytr_idx);
    Ytr = -ones(ntr,nclasses);
    Ytr(sub2ind([ntr, nclasses], (1:ntr)', Ytr_idx)) = 1;
    REG = input_registry{1};
    
    fcstruct = load(fullfile(fc_dir, [REG{1}(1:(end-4)) '.mat']));
    feat_length = size(fcstruct.fc,1);
    
    Xtr = zeros(length(REG),feat_length);
    for ff=1:length(REG)
        fcstruct = load(fullfile(fc_dir, [REG{ff}(1:(end-4)) '.mat']));
        %Xtr(ff,:) = max(fcstruct.fc, [], 2);
        Xtr(ff,:) = mean(fcstruct.fc, 2);
    end
    
    if save_Xtr
        check_output_dir(XYtr_dirpath);
        % the saved Y is 0-indexed
        Ytr_idx = Ytr_idx - 1;
        if strcmp(ext, '.mat')
            save(fullfile(XYtr_dirpath, 'xy.mat'), 'Xtr', 'Ytr_idx');
        elseif strcmp(ext, '.bin')
            fid = fopen(fullfile(XYtr_dirpath, 'x.bin'),'w');
            fwrite(fid, size(Xtr), 'int32');
            fwrite(fid, Xtr, 'float');
            fclose(fid);
            fid = fopen(fullfile(XYtr_dirpath, 'y.bin'),'w');
            fwrite(fid, size(Ytr_idx), 'int32');
            fwrite(fid, Ytr_idx, 'int32');
            fclose(fid);
        end
    end
    
else
    
    % load X and Y
    warning('Using already created matrices: %s', XYtr_dirpath); 
    if strcmp(ext, '.mat')
        load(fullfile(XYtr_dirpath, 'xy.mat'));
    elseif strcmp(ext, '.bin')
        fid = fopen(fullfile(XYtr_dirpath, 'x.bin'));
        sz = fread(fid, [1, 2], 'int32');
        Xtr = fread(fid, sz, 'float');
        
        feat_length = size(Xtr,2); % used at line 88
        
        fclose(fid);
        fid = fopen(fullfile(XYtr_dirpath, 'y.bin'));
        sz = fread(fid, [1 2], 'int32');
        Ytr_idx = fread(fid, sz, 'int32');
        fclose(fid);
    end
    
    Ytr_idx = Ytr_idx + 1; % the saved Y is 0-indexed
    nclasses = max(Ytr_idx);
    n = numel(Ytr_idx);
    Ytr = -ones(n,nclasses);
    Ytr(sub2ind([n, nclasses], (1:n)', Ytr_idx)) = 1;
end

%% VAL
 
if ~exist(XYval_dirpath, 'dir')
    
    % collect X and Y and (optionally) save them

    fid = fopen(XYval_regpath);
    input_registry = textscan(fid, '%s %d');
    fclose(fid);
    Yval_idx = input_registry{2} + 1; % the saved Y is 0-indexed
    nval = numel(Yval_idx);
    Yval = -ones(nval,nclasses);
    Yval(sub2ind([nval, nclasses], (1:nval)', Yval_idx)) = 1;
    
    REG = input_registry{1};
    
    Xval = zeros(length(REG),feat_length);
    for ff=1:length(REG)
        fcstruct = load(fullfile(fc_dir, [REG{ff}(1:(end-4)) '.mat']));
        %Xval(ff,:) = max(fcstruct.fc, [], 2);
        Xval(ff,:) = mean(fcstruct.fc, 2);
    end
    
    if save_Xval
        check_output_dir(XYval_dirpath);
        % the saved Y is 0-indexed
        Yval_idx = Yval_idx - 1;
        if strcmp(ext, '.mat')
            save(fullfile(XYval_dirpath, 'xy.mat'), 'Xval', 'Yval_idx');
        elseif strcmp(ext, '.bin')
            fid = fopen(fullfile(XYval_dirpath, 'x.bin'),'w');
            fwrite(fid, size(Xval), 'int32');
            fwrite(fid, Xval, 'float');
            fclose(fid);
            fid = fopen(fullfile(XYval_dirpath, 'y.bin'),'w');
            fwrite(fid, size(Yval_idx), 'int32');
            fwrite(fid, Yval_idx, 'int32');
            fclose(fid);
        end
    end
    
else
    
    % load X and Y
    warning('Using already created matrices: %s', XYval_dirpath); 
    
    if strcmp(ext, '.mat')
        load(fullfile(XYval_dirpath, 'xy.mat'));
    elseif strcmp(ext, '.bin')
        fid = fopen(fullfile(XYval_dirpath, 'x.bin'));
        sz = fread(fid, [1 2], 'int32');
        Xval = fread(fid, sz, 'float');
        fclose(fid);
        fid = fopen(fullfile(XYval_dirpath, 'y.bin'));
        sz = fread(fid, [1 2], 'int32');
        Yval_idx = fread(fid, sz, 'int32');
        fclose(fid);
    end
    
    Yval_idx = Yval_idx + 1; % the saved Y is 0-indexed
    n = numel(Yval_idx);
    Yval = -ones(n,nclasses);
    Yval(sub2ind([n, nclasses], (1:n)', Yval_idx)) = 1;
    
end
