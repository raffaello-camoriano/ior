function [X, Y, Ntrain] = collectORload_xy_stacked(XYtr_dirpath, XYval_dirpath, save_Xtr, save_Xval, ext, XYtr_regpath, XYval_regpath, fc_dir)
       
% XYtr_dirpath: percorso di salvataggio training
% XYval_dirpath: percorso di salvataggio validation
% save_Xtr: flag to save training matrix
% save_Xval: flag to save validation matrix
% ext: estensione del file da salvare (biunario o mat)
% XYtr_regpath: percorso dei file di registro di training
% XYval_regpath: percorso dei file di registro di validation
% fc_dir: full path agli scores (e.g. /data/giulia/ICUBWORLD_ULTIMATE/RGBD_experiments/caffenet/scores/fc6)
 
%% TRAIN

if ~exist(fullfile(XYtr_dirpath, 'xy.mat'), 'file')
    
    % collect X and Y and (optionally) save them

    fid = fopen(XYtr_regpath);
    input_registry = textscan(fid, '%s %d');
    fclose(fid);  
    Y_idx = input_registry{2} + 1; % the saved Y is 0-indexed
    nclasses = max(Y_idx);
    Ntrain = numel(Y_idx);
    Y = -ones(Ntrain,nclasses);
    Y(sub2ind([Ntrain, nclasses], (1:Ntrain)', Y_idx)) = 1;
    REG = input_registry{1};
    
    fcstruct = load(fullfile(fc_dir, [REG{1}(1:(end-4)) '.mat']));
    feat_length = size(fcstruct.fc,1);
    
    X = zeros(numel(REG),feat_length);
    
    for ff=1:Ntrain
        fcstruct = load(fullfile(fc_dir, [REG{ff}(1:(end-4)) '.mat']));
        %X(ff,:) = max(fcstruct.fc, [], 2);
        X(ff,:) = mean(fcstruct.fc, 2);
    end

    if save_Xtr
        
        check_output_dir(XYtr_dirpath);
        Xtr = X;
        % the saved Y is 0-indexed
        Ytr_idx = Y_idx - 1;
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
        
        clear Xtr Ytr_idx
    end
    
else
    
    % load X and Y
    warning('Using already created matrices: %s', XYtr_dirpath); 
    
    if strcmp(ext, '.mat')
        load(fullfile(XYtr_dirpath, 'xy.mat'));
        X = Xtr;
        Y_idx = Ytr_idx;
        clear Xtr Ytr_idx
    elseif strcmp(ext, '.bin')
        fid = fopen(fullfile(XYtr_dirpath, 'x.bin'));
        sz = fread(fid, [1 2], 'int32');
        X = fread(fid, sz, 'float');
        fclose(fid);
        fid = fopen(fullfile(XYtr_dirpath, 'y.bin'));
        sz = fread(fid, [1 2], 'int32');
        Y_idx = fread(fid, sz, 'int32');
        fclose(fid);
    end
    
    Y_idx = Y_idx + 1; % the saved Y is 0-indexed
    nclasses = max(Y_idx);
    Ntrain = numel(Y_idx);
    Y = -ones(Ntrain,nclasses);
    Y(sub2ind([Ntrain, nclasses], (1:Ntrain)', Y_idx)) = 1;
    
end

%% VAL

if ~exist(fullfile(XYval_dirpath, 'xy.mat'), 'file')

    % collect X and Y and (optionally) save them
    
    fid = fopen(XYval_regpath);
    input_registry = textscan(fid, '%s %d');
    fclose(fid);
    
    Y_idx = input_registry{2} + 1; % 0-indexed
    nclasses = max(Y_idx);
    Nval = numel(Y_idx);
    Ytmp = -ones(Nval,nclasses);
    Ytmp(sub2ind([Nval, nclasses], (1:Nval)', Y_idx)) = 1;
    Y((end+1):(end+Nval),:) = Ytmp;

    REG = input_registry{1};
    
    X((end+1):(end+Nval),:) = zeros(Nval,feat_length);
    for ff=1:Nval
        fcstruct = load(fullfile(fc_dir, [REG{ff}(1:(end-4)) '.mat']));
        %X(Ntrain+ff,:) = max(fcstruct.fc, [], 2);
        X(Ntrain+ff,:) = mean(fcstruct.fc, 2);
    end
    
    if save_Xval
        
        check_output_dir(XYval_dirpath);
        Xval = X(Ntrain+1:end,:);
        % the saved Y is 0-indexed
        Yval_idx = Y_idx - 1;
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
       
        clear Xval Yval_idx
        
    end

else
    
    % load X and Y
    warning('Using already created matrices: %s', XYval_dirpath); 
    
    if strcmp(ext, '.mat')
        load(fullfile(XYval_dirpath, 'xy.mat'));
        X((end+1):(end+sz(1)),:) = Xval;
        Y_idx = Yval_idx;
        clear Xval Yval_idx
    elseif strcmp(ext, '.bin')
        fid = fopen(fullfile(XYval_dirpath, 'x.bin'));
        sz = fread(fid, [1 2], 'int32');
        X((end+1):(end+sz(1)),:) = fread(fid, sz, 'float');
        fclose(fid);
        fid = fopen(fullfile(XYval_dirpath, 'y.bin'));
        sz = fread(fid, [1 2], 'int32');
        Y_idx = fread(fid, sz, 'int32');
        fclose(fid);
    end
    
    Y_idx = Y_idx + 1; % the saved Y is 0-indexed
    nclasses = max(Y_idx);
    Nval = numel(Y_idx);
    Y((end+1):(end+Nval)) = -ones(Nval,nclasses);
    Y(Ntrain*nclasses + sub2ind([Nval, nclasses], (1:Nval)', Y_idx)) = 1;
    
end