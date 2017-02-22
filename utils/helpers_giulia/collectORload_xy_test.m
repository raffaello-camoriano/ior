function [X, Y] = collectORload_xy_test(XY_dirpath, save_X, ext, XY_regpath, fc_dir)
    
% XY_dirpath: percorso di salvataggio test
% save_X: flag to save test matrix
% ext: estensione del file da salvare (biunario o mat)
% XY_regpath: percorso dei file di registro di test
% fc_dir: full path agli scores (e.g. /data/giulia/ICUBWORLD_ULTIMATE/RGBD_experiments/caffenet/scores/fc6)

if ~exist(XY_dirpath, 'dir')
    
    % collect X and Y and (optionally) save them

    fid = fopen(XY_regpath);
    input_registry = textscan(fid, '%s %d');
    fclose(fid);
    Y_idx = input_registry{2} + 1; % the saved Y is 0-indexed
    nclasses = max(Y_idx);
    n = numel(Y_idx);
    Y = -ones(n,nclasses);
    Y(sub2ind([n, nclasses], (1:n)', Y_idx)) = 1;
    REG = input_registry{1};
    
    fcstruct = load(fullfile(fc_dir, [REG{1}(1:(end-4)) '.mat']));
    feat_length = size(fcstruct.fc,1);
    
    X = zeros(length(REG),feat_length);
    for ff=1:length(REG)
        fcstruct = load(fullfile(fc_dir, [REG{ff}(1:(end-4)) '.mat']));
        %Xtr(ff,:) = max(fcstruct.fc, [], 2);
        X(ff,:) = mean(fcstruct.fc, 2);
    end
    
    if save_X
        check_output_dir(XY_dirpath);
        % the saved Y is 0-indexed
        Y_idx = Y_idx - 1;
        if strcmp(ext, '.mat')
            save(fullfile(XY_dirpath, 'xy.mat'), 'X', 'Y_idx');
        elseif strcmp(ext, '.bin')
            fid = fopen(fullfile(XY_dirpath, 'x.bin'),'w');
            fwrite(fid, size(X), 'int32');
            fwrite(fid, X, 'float');
            fclose(fid);
            fid = fopen(fullfile(XY_dirpath, 'y.bin'),'w');
            fwrite(fid, size(Y_idx), 'int32');
            fwrite(fid, Y_idx, 'int32');
            fclose(fid);
        end
        
    end
    
else
    
    % load X and Y
    warning('Using already created matrices: %s', XY_dirpath); 
    
    if strcmp(ext, '.mat')
        load(fullfile(XY_dirpath, 'xy.mat'));
    elseif strcmp(ext, '.bin')
        fid = fopen(fullfile(XY_dirpath, 'x.bin'));
        sz = fread(fid, [1 2], 'int32');
        X = fread(fid, sz, 'float');
        fclose(fid);
        fid = fopen(fullfile(XY_dirpath, 'y.bin'));
        sz = fread(fid, [1 2], 'int32');
        Y_idx = fread(fid, sz, 'int32');
        fclose(fid);
    end

    Y_idx = Y_idx + 1; % the saved Y is 0-indexed
    nclasses = max(Y_idx);
    n = numel(Y_idx);
    Y = -ones(n,nclasses);
    Y(sub2ind([n, nclasses], (1:n)', Y_idx)) = 1;
    
end
