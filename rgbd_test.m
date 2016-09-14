% clc;
close all;

addpath(genpath('./utils'));


% Features
features_root = '/data/giulia/ICUBWORLD_ULTIMATE/RGBD_experiments/caffenet/scores';
addpath(genpath(features_root));

% Registries
registries_root = '/data/giulia/ICUBWORLD_ULTIMATE/RGBD_registries';
addpath(genpath(registries_root));

% Training set conf
save_Xtr = 1;
save_Xval = 1;
save_Xte = 1;
ext = '.bin';
fc_dir = 'fc6';

for trial = 1:10
    
    trialName = ['trial' , num2str(trial)];
    
    
    % Name of paths for saving X, Y matrices
    XYtrain_dirpath = ['./rgbd/trial' , num2str(trial) , '/train/'];
    XYval_dirpath = ['./rgbd/trial' , num2str(trial) , '/val/'];
    XYtest_dirpath = ['./rgbd/trial' , num2str(trial) , '/test/'];
    
    % Path to registries
    XYtr_regpath = [registries_root , '/cat/' , trialName , '/train_Y.txt'];
    XYval_regpath = [registries_root , '/cat/' , trialName , '/val_Y.txt'];
    XYtest_regpath = [registries_root , '/cat/' , trialName , '/test.txt'];
        
    % Load training and validation sets
    [Xtr, Ytr, Xval, Yval] = ...
        collectORload_xy(XYtrain_dirpath, XYval_dirpath, save_Xtr, save_Xval, ext, XYtr_regpath, XYval_regpath, fc_dir);
    
    % Load test set
    [Xte, Yte] = collectORload_xy_test(XYtest_dirpath, save_Xte, ext, XYtest_regpath, fc_dir);
            
end