coding = 'zeroOne';

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

dsDir = '/home/raffaello/Repos/Enitor/dataset';

%% Snapshot settings

snaps = [1, 2, 5, 10, 20, 50, 100, 500];   % Iterations for which batch and incremental 
                                % solutions will be computed and compared
                                % on the test set in terms of accuracy
numSnaps = numel(snaps);

%%
classes = 1:51;

% imbClassArr = 48;   % Imbalanced class(es): tomato
imbClassArr = 1:51;   % Imbalanced class(es): tomato

ntr = [];
nte = []; 

% testClassFreq = 1/28 * ones(1,28);
% testClassNum = [];

% Class frequencies for train and test sets
% trainClassFreq = [0.1 0.9];
% trainClassFreq = [ 0.1067*ones(1,9) 0.04];
% trainClassFreq = [ 200*ones(1,9) , 10] / ntr;
% trainClassFreq = [0.1658*ones(1,6) 0.005];
% trainClassFreq = [0.1658*ones(1,2) 0.005 0.1658*ones(1,4)];
% trainClassFreq = [0.1633*ones(1,2) 0.02 0.1633*ones(1,4)];
% trainClassFreq = [0.3250*ones(1,3) 0.025];
% trainClassFreq = [0.0369*ones(1,27) 0.004];
% trainClassFreq = [];
% testClassFreq = [];

%% Alpha setting (only for recoding)

% alphaArr = linspace(0,1,5);
alphaArr = [0, 0.7];
numAlpha = numel(alphaArr);
resultsArr = struct();
recod_alpha_idx  = 2;

%% Snapshot settings

snaps = 1:100;   % Iterations for which incremental 
                    % solutions will be computed and compared
                    % on the test set in terms of accuracy
