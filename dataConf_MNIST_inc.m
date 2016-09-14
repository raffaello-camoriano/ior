coding = 'zeroOne';

% reweighting parameter
% alpha = 1;
% alpha = 0.5;
% alpha = 0;

dsRef = @MNIST;

% ntr = [];
ntr = 10000;
nte = [];

%% Snapshot settings

% snaps = [1, 2, 5, 10, 20, 50, 100, 500];   % Iterations for which batch and incremental 
snaps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];   % Iterations for which batch and incremental 
                                % solutions will be computed and compared
                                % on the test set in terms of accuracy
numSnaps = numel(snaps);

%%

classes = 0:9; % classes to be extracted


% Class frequencies for train and test sets
imbClassArr = 9;   % Imbalanced class(es)

% Class frequencies for train and test sets

% nLow = 2;
% nLow = 5;
% nLow = 10;
% nLow = 20;
% nLow = 50;
nLow = 1000;

lowFreq = 0.01;

if ~isempty(nLow)
    
    lowFreq = nLow/ntr;
end

highFreq = (1-lowFreq)/(numel(classes)-1);
trainClassFreq = [ highFreq * ones(1,9) lowFreq];

testClassFreq = [];
