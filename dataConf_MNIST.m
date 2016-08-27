coding = 'zeroOne';

% reweighting parameter
alpha = 1;

dsRef = @MNIST;

% ntr = [];
ntr = 1000;
nte = [];

classes = 0:9; % classes to be extracted

% Class frequencies for train and test sets

% nLow = 2;
nLow = 5;
% nLow = 10;
% nLow = 20;
% nLow = 50;
% nLow = 100;

lowFreq = 0.01;

if ~isempty(nLow)
    
    lowFreq = nLow/ntr;
end

highFreq = (1-lowFreq)/(numel(classes)-1);
trainClassFreq = [ highFreq * ones(1,9) lowFreq];

testClassFreq = [];
