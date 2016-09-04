coding = 'zeroOne';

% reweighting parameter
alpha = 0.5;

dsRef = @iCubWorld28;

dataRoot =  '/home/kammo/Repos/ior/data/caffe_centralcrop_meanimagenet2012/';
trainFolder = {'lunedi22','martedi23','mercoledi24','venerdi26'};
% trainFolder = 'venerdi26';
% testFolder = 'martedi23';
testFolder = {'lunedi22','martedi23','mercoledi24','venerdi26'};




classes = 1:28; % classes to be extracted
% classes = 1:4; % classes to be extracted
% classes = 1:4:28; % classes to be extracted
% classes = [1 8]; % classes to be extracted
% classes = 0:9; % classes to be extracted


% Class frequencies for train and test sets
imbClassArr = 28;   % Imbalanced class(es)

% nLow = 2;
% nLow = 5;
% nLow = 11;
% nLow = 20;
% nLow = 50;
% nLow = 100;
nLow = [];

% lowFreq = 0.01;

% if ~isempty(nLow)
%     
%     lowFreq = nLow/ntr;
% end
% 
% highFreq = (1-lowFreq)/(numel(classes)-1);

% trainClassFreq = [ highFreq * ones(1,27) lowFreq];
% trainClassFreq = 1/28 * ones(1,28);
trainClassFreq = [];
trainClassNum = 100 * ones(1,28);

% ntr = [];
% ntr = 19629;
ntr = 2800;
% ntr = 200;
nte = []; 

testClassFreq = 1/28 * ones(1,28);
testClassNum = [];

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
