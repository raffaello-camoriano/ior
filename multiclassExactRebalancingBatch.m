addpath(genpath('/home/kammo/Repos/Enitor/utils'));
addpath(genpath('/home/kammo/Repos/objrecpipe_mat'));
clearAllButBP;
close all;

numrep = 20;
testAccBuf = zeros(1,numrep);
valAccBuf = zeros(1,numrep);
for k = 1:numrep

%% Load data

% generateData;

classes = 1:28; % classes to be extracted

% Class frequencies for train and test sets
trainClassFreq = 1/(27*2) .* ones(1,numel(classes));
trainClassFreq(1) = 1/2;
testClassFreq = 1/28 .* ones(1,numel(classes));


loadIcub28;

%% Batch RLSC
% Naive Linear Regularized Least Squares Classifier, 
% with Tikhonov regularization parameter selection

retrain = 1;
trainPart = 0.8;

% Parameter selection
lrng = logspace(1 , -6 , 7);

% Splitting
ntr1 = round(ntr*trainPart);
nval1 = round(ntr*(1-trainPart));
tr1idx = 1:ntr1;
val1idx = (1:nval1) + ntr1;
Xtr1 = Xtr(tr1idx,:);
Xval1 = Xtr(val1idx,:);
Ytr1 = Ytr(tr1idx,:);
Yval1 = Ytr(val1idx,:);

% Compute rebalancing matrix Gamma
Gamma = zeros(ntr1);
reweighting = (1-trainClassFreq)/sum(1-trainClassFreq);
for i = 1:ntr1
    [ ~ , currentClass ] = find(Ytr1(i,:) == 1);
    Gamma(i,i) = reweighting(currentClass);
end

% Precompute cov mat
XtX = Xtr1'*Gamma*Xtr1;
XtY = Xtr1'*Gamma*Ytr1;
lstar = 0;      % Best lambda
bestAcc = 0;    % Highest accuracy
for lidx = 1:numel(lrng)

    l = lrng(lidx);

    % Train on TR1
    w = (XtX + ntr1*l*eye(d)) \ XtY;

    % Predict validation labels
    Yval1pred_raw = Xval1 * w;

    % Encode output
    Yval1pred = -ones(nval1,t);
    for i = 1:nval1
        [~,maxIdx] = max(Yval1pred_raw(i,:));
        Yval1pred(i,maxIdx) = 1;
    end
    clear Yval1pred_raw;

    % Compute current validation reweighted accuracy
    currAcc = weightedAccuracy( Yval1', Yval1pred' , trainClassFreq);

    if currAcc > bestAcc
        bestAcc = currAcc;
        lstar = l;
    end
end

%% Retrain on full training set with selected model parameters,
%  if requested

if retrain == 1

    % Compute rebalancing matrix Gamma
    Gamma1 = zeros(ntr1);
    reweighting = (1-trainClassFreq)/sum(1-trainClassFreq);
    for i = 1:ntr
        [ ~ , currentClass ] = find(Ytr(i,:) == 1);
        Gamma1(i,i) = reweighting(currentClass);
    end

    % Compute cov mat
    XtX = Xtr'*Gamma1*Xtr;
    XtY = Xtr'*Gamma1*Ytr;    
    
    % Train on TR
    w = (XtX + ntr*lstar*eye(d)) \ XtY;
end

%% Test on test set & compute accuracy


% Predict validation labels
Ytepred_raw = Xte * w;

% Encode output
if t == 2
    Ytepred = -ones(nte,1);
    for i = 1:nte
        Ytepred(i,maxIdx) = sign(Ytepred_raw(i,:));
    end    
else
    Ytepred = -ones(nte,t);
    for i = 1:nte
        [~,maxIdx] = max(Ytepred_raw(i,:));
        Ytepred(i,maxIdx) = 1;
    end
end
clear Ytepred_raw;

% Compute test set accuracy
if t>2
    C = transpose(bsxfun(@eq, Yte', Ytepred'));
    D = sum(C,2);
    E = D == t;
    numCorrect = sum(E);
    testAcc = (numCorrect / nte);
else
    C = transpose(bsxfun(@eq, Yte', Ytepred'));
    D = sum(C,2);
    numCorrect = sum(D);
    testAcc = (numCorrect / nte);
end

testAccBuf(k) = testAcc;
valAccBuf(k) = bestAcc;

end

%% Plots

figure
hold on
title(['Test accuracy over ' , num2str(numrep) , ' runs']);
boxplot(testAccBuf')
hold off 

figure
hold on
title(['Best validation accuracy over ' , num2str(numrep) , ' runs']);
boxplot(valAccBuf')
hold off 
