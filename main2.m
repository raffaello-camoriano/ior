addpath(genpath('/home/kammo/Repos/Enitor/utils'));
addpath(genpath('/home/kammo/Repos/Enitor/dataset'));
addpath(genpath('/home/kammo/Repos/objrecpipe_mat'));
clearAllButBP;
close all;

%% Set experimental results relative directory name

saveResult = 1;

customStr = '';
dt = clock;
dt = fix(dt); 	% Get timestamp
expDir = ['Exp_' , customStr , '_' , mat2str(dt)];

resdir = ['results/' , expDir];
mkdir(resdir);

%% Save current script and loadExp in results
tmp = what;
[ST,I] = dbstack('-completenames');
copyfile([tmp.path ,'/', ST.name , '.m'],[ resdir ,'/', ST.name , '.m'])

%% Experiments setup
run_bat_rlsc_noreb = 1;
run_bat_rlsc_yesreb = 1;
run_inc_rlsc_yesreb = 1;

retrain = 1;
trainPart = 0.8;

dataRoot =  '/home/kammo/Repos/ior/data/caffe_centralcrop_meanimagenet2012/';
trainFolder = 'lunedi22';
testFolder = 'martedi23';

ntr = 165;
nte = []; 

classes = 1:2; % classes to be extracted

% Class frequencies for train and test sets
trainClassFreq = [0.5 0.5];
testClassFreq = [0.5 0.5];

% Parameter selection
numLambdas = 7;
minLambdaExp = -6;
maxLambdaExp = 0;
lrng = logspace(maxLambdaExp , minLambdaExp , numLambdas);

numrep = 10;

results.bat_rlsc_yesreb.testAccBuf = zeros(1,numrep);
results.bat_rlsc_yesreb.bestValAccBuf = zeros(1,numrep);
results.bat_rlsc_yesreb.valAcc = zeros(numrep,numLambdas);

results.bat_rlsc_noreb = results.bat_rlsc_yesreb;

results.inc_rlsc_yesreb = results.bat_rlsc_yesreb;

for k = 1:numrep
    
    clc
    display(['Repetition # ', num2str(k), ' of ' , num2str(numrep)]);
    display(' ');
	progressBar(k,numrep);
    display(' ');
    display(' ');

    %% Load data

    if ~exist('ds','class')
        ds = iCubWorld28(ntr , ntr, 'plusMinusOne' , 1, 1, 1, {classes , {}, {}, {}, {}, {}});
    else
        % Just reshuffle ds
        ds.shuffleTrainIdx();
        ds.shuffleTestIdx();
        ds.shuffleAllIdx();
    end
    
    Xtr = ds.X(ds.trainIdx,:);
    Ytr = ds.Y(ds.trainIdx,:);
    Xte = ds.X(ds.testIdx,:);
    Yte = ds.Y(ds.testIdx,:);
    
    ntr = size(Xtr,1);
    nte = size(Xte,1);
    d = size(Xtr,2);
    t  = size(Ytr,2);

    % Splitting
    ntr1 = round(ntr*trainPart);
    nval1 = round(ntr*(1-trainPart));
    tr1idx = 1:ntr1;
    val1idx = (1:nval1) + ntr1;
    Xtr1 = Xtr(tr1idx,:);
    Xval1 = Xtr(val1idx,:);
    Ytr1 = Ytr(tr1idx,:);
    Yval1 = Ytr(val1idx,:);
    
    
    %% Batch RLSC, exact rebalancing
    % Naive Linear Regularized Least Squares Classifier, 
    % with Tikhonov regularization parameter selection

    if run_bat_rlsc_yesreb == 1    

        % Compute rebalancing matrix Gamma
        Gamma = zeros(ntr1);
        if ds.t == 2
            for i = 1:ntr1
                currentClassIdx = (sign(Ytr1(i,:) - 0.5) / 2) + 1.5;
                Gamma(i,i) = 1-trainClassFreq(currentClassIdx);
            end
        else
            for i = 1:ntr1
                [ ~ , currentClass ] = find(Ytr1(i,:) == 1);
                Gamma(i,i) = 1-trainClassFreq(currentClass);
            end
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

            % Compute current accuracy
            currAcc = 1 - ds.performanceMeasure(Yval1 , Yval1pred_raw);
            
            results.bat_rlsc_yesreb.valAcc(k,lidx) = currAcc;

            if currAcc > bestAcc
                bestAcc = currAcc;
                lstar = l;
            end
        end

        %% Retrain on full training set with selected model parameters,
        %  if requested

        if retrain == 1

            % Compute rebalancing matrix Gamma
            Gamma1 = zeros(ntr);
            if ds.t == 2
                for i = 1:ntr
                    currentClassIdx = (sign(Ytr(i,:) - 0.5) / 2) + 1.5;
                    Gamma1(i,i) = 1-trainClassFreq(currentClassIdx);
                end
            else
                for i = 1:ntr
                    [ ~ , currentClass ] = find(Ytr(i,:) == 1);
                    Gamma1(i,i) = 1-trainClassFreq(currentClass);
                end
            end

            % Compute cov mat
            XtX = Xtr'*Gamma1*Xtr;
            XtY = Xtr'*Gamma1*Ytr;    

            % Train on TR
            w = (XtX + ntr*lstar*eye(d)) \ XtY;
        end

        %% Test on test set & compute accuracy

        % Predict test labels
        Ytepred_raw = Xte * w;
        
        % Compute current accuracy
        testAcc = 1 - ds.performanceMeasure(Yte , Ytepred_raw);
            
        results.bat_rlsc_yesreb.ntr = ntr;
        results.bat_rlsc_yesreb.nte = nte;
        results.bat_rlsc_yesreb.testAccBuf(k) = testAcc;
        results.bat_rlsc_yesreb.bestValAccBuf(k) = bestAcc;

    end

    
    %% Batch RLSC, no rebalancing
    % Naive Linear Regularized Least Squares Classifier, 
    % with Tikhonov regularization parameter selection

    if run_bat_rlsc_yesreb == 1    

        % Precompute cov mat
        XtX = Xtr1'*Xtr1;
        XtY = Xtr1'*Ytr1;

        lstar = 0;      % Best lambda
        bestAcc = 0;    % Highest accuracy
        for lidx = 1:numel(lrng)

            l = lrng(lidx);

            % Train on TR1
            w = (XtX + ntr1*l*eye(d)) \ XtY;

            % Predict validation labels
            Yval1pred_raw = Xval1 * w;

            % Compute current accuracy
            currAcc = 1 - ds.performanceMeasure(Yval1 , Yval1pred_raw);
            
            results.bat_rlsc_noreb.valAcc(k,lidx) = currAcc;

            if currAcc > bestAcc
                bestAcc = currAcc;
                lstar = l;
            end
        end

        %% Retrain on full training set with selected model parameters,
        %  if requested

        if retrain == 1

            % Compute cov mat
            XtX = Xtr'*Xtr;
            XtY = Xtr'*Ytr;    

            % Train on TR
            w = (XtX + ntr*lstar*eye(d)) \ XtY;
        end

        %% Test on test set & compute accuracy

        % Predict validation labels
        Ytepred_raw = Xte * w;

        % Compute current accuracy
        testAcc = 1 - ds.performanceMeasure(Yte , Ytepred_raw);
            
        results.bat_rlsc_noreb.ntr = ntr;
        results.bat_rlsc_noreb.nte = nte;
        results.bat_rlsc_noreb.testAccBuf(k) = testAcc;
        results.bat_rlsc_noreb.bestValAccBuf(k) = bestAcc;

    end
    
    
end

%% Print results

clc
display('Results');
display(' ');

if run_bat_rlsc_yesreb == 1    

    display('Batch RLSC, exact rebalancing');
    best_val_acc_avg = mean(results.bat_rlsc_yesreb.bestValAccBuf);
    best_val_acc_std = std(results.bat_rlsc_yesreb.bestValAccBuf,1);

    display(['Best validation accuracy = ', num2str(best_val_acc_avg) , ' +/- ' , num2str(best_val_acc_std)])

    test_acc_avg = mean(results.bat_rlsc_yesreb.testAccBuf);
    test_acc_std = std(results.bat_rlsc_yesreb.testAccBuf,1);

    display(['Test accuracy = ', num2str(test_acc_avg) , ' +/- ' , num2str(test_acc_std)])
    display(' ');
end

if run_bat_rlsc_noreb == 1    

    display('Batch RLSC, no rebalancing');
    best_val_acc_avg = mean(results.bat_rlsc_noreb.bestValAccBuf);
    best_val_acc_std = std(results.bat_rlsc_noreb.bestValAccBuf,1);

    display(['Best validation accuracy = ', num2str(best_val_acc_avg) , ' +/- ' , num2str(best_val_acc_std)])

    test_acc_avg = mean(results.bat_rlsc_noreb.testAccBuf);
    test_acc_std = std(results.bat_rlsc_noreb.testAccBuf,1);

    display(['Test accuracy = ', num2str(test_acc_avg) , ' +/- ' , num2str(test_acc_std)]);
    display(' ');
end


%% Save workspace

if saveResult == 1

    save([resdir '/workspace.mat']);
end

%% Plots

% Batch RLSC, exact rebalancing
figure
hold on
title({ 'Batch RLSC, exact rebalancing' ; ['Test accuracy over ' , num2str(numrep) , ' runs'] } );
boxplot(results.bat_rlsc_yesreb.testAccBuf')
hold off 

figure
hold on
title({ 'Batch RLSC, exact rebalancing' ; ['Validation accuracy over ' , num2str(numrep) , ' runs'] } );
boxplot(results.bat_rlsc_yesreb.bestValAccBuf')
hold off 


% Batch RLSC, no rebalancing
figure
hold on
title({ 'Batch RLSC, no rebalancing' ; ['Test accuracy over ' , num2str(numrep) , ' runs'] } );
boxplot(results.bat_rlsc_noreb.testAccBuf')
hold off 

figure
hold on
title({ 'Batch RLSC, no rebalancing' ; ['Validation accuracy over ' , num2str(numrep) , ' runs'] } );
boxplot(results.bat_rlsc_noreb.bestValAccBuf')
hold off 


%%  Play sound

load gong;
player = audioplayer(y, Fs);
play(player);
