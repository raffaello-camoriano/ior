%% this program computes solutions for various alphas

close all;
conf;

%% Set experimental results relative directory name

saveResult = 1;

customStr = '';
dt = clock;
dt = fix(dt); 	% Get timestamp
expDir = ['Exp_' , customStr , '_' , mat2str(dt)];

resdir = [resRoot , 'results/' , expDir];
mkdir(resdir);

%% Save current script and loadExp in results
tmp = what;
[ST,I] = dbstack('-completenames');
copyfile([tmp.path ,'/', ST.name , '.m'],[ resdir ,'/', ST.name , '.m'])

%% Experiments setup
run_bat_rlsc_noreb = 1;
run_bat_rlsc_yesreb = 0;
run_bat_rlsc_yesreb2 = 1;
run_bat_rlsc_yesrec = 0;
run_bat_rlsc_yesrec2 = 1;

retrain = 1;
trainPart = 0.8;

switch datasetName
    case 'MNIST'
        dataConf_MNIST;
    case 'iCub28'
        dataConf_iCub28;
    otherwise
        error('dataset not recognized')
end

%% Alpha setting (only for recoding)

alphaArr = 0.1:0.1:1;
numAlpha = numel(alphaArr);
resultsArr = struct();

%% Parameter selection
numLambdas = 30;
minLambdaExp = -15;
maxLambdaExp = 10;
lrng = logspace(maxLambdaExp , minLambdaExp , numLambdas);

numrep = 10;

results.bat_rlsc_yesreb.testAccBuf = zeros(1,numrep);
results.bat_rlsc_yesreb.testCM = zeros(numrep, numel(classes), numel(classes));
results.bat_rlsc_yesreb.bestValAccBuf = zeros(1,numrep);
results.bat_rlsc_yesreb.valAcc = zeros(numrep,numLambdas);
results.bat_rlsc_yesreb.teAcc = zeros(numrep,numLambdas);
results.bat_rlsc_yesreb2 = results.bat_rlsc_yesreb;
results.bat_rlsc_noreb = results.bat_rlsc_yesreb;
results.inc_rlsc_yesreb = results.bat_rlsc_yesreb;
results.inc_rlsc_yesreb2 = results.bat_rlsc_yesreb;

results.deltas = [];


for k = 1:numrep

    clc
    display(['Repetition # ', num2str(k), ' of ' , num2str(numrep)]);
    progressBar(k,numrep);
    display(' ');
    display(' ');

    %% Load data

    switch datasetName
        case 'MNIST'
            ds = dsRef(ntr , nte, coding , 0, 0, 0, {classes , trainClassFreq, testClassFreq});

        case 'iCub28'
            ds = iCubWorld28(ntr , nte, 'zeroOne' , 1, 1, 0, {classes , trainClassFreq, testClassFreq, {}, trainFolder, testFolder});

        otherwise
            error('dataset not recognized')
    end

    %       Mix up sampled points
    ds.mixUpTrainIdx;
    ds.mixUpTestIdx;

    Xtr = ds.X(ds.trainIdx,:);
    Ytr = ds.Y(ds.trainIdx,:);
    Xte = ds.X(ds.testIdx,:);
    Yte = ds.Y(ds.testIdx,:);

    ntr = size(Xtr,1);
    nte = size(Xte,1);
    d = size(Xtr,2);
    t  = size(Ytr,2);
    p = ds.trainClassNum / ntr;


    switch datasetName
        case 'MNIST'
            % Splitting MNIST
            ntr1 = round(ntr*trainPart);
            nval1 = round(ntr*(1-trainPart));
            tr1idx = 1:ntr1;
            val1idx = (1:nval1) + ntr1;
            Xtr1 = Xtr(tr1idx,:);
            Xval1 = Xtr(val1idx,:);
            Ytr1 = Ytr(tr1idx,:);
            Yval1 = Ytr(val1idx,:);
        case 'iCub28'
            % Splitting iCub28
            ntr1 = ntr;
            nval1 = round(ntr*(1-trainPart));
            tr1idx = 1:ntr;
            validx = (1:nval1);
            teidx = (1:nval1);
            Xtr1 = Xtr(tr1idx,:);
            Ytr1 = Ytr(tr1idx,:);
            Xval1 = Xte(validx,:);
            Yval1 = Yte(validx,:);
            Xte = Xte(nval1+1:end,:);
            Yte = Yte(nval1+1:end,:);
        otherwise
            error('dataset not recognized')
    end


    %% Batch RLSC, no rebalancing
    % Naive Linear Regularized Least Squares Classifier, 
    % with Tikhonov regularization parameter selection

    if run_bat_rlsc_noreb == 1    

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

            if t > 2
                Yval1pred = ds.scoresToClasses(Yval1pred_raw);
                [currAcc , ~] = weightedAccuracy2( Yval1, Yval1pred , classes);
            else
                CM = confusionmat(Yval1,sign(Yval1pred_raw));
                CM = CM ./ repmat(sum(CM,2),1,2);
                currAcc = trace(CM)/2;                
            end

            results(1).bat_rlsc_noreb.valAcc(k,lidx) = currAcc;

            if currAcc > bestAcc
                bestAcc = currAcc;
                lstar = l;
            end

            % Compute current test accuracy
            Ytepred_raw = Xte * w;

            if t > 2
                Ytepred = ds.scoresToClasses(Ytepred_raw);
                [currAcc , ~] = weightedAccuracy2( Yte, Ytepred , classes);
            else
                CM = confusionmat(Yte,sign(Ytepred_raw));
                CM = CM ./ repmat(sum(CM,2),1,2);
                currAcc = trace(CM)/2;                
            end

            results(1).bat_rlsc_noreb.teAcc(k,lidx) = currAcc;
        end

        % Retrain on full training set with selected model parameters,
        %  if requested

        if retrain == 1

            % Compute cov mat
            XtX = Xtr'*Xtr;
            XtY = Xtr'*Ytr;    

            % Train on TR
            w = (XtX + ntr*lstar*eye(d)) \ XtY;
        end

        % Test on test set & compute accuracy

        % Predict test labels
        Ytepred_raw = Xte * w;

        % Compute current accuracy

        if t > 2
            Ytepred = ds.scoresToClasses(Ytepred_raw);            
            [currAcc , CM] = weightedAccuracy2( Yte, Ytepred , classes);
        else
            CM = confusionmat(Yte,sign(Ytepred_raw));
            CM = CM ./ repmat(sum(CM,2),1,2);
            currAcc = sum(trace(CM))/2;
        end

        results(1).bat_rlsc_noreb.ntr = ntr;
        results(1).bat_rlsc_noreb.nte = nte;
        results(1).bat_rlsc_noreb.testAccBuf(k) = currAcc;
        results(1).bat_rlsc_noreb.testCM(k,:,:) = CM;
        results(1).bat_rlsc_noreb.bestValAccBuf(k) = bestAcc;

    end


    %% Batch RLSC, exact rebalancing (Gamma)
    % Naive Linear Regularized Least Squares Classifier, 
    % with Tikhonov regularization parameter selection

    if run_bat_rlsc_yesreb2 == 1    

        % Compute rebalancing matrix Gamma
        Gamma = zeros(ntr1);
        for i = 1:ntr1
            currClassIdx = find(Ytr1(i,:) == 1);
            Gamma(i,i) = computeGamma(p,currClassIdx);
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

            % Compute current validation accuracy

            if t > 2
                Yval1pred = ds.scoresToClasses(Yval1pred_raw);
                [currAcc , ~] = weightedAccuracy2( Yval1, Yval1pred , classes);
            else

                CM = confusionmat(Yval1,sign(Yval1pred_raw));
                CM = CM ./ repmat(sum(CM,2),1,2);
                currAcc = trace(CM)/2;                
            end
            results(1).bat_rlsc_yesreb2.valAcc(k,lidx) = currAcc;

            if currAcc > bestAcc
                bestAcc = currAcc;
                lstar = l;
            end

            % Compute current test accuracy
            Ytepred_raw = Xte * w;

            if t > 2
                Ytepred = ds.scoresToClasses(Ytepred_raw);
                [currAcc , ~] = weightedAccuracy2( Yte, Ytepred , classes);
            else
                CM = confusionmat(Yte,sign(Ytepred_raw));
                CM = CM ./ repmat(sum(CM,2),1,2);
                currAcc = trace(CM)/2;                
            end
            results(1).bat_rlsc_yesreb2.teAcc(k,lidx) = currAcc;
        end

        % Retrain on full training set with selected model parameters,
        %  if requested

        if retrain == 1

            % Compute rebalancing matrix Gamma
            Gamma1 = zeros(ntr);
            for i = 1:ntr
                currClassIdx = find(Ytr(i,:) == 1);
                Gamma1(i,i) = computeGamma(p,currClassIdx);
            end

            % Compute cov mat
            XtX = Xtr'*Gamma1*Xtr;
            XtY = Xtr'*Gamma1*Ytr;    

            % Train on TR
            w = (XtX + ntr*lstar*eye(d)) \ XtY;
        end

        % Test on test set & compute accuracy

        % Predict test labels
        Ytepred_raw = Xte * w;

        % Compute current accuracy

        if t > 2
            Ytepred = ds.scoresToClasses(Ytepred_raw);
            [currAcc , CM] = weightedAccuracy2( Yte, Ytepred , classes);
        else
            CM = confusionmat(Yte,sign(Ytepred_raw));
            CM = CM ./ repmat(sum(CM,2),1,2);
            currAcc = trace(CM)/2;
        end

        results(1).bat_rlsc_yesreb2.ntr = ntr;
        results(1).bat_rlsc_yesreb2.nte = nte;
        results(1).bat_rlsc_yesreb2.testAccBuf(k) = currAcc;
        results(1).bat_rlsc_yesreb2.testCM(k,:,:) = CM;
        results(1).bat_rlsc_yesreb2.bestValAccBuf(k) = bestAcc;

        % compute accuracy deltas
        results(1).deltas.bat_rlsc_yesreb2(k) = ...
            results(1).bat_rlsc_yesreb2.testAccBuf(k) - results(1).bat_rlsc_noreb.testAccBuf(k);

    end
        
        

    for kk = 1:numAlpha

        alpha = alphaArr(kk);        

        %% Batch RLSC, recoding (Gamma)
        % Naive Linear Regularized Least Squares Classifier, 
        % with Tikhonov regularization parameter selection

        if run_bat_rlsc_yesrec2 == 1

            % Compute rebalancing matrix Gamma
            Gamma = zeros(ntr1);
            for i = 1:ntr1

                currClassIdx = find(Ytr1(i,:) == 1);
                Gamma(i,i) = computeGamma(p,currClassIdx)^alpha;
            end


            % Precompute cov mat
            XtX = Xtr1'*Xtr1;
            XtY = Xtr1'*Gamma*Ytr1;

            lstar = 0;      % Best lambda
            bestAcc = 0;    % Highest accuracy
            for lidx = 1:numel(lrng)

                l = lrng(lidx);

                % Train on TR1
                w = (XtX + ntr1*l*eye(d)) \ XtY;

                % Predict validation labels
                Yval1pred_raw = Xval1 * w;

                % Compute current validation accuracy

                if t > 2
                    Yval1pred = ds.scoresToClasses(Yval1pred_raw);
                    [currAcc , ~] = weightedAccuracy2( Yval1, Yval1pred , classes);
                else

                    CM = confusionmat(Yval1,sign(Yval1pred_raw));
                    CM = CM ./ repmat(sum(CM,2),1,2);
                    currAcc = trace(CM)/2;                
                end
                results(kk).bat_rlsc_yesrec2.valAcc(k,lidx) = currAcc;

                if currAcc > bestAcc
                    bestAcc = currAcc;
                    lstar = l;
                end

                % Compute current test accuracy
                Ytepred_raw = Xte * w;

                if t > 2
                    Ytepred = ds.scoresToClasses(Ytepred_raw);
                    [currAcc , ~] = weightedAccuracy2( Yte, Ytepred , classes);
                else
                    CM = confusionmat(Yte,sign(Ytepred_raw));
                    CM = CM ./ repmat(sum(CM,2),1,2);
                    currAcc = trace(CM)/2;                
                end
                results(kk).bat_rlsc_yesrec2.teAcc(k,lidx) = currAcc;
            end

            % Retrain on full training set with selected model parameters,
            %  if requested

            if retrain == 1

                % Compute rebalancing matrix Gamma
                Gamma1 = zeros(ntr);
                for i = 1:ntr
                    currClassIdx = find(Ytr(i,:) == 1);
                    Gamma1(i,i) = computeGamma(p,currClassIdx)^alpha;
                end

                % Compute cov mat
                XtX = Xtr'*Xtr;
                XtY = Xtr'*Gamma1*Ytr;    

                % Train on TR
                w = (XtX + ntr*lstar*eye(d)) \ XtY;
            end

            % Test on test set & compute accuracy

            % Predict test labels
            Ytepred_raw = Xte * w;

            % Compute current accuracy

            if t > 2
                Ytepred = ds.scoresToClasses(Ytepred_raw);
                [currAcc , CM] = weightedAccuracy2( Yte, Ytepred , classes);
            else
                CM = confusionmat(Yte,sign(Ytepred_raw));
                CM = CM ./ repmat(sum(CM,2),1,2);
                currAcc = trace(CM)/2;
            end

            results(kk).bat_rlsc_yesrec2.ntr = ntr;
            results(kk).bat_rlsc_yesrec2.nte = nte;
            results(kk).bat_rlsc_yesrec2.testAccBuf(k) = currAcc;
            results(kk).bat_rlsc_yesrec2.testCM(k,:,:) = CM;
            results(kk).bat_rlsc_yesrec2.bestValAccBuf(k) = bestAcc;

            % compute accuracy deltas
            results(kk).deltas.bat_rlsc_yesrec2(k) = ...
                results(kk).bat_rlsc_yesrec2.testAccBuf(k) - results(1).bat_rlsc_noreb.testAccBuf(k);
        end
    end
end


%% Save workspace

if saveResult == 1

    save([resdir '/workspace.mat']  , '-v7.3');
end


%% Plots


if run_bat_rlsc_noreb == 1  
 
    % Barplot
    
    test_acc_avg = mean(results(1).bat_rlsc_noreb.testAccBuf);
    test_acc_std = std(results(1).bat_rlsc_noreb.testAccBuf,1);

    figure
    hold on
    title({ 'Batch RLSC, No rebalancing' ; ['Test accuracy over ' , num2str(numrep) , ' runs'] } );    
    bar(test_acc_avg);
    errorbar(test_acc_avg,test_acc_std);
    ylabel('Mean test accuracy');
    hold off
end

if run_bat_rlsc_yesreb2 == 1    
    
    % Barplot
    
    test_acc_avg = mean(results(1).bat_rlsc_yesreb2.testAccBuf);
    test_acc_std = std(results(1).bat_rlsc_yesreb2.testAccBuf,1);

    figure
    hold on
    title({ 'Batch RLSC, rebalancing' ; ['Test accuracy over ' , num2str(numrep) , ' runs'] } );    
    bar(test_acc_avg);
    errorbar(test_acc_avg,test_acc_std);
    ylabel('Mean test accuracy');
    hold off
end

if run_bat_rlsc_yesrec2 == 1    
    
    
    % Barplot
    
    for kk = 1:numAlpha

        test_acc_avg(kk) = mean(results(kk).bat_rlsc_yesrec2.testAccBuf);
        test_acc_std(kk) = std(results(kk).bat_rlsc_yesrec2.testAccBuf,1);

    end
    
    figure
    hold on
    title({ 'Batch RLSC, recoding' ; ['Test accuracy over ' , num2str(numrep) , ' runs'] } );    
    bar(alphaArr,test_acc_avg);
    errorbar(alphaArr,test_acc_avg,test_acc_std);
    xlabel('\alpha');
    ylabel('Mean test accuracy');
    hold off
    
end


%%  Play sound
% 
% load gong;
% player = audioplayer(y, Fs);
% play(player);
