clc;
close all;
confIncremental;


%% Experiments setup
run_bat_rlsc_yesreb = 0;    % Batch RLSC with exact loss rebalancing
run_inc_rlsc_norec = 0;     % Naive incremental RLSC with no recoding
run_inc_rlsc_yesrec = 1;    % Incremental RLSC with recoding

trainPart = 0.8;    % Training set part
maxiter = 100;      % Maximum number of updates
numrep = 5;        % Number of repetitions of the experiment

saveResult = 1;

switch datasetName
    case 'MNIST'
        dataConf_MNIST_inc;
    case 'iCub28'
        dataConf_iCub28_inc;
    otherwise
        error('dataset not recognized')
end

if strcmp(coding, 'zeroOne') ~= 1
    error('This script uses the recoding type of the form: X''*Y*C. It is only compatible with the zeroOne coding.')
end

%% Tikhonov Parameter range
numLambdas = 30;
minLambdaExp = -10;
maxLambdaExp = 5;
lrng = logspace(maxLambdaExp , minLambdaExp , numLambdas);


%% Alpha setting (only for recoding)

% alphaArr = 0.1:0.1:1;
alphaArr = 0:0.1:1;
numAlpha = numel(alphaArr);
resultsArr = struct();


%% Instantiate storage structures
results.bat_rlsc_yesreb.testCM = zeros(numrep,numel(classes),1, numel(classes), numel(classes));
results.bat_rlsc_yesreb.bestValAccBuf = zeros(numrep,numel(classes),1);
results.bat_rlsc_yesreb.bestCMBuf = zeros(numrep,numel(classes),1, numel(classes), numel(classes));
results.bat_rlsc_yesreb.bestLambdaBuf = zeros(numrep,numel(classes),1);
results.bat_rlsc_yesreb.valAcc = zeros(numrep,numel(classes),1,numLambdas);
results.bat_rlsc_yesreb.teAcc = zeros(numrep,numel(classes),1,numLambdas);
results.bat_rlsc_yesreb.trainTime = zeros(numrep,numel(classes),1);
results.bat_rlsc_yesreb.testAccBuf = zeros(numrep,numel(classes),1);

results.inc_rlsc_norec = results.bat_rlsc_yesreb;

results.inc_rlsc_yesrec = results.bat_rlsc_yesreb;


for k = 1:numrep
    
    clc
    display(['Repetition # ', num2str(k), ' of ' , num2str(numrep)]);
    display(' ');
	progressBar(k,numrep);
    display(' ');
    display(' ');

    %% Load data

    switch datasetName
        case 'MNIST'
            ds = MNIST(ntr , nte, coding , 0, 0, 0, {classes , trainClassFreq, testClassFreq});

        case 'iCub28'
            ds = iCubWorld28(ntr , nte, coding , 1, 1, 0, {classes , trainClassFreq, testClassFreq, trainClassNum, testClassNum, {}, trainFolder, testFolder});

        otherwise
            error('dataset not recognized')
    end

    %       Mix up sampled points
    ds.mixUpTrainIdx;
    ds.mixUpTestIdx;
    
    if applyPCA == 1

        % Apply PCA to reduce d
        [~, ~, X] = PCA(ds.X, m);
        Xtr = X(ds.trainIdx,:);
        Xte = X(ds.testIdx,:);
    else
        Xtr = ds.X(ds.trainIdx,:);
        Xte = ds.X(ds.testIdx,:);
    end
    
    Ytr = ds.Y(ds.trainIdx,:);
    Yte = ds.Y(ds.testIdx,:);

    
    ntr = size(Xtr,1);
    nte = size(Xte,1);
    d = size(Xtr,2);
    t  = size(Ytr,2);
    p = ds.trainClassNum / ntr; % Class frequencies array


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
    
    
    for imbClass = imbClassArr
        
        % Split training set in balanced (for pretraining) and imbalanced
        % (for incremental learning) subsets
        
        [tmp1,tmp2] = find(Ytr == 1);
        idx_bal = tmp1(tmp2 ~= imbClass);   % Compute indexes of balanced samples
        Xtr_bal = Xtr(idx_bal , :);
        Ytr_bal = Ytr(idx_bal , :);
        ntr_bal = size(Xtr_bal,1);
        
        idx_imbal = setdiff(1:ntr , idx_bal);   % Compute indexes of imbalanced samples
        Xtr_imbal = Xtr(idx_imbal , :);
        Ytr_imbal = Ytr(idx_imbal , :);
        ntr_imbal = min([maxiter, numel(idx_imbal)]);
        
        
        % Pre-train batch model on points not belonging to imbalanced class
        XtX = Xtr_bal'*Xtr_bal;
        XtY = Xtr_bal'*Ytr_bal;

        lstar = 0;      % Best lambda
        bestAcc = 0;    % Highest accuracy
        w = cell(1,numel(lrng));
        R = cell(1,numel(lrng));
        
        for lidx = 1:numel(lrng)

            l = lrng(lidx);
            R{lidx} = chol(XtX + ntr_bal * l * eye(d), 'upper');  
        end
        
    
        %% Batch RLSC, exact rebalancing
        % Naive Linear Regularized Least Squares Classifier, 
        % with Tikhonov regularization parameter selection

        if run_bat_rlsc_yesreb == 1
            
            Xtr_tmp = Xtr_bal;
            Ytr_tmp = Ytr_bal;
            
            trainTime = 0;
            
            for q = 1:ntr_imbal
                
                Xtr_tmp = [Xtr_tmp ; Xtr_imbal(q,:)];
                Ytr_tmp = [Ytr_tmp ; Ytr_imbal(q,:)];
                ntr_tmp = size(Xtr_tmp,1);
                
                tic 
                % Compute p
                [~,tmp] = find(Ytr_tmp == 1);
                a = unique(tmp);
                out = [a,histc(tmp(:),a)];
                p = out(:,2)'/ntr_tmp;
                
                % Compute rebalancing matrix Gamma
                Gamma = zeros(ntr_tmp);
                for i = 1:ntr_tmp

                    currClassIdx = find(Ytr_tmp(i,:) == 1);
                    Gamma(i,i) = computeGamma(p,currClassIdx);
                end
                XtX = Xtr_tmp'*Gamma*Xtr_tmp;
                XtY = Xtr_tmp'*Gamma*Ytr_tmp;

                lstar = 0;      % Best lambda
                bestAcc = 0;    % Highest accuracy
                for lidx = 1:numel(lrng)

                    l = lrng(lidx);

                    % Train on TR1
                    w = (XtX + ntr_tmp*l*eye(d)) \ XtY;

                    % Predict validation labels
                    Yval1pred_raw = Xval1 * w;

                    % Compute current validation accuracy

                    if t > 2
                        Yval1pred = ds.scoresToClasses(Yval1pred_raw);
                        [currAcc , CM] = weightedAccuracy2( Yval1, Yval1pred , classes);
                    else

                        CM = confusionmat(Yval1,sign(Yval1pred_raw));
                        CM = CM ./ repmat(sum(CM,2),1,2);
                        currAcc = trace(CM)/2;                
                    end
                    results(1).bat_rlsc_yesreb.valAcc(k,imbClass,q,lidx) = currAcc;

                    if currAcc > bestAcc
                        bestAcc = currAcc;
                        bestCM = CM;
                        lstar = l;
                    end

                    % Compute current test accuracy
%                     Ytepred_raw = Xte * w;
% 
%                     if t > 2
%                         Ytepred = ds.scoresToClasses(Ytepred_raw);
%                         [currAcc , CM] = weightedAccuracy2( Yte, Ytepred , classes);
%                     else
%                         CM = confusionmat(Yte,sign(Ytepred_raw));
%                         CM = CM ./ repmat(sum(CM,2),1,2);
%                         currAcc = trace(CM)/2;                
%                     end
%                     results.bat_rlsc_yesreb.teAcc(k,j,q,lidx) = currAcc;
                end
                
                trainTime = trainTime + toc;

                results(1).bat_rlsc_yesreb.ntr = ntr;
                results(1).bat_rlsc_yesreb.nte = nte;
                results(1).bat_rlsc_yesreb.testAccBuf(k,imbClass,q) = currAcc;
                results(1).bat_rlsc_yesreb.testCM(k,imbClass,q,:,:) = CM;
                results(1).bat_rlsc_yesreb.bestValAccBuf(k,imbClass,q) = bestAcc;
                results(1).bat_rlsc_yesreb.bestCMBuf(k,imbClass,q,:,:) = bestCM;
                results(1).bat_rlsc_yesreb.trainTime(k,imbClass,q) = trainTime;
            end
        end



        %% Incremental RLSC, no recoding
        % Naive Linear Regularized Least Squares Classifier, 
        % with Tikhonov regularization parameter selection

        if run_inc_rlsc_norec == 1    
            
            % Helper variables
            R_tmp = cell(1,numLambdas);
            trainTime = 0;
            
            % Cycle over the imbalanced class
            for q = 1:ntr_imbal

                if q == 1
                    XtY_tmp = XtY;   
                    ntr_tmp = ntr_bal;
                end

                tic

                % Update XtY
                XtY_tmp = XtY_tmp + Xtr_imbal(q,:)' * Ytr_imbal(q,:);
                
                ntr_tmp = ntr_tmp + 1; % Update current number of training points
                

                lstar = 0;      % Best lambda
                bestAcc = 0;    % Highest accuracy
                for lidx = 1:numel(lrng)

                    l = lrng(lidx);

                    if q == 1
                        R_tmp{lidx} = R{lidx};  % Compute first Cholesky factorization of XtX + n * lambda * I
                    end
                    % Update Cholesky factor
                    R_tmp{lidx} = cholupdatek(R_tmp{lidx}, Xtr_imbal(q,:)' , '+');                
                    
                    % Training
                    w = R_tmp{lidx} \ (R_tmp{lidx}' \ XtY_tmp );                    

                    % Predict validation labels
                    Yval1pred_raw = Xval1 * w;

                    % Compute current accuracy

                    if t > 2
                        Yval1pred = ds.scoresToClasses(Yval1pred_raw);
                        [currAcc , CM] = weightedAccuracy2( Yval1, Yval1pred , classes);
                    else
                        CM = confusionmat(Yval1,sign(Yval1pred_raw));
                        CM = CM ./ repmat(sum(CM,2),1,2);
                        currAcc = trace(CM)/2;                
                    end

                    results(1).inc_rlsc_norec.valAcc(k,imbClass,q,lidx) = currAcc;

                    if currAcc > bestAcc
                        bestAcc = currAcc;
                        bestCM = CM;
                        lstar = l;
                    end

%                     % Compute current test accuracy
%                     Ytepred_raw = Xte * w;
% 
%                     if t > 2
%                         Ytepred = ds.scoresToClasses(Ytepred_raw);
%                         [currAcc , CM] = weightedAccuracy2( Yte, Ytepred , classes);
%                     else
%                         CM = confusionmat(Yte,sign(Ytepred_raw));
%                         CM = CM ./ repmat(sum(CM,2),1,2);
%                         currAcc = trace(CM)/2;                
%                     end
% 
%                     results.inc_rlsc_norec.teAcc(k,j,q,lidx) = currAcc;
                end
                
                
                trainTime = trainTime + toc;
                

                results(1).inc_rlsc_norec.ntr = ntr;
                results(1).inc_rlsc_norec.nte = nte;
                results(1).inc_rlsc_norec.testAccBuf(k,imbClass,q) = currAcc;
                results(1).inc_rlsc_norec.testCM(k,imbClass,q,:,:) = CM;
                results(1).inc_rlsc_norec.bestValAccBuf(k,imbClass,q) = bestAcc;
                results(1).inc_rlsc_norec.bestCMBuf(k,imbClass,q,:,:) = bestCM;
                results(1).inc_rlsc_norec.bestLambdaBuf(k,imbClass,q) = lstar;
                results(1).inc_rlsc_norec.trainTime(k,imbClass,q) = trainTime;
         
            end
        end

        
 
        

        %% Incremental RLSC, recoding
        % Naive Linear Regularized Least Squares Classifier, 
        % with Tikhonov regularization parameter selection

        if run_inc_rlsc_yesrec == 1    

            %Init
            Xtr_tmp = zeros(size(Xtr_bal,1)+size(Xtr_imbal,1),d);
            Ytr_tmp = zeros(size(Ytr_bal,1)+size(Ytr_imbal,1),t);

            Xtr_tmp(1:ntr_bal,:) = Xtr_bal;
            Ytr_tmp(1:ntr_bal,:) = Ytr_bal;

            R_tmp = cell(1,numLambdas);
            trainTime = 0;
            ntr_tmp = size(Xtr_bal,1);

            % cycle over the imbalanced class' samples
            for q = 1:ntr_imbal

                ntr_tmp = ntr_tmp + 1;
                Xtr_tmp(ntr_tmp,:) = Xtr_imbal(q,:);
                Ytr_tmp(ntr_tmp,:) = Ytr_imbal(q,:);

                tic

                % Compute p
                % p: Relative class frequencies vector
                [~,tmp] = find(Ytr_tmp == 1);
                a = unique(tmp);
                out = [a,histc(tmp(:),a)];
                p = out(:,2)'/ntr_tmp;

                % Compute t x t recoding matrix C
                C = zeros(t);
                for i = 1:t
                    currClassIdx = i;
                    C(i,i) = computeGamma(p,currClassIdx);
                end

                % Compute b
                XtY_tmp = Xtr_tmp(1:ntr_tmp,:)' * Ytr_tmp(1:ntr_tmp,:);

                % Buffer variables
                lstar = zeros(1,numAlpha);              % Best lambda
                currAcc = zeros(1,numAlpha);            % Best lambda
                bestAcc = zeros(1,numAlpha);       % Highest accuracy
                CM = zeros(t,t,numAlpha);
                bestCM = zeros(t,t,numAlpha);
                
                for lidx = 1:numel(lrng)

                    l = lrng(lidx);

                    if q == 1
                        R_tmp{lidx} = R{lidx};  % Compute first Cholesky factorization of XtX + n * lambda * I
                    end
                    % Update Cholesky factor
                    R_tmp{lidx} = cholupdatek(R_tmp{lidx}, Xtr_imbal(q,:)' , '+');                

                    w0 = R_tmp{lidx} \ (R_tmp{lidx}' \ XtY_tmp);                    

                    for kk = 1:numAlpha

                        alpha = alphaArr(kk);       
                        % Training with specified alpha
                        w = w0 * (C ^ alpha);

                        % Predict validation labels
                        Yval1pred_raw = Xval1 * w;

                        % Compute current accuracy

                        if t > 2
                            Yval1pred = ds.scoresToClasses(Yval1pred_raw);
                            [currAcc(kk) , CM(:,:,kk)] = weightedAccuracy2( Yval1, Yval1pred , classes);
                        else
                            CM(:,:,kk) = confusionmat(Yval1,sign(Yval1pred_raw));
                            CM(:,:,kk) = CM(:,:,kk) ./ repmat(sum(CM(:,:,kk),2),1,2);
                            currAcc(kk) = trace(CM(:,:,kk))/2;                
                        end

                        results(kk).inc_rlsc_yesrec.valAcc(k,imbClass,q,lidx) = currAcc(kk);

                        if currAcc(kk) > bestAcc(kk)
                            bestAcc(kk) = currAcc(kk);
                            bestCM(:,:,kk) = CM(:,:,kk);
                            lstar(kk) = l;
                        end

                        % Compute current test accuracy
    %                     Ytepred_raw = Xte * w;
    % 
    %                     if t > 2
    %                         Ytepred = ds.scoresToClasses(Ytepred_raw);
    %                         [currAcc , ~] = weightedAccuracy2( Yte, Ytepred , classes);
    %                     else
    %                         CM = confusionmat(Yte,sign(Ytepred_raw));
    %                         CM = CM ./ repmat(sum(CM,2),1,2);
    %                         currAcc = trace(CM)/2;                
    %                     end
    % 
    %                     results.inc_rlsc_yesrec.teAcc(k,j,q,lidx) = currAcc;

%                         trainTime = trainTime + toc;

                    % Test on test set & compute accuracy

%                     % Predict test labels
%                     Ytepred_raw = Xte * w;
% 
%                     % Compute current accuracy
% 
%                     if t > 2
%                         Ytepred = ds.scoresToClasses(Ytepred_raw);
%                         [currAcc , CM] = weightedAccuracy2( Yte, Ytepred , classes);
%                     else
%                         CM = confusionmat(Yte,sign(Ytepred_raw));
%                         CM = CM ./ repmat(sum(CM,2),1,2);
%                         currAcc = trace(CM)/2;
%                     end

                        results(kk).inc_rlsc_yesrec.ntr = ntr;
                        results(kk).inc_rlsc_yesrec.nte = nte;
                        results(kk).inc_rlsc_yesrec.testAccBuf(k,imbClass,q) = currAcc(kk);
                        results(kk).inc_rlsc_yesrec.testCM(k,imbClass,q,:,:) = CM(:,:,kk);
                        results(kk).inc_rlsc_yesrec.bestValAccBuf(k,imbClass,q) = bestAcc(kk);
                        results(kk).inc_rlsc_yesrec.bestCMBuf(k,imbClass,q,:,:) = bestCM(:,:,kk);
                        results(kk).inc_rlsc_yesrec.bestLambdaBuf(k,imbClass,q) = lstar(kk);                
%                         results(kk).inc_rlsc_yesrec.trainTime(k,imbClass,q) = trainTime;
                 
%                     % compute accuracy deltas
%                     results(kk).deltas.inc_rlsc_yesrec(k) = ...
%                         results(kk).inc_rlsc_yesrec.testAccBuf(k) - results(1).inc_rlsc_yesrec.testAccBuf(k);

                    end
                end
            end
        end
    end
end


%% Save workspace

if saveResult == 1

    save([resdir '/workspace.mat'] , '-v7.3');
end

%% Plots

for c = imbClassArr

    % Test error comparison plots
    
    if numrep == 1

        figure
        hold on
        plot(squeeze(mean(results.inc_rlsc_norec.bestValAccBuf(:,c,1:min(maxiter,out(c,2))),1)))
        plot(squeeze(mean(results.inc_rlsc_yesrec.bestValAccBuf(:,c,1:min(maxiter,out(c,2))),1)))
        title(['Test Error for imbalanced class # ' , num2str(c)]);
        legend('Naive RRLSC','Recoded RRLSC')
        xlabel('n_{imb}')
        ylabel('Test Accuracy')
        hold off    

        figure
        hold on
        plot(squeeze(mean(results.inc_rlsc_norec.trainTime(:,c,1:min(maxiter,out(c,2))),1)))
        plot(squeeze(mean(results.inc_rlsc_yesrec.trainTime(:,c,1:min(maxiter,out(c,2))),1)))
        title(['Training Time for imbalanced class # ' , num2str(c)]);
        legend('Naive RRLSC','Recoded RRLSC')
        xlabel('n_{imb}')
        ylabel('Training Time')
        hold off  

    else
        
%         for kk = 1: numAlpha
%             figure
%             hold on
%             h1 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(1).inc_rlsc_norec.bestValAccBuf(:,c,1:min(maxiter,out(c,2)))), ...
%                 'r' , 0.1 , 0 , 1 , '-');
%             h2 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(kk).inc_rlsc_yesrec.bestValAccBuf(:,c,1:min(maxiter,out(c,2)))), ...
%                 'b' , 0.1 , 0 , 1 , '-');
%             title({['Test Error for imbalanced class # ' , num2str(c)] ; ['\alpha = ', num2str(alphaArr(kk))]});
%             legend([h1,h2],'Naive RRLSC','Recoded RRLSC','Location','southeast')
%             xlabel('n_{imb}')
%             ylabel('Test Accuracy')
%             hold off    
%         end
        
        figure
        hold on
        for kk = 2: numAlpha
            subplot(2,5,kk-1)
            h1 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(1).inc_rlsc_yesrec.bestValAccBuf(:,c,1:min(maxiter,out(c,2)))), ...
                'r' , 0.1 , 0 , 1 , '-');
            h2 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(kk).inc_rlsc_yesrec.bestValAccBuf(:,c,1:min(maxiter,out(c,2)))), ...
                'b' , 0.1 , 0 , 1 , '-');
            title({'Test Acc.' ; ['\alpha = ', num2str(alphaArr(kk))]});
            legend([h1,h2],'Naive RRLSC','Recoded RRLSC','Location','southeast')
            xlabel('n_{imb}')
            ylabel('Test Accuracy')
            hold off    
        end
        
        figure
        hold on
        for kk = 2: numAlpha
            subplot(2,5,kk-1)
            h1 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(1).inc_rlsc_yesrec.bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c)), ...
                'r' , 0.1 , 0 , 1 , '-');
            h2 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(kk).inc_rlsc_yesrec.bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c)), ...
                'b' , 0.1 , 0 , 1 , '-');
            title({['Test Acc.; Imb. C' , num2str(c)] ; ['\alpha = ', num2str(alphaArr(kk))]});
            legend([h1,h2],'Naive RRLSC','Recoded RRLSC','Location','southeast')
            xlabel('n_{imb}')
            ylabel('Test Accuracy')
            hold off    
        end
        
        figure
        hold on
        for kk = 2: numAlpha
            
            a1 = squeeze(results(1).inc_rlsc_yesrec.bestValAccBuf(:,c,1:min(maxiter,out(c,2))));
            b1 = squeeze(results(1).inc_rlsc_yesrec.bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c));
            c1 = (t*a1 - b1) / (t-1);
            
            a2 = squeeze(results(kk).inc_rlsc_yesrec.bestValAccBuf(:,c,1:min(maxiter,out(c,2))));
            b2 = squeeze(results(kk).inc_rlsc_yesrec.bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c));
            c2 = (t*a2 - b2) / (t-1);
            
            subplot(2,5,kk-1)
            h1 = bandplot(1:min(maxiter,out(c,2)),c1, ...
                'r' , 0.1 , 0 , 1 , '-');
            h2 = bandplot(1:min(maxiter,out(c,2)),c2, ...
                'b' , 0.1 , 0 , 1 , '-');
            title({['Test Acc.; Bal. !C' , num2str(c)] ; ['\alpha = ', num2str(alphaArr(kk))]});
            legend([h1,h2],'Naive RRLSC','Recoded RRLSC','Location','southeast')
            xlabel('n_{imb}')
            ylabel('Test Accuracy')
            hold off    
        end
        


%         figure
%         hold on
%         h1 = bandplot(1:min(maxiter,out(c,2)),squeeze(results.inc_rlsc_norec.trainTime(:,c,1:min(maxiter,out(c,2)))), ...
%             'r' , 0.1 , 0 , 1 , '-');
%         h2 = bandplot(1:min(maxiter,out(c,2)),squeeze(results.inc_rlsc_yesrec.trainTime(:,c,1:min(maxiter,out(c,2)))), ...
%             'b' , 0.1 , 0 , 1 , '-');
%         title(['Training Time for imbalanced class # ' , num2str(c)]);
%         legend([h1,h2],'Naive RRLSC','Recoded RRLSC','Location','southeast')
%         xlabel('n_{imb}')
%         ylabel('Training Time')
%         hold off    
        
    end    
    
    
%     % Best lambda plots
%     
%     if numrep == 1
% 
% 
%         figure
%         hold on
%         title(['Lambda* for imbalanced class # ' , num2str(c)]);
%         scatter(1:ntr_imbal , squeeze(results(1).inc_rlsc_norec.bestLambdaBuf(:,c,:) , 1));
%         xlabel('Iteration')
%         ylabel('\lambda^*')
%         set(gca,'YScale','log');
%         hold off 
%         
%     else
% 
%         figure
%         hold on
%         title(['Lambda* for imbalanced class # ' , num2str(c)]);
%         scatter(1:ntr_imbal , squeeze(median(results(1).inc_rlsc_norec.bestLambdaBuf(:,c,:) , 1)));
%         xlabel('Iteration')
%         ylabel('\lambda^*')
%         set(gca,'YScale','log');
%         hold off 
%         
%         for kk = 1: numAlpha
%             
%             figure
%             hold on
%             title({['Lambda* for imbalanced class # ' , num2str(c)] ; ['\alpha = ', num2str(alphaArr(kk))]});
%             scatter(1:ntr_imbal , squeeze(median(results(kk).inc_rlsc_yesrec.bestLambdaBuf(:,c,:) , 1)));
%             xlabel('Iteration')
%             ylabel('\lambda^*')
%             set(gca,'YScale','log');
%             hold off
%             
%             figure
%             hold on
%             title({'Regularization path' ; ['\alpha = ', num2str(alphaArr(kk))]});
%             surf(lrng , 1:ntr_imbal, squeeze(results(kk).inc_rlsc_yesrec.valAcc(1,c,:,:)))
%             xlabel('\lambda');
%             ylabel('# update');
%             zlabel('Validation accuracy');
%             set(gca,'XScale','log');
%             hold off
%         end        
%         
%     end        
    
end

%% Save figures

figsdir = [ resdir , '/figures/'];
mkdir(figsdir);
saveAllFigs;

% %%  Play sound
% 
% load gong;
% player = audioplayer(y, Fs);
% play(player);
