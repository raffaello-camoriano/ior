clc;
close all;
confIncremental;

%% Experiments setup
run_bat_rlsc_yesreb = 1;    % Batch RLSC with exact loss rebalancing
run_inc_rlsc_yesrec = 1;    % Incremental RLSC with recoding

computeTestAcc = 1;     % flag for test accuracy computation

trainPart = 0.8;    % Training set part
maxiter = 1000;     % Maximum number of updates
numrep = 10;         % Number of repetitions of the experiment

saveResult = 1;

switch datasetName
    case 'MNIST'
        dataConf_MNIST_inc;
    case 'iCub28'
        dataConf_iCub28_inc;
    case 'rgbd'
        dataConf_rgbd_inc;
    otherwise
        error('dataset not recognized')
end

if strcmp(coding, 'zeroOne') ~= 1
    error('This script uses the recoding type of the form: X''*Y*C. It is only compatible with the zeroOne coding.')
end

%% Tikhonov Parameter range
numLambdas = 6;
minLambdaExp = -3;
maxLambdaExp = 2;
lrng = logspace(maxLambdaExp , minLambdaExp , numLambdas);


%% Alpha setting (only for recoding)

% alphaArr = linspace(0,1,5);
alphaArr = [0 0.7];
numAlpha = numel(alphaArr);
resultsArr = struct();
recod_alpha_idx  = 2;

%% Instantiate storage structures

results = repmat(struct(...
            'testCM' , zeros(numrep,numel(classes),numSnaps, numel(classes), numel(classes)),...
            'bestValAccBuf' , zeros(numrep,numel(classes),numSnaps),...
            'bestCMBuf' , zeros(numrep,numel(classes),numSnaps, numel(classes), numel(classes)),...
            'bestLambdaBuf' , zeros(numrep,numel(classes),numSnaps),...
            'valAcc' , zeros(numrep,numel(classes),numSnaps,numLambdas),...
            'teAcc' , zeros(numrep,numel(classes),numSnaps,numLambdas),...
            'trainTime' , zeros(numrep,numel(classes),numSnaps),...
            'testAccBuf' , zeros(numrep,numel(classes),numSnaps)...
            ), ...
            numAlpha, 1);

results_bat = struct(...
            'testCM' , zeros(numrep,numel(classes),numSnaps, numel(classes), numel(classes)),...
            'bestValAccBuf' , zeros(numrep,numel(classes),numSnaps),...
            'bestCMBuf' , zeros(numrep,numel(classes),numSnaps, numel(classes), numel(classes)),...
            'bestLambdaBuf' , zeros(numrep,numel(classes),numSnaps),...
            'valAcc' , zeros(numrep,numel(classes),numSnaps,numLambdas),...
            'teAcc' , zeros(numrep,numel(classes),numSnaps,numLambdas),...
            'trainTime' , zeros(numrep,numel(classes),numSnaps),...
            'testAccBuf' , zeros(numrep,numel(classes),numSnaps)...
            );
        
    
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
            ds = dsRef(ntr , nte, coding , 0, 0, 0, {classes , trainClassFreq, testClassFreq});

        case 'iCub28'
            ds = dsRef(ntr , nte, coding , 1, 1, 0, {classes , trainClassFreq, testClassFreq, trainClassNum, testClassNum, {}, trainFolder, testFolder});

        case 'rgbd'
            
            if numrep > 10
                error('RGB-D cannot work with more than 10 repetitions.');
            end

            trialName = ['trial' , num2str(k)];

            % Name of paths for saving X, Y matrices
            XYtrain_dirpath = [ dsDir , '/rgbd/trial' , num2str(k) , '/train/'];
            XYval_dirpath = [ dsDir , '/rgbd/trial' , num2str(k) , '/val/'];
            XYtest_dirpath = [ dsDir , '/rgbd/trial' , num2str(k) , '/test/'];

            % Path to registries
            XYtr_regpath = [registries_root , '/cat/' , trialName , '/train_Y.txt'];
            XYval_regpath = [registries_root , '/cat/' , trialName , '/val_Y.txt'];
            XYtest_regpath = [registries_root , '/cat/' , trialName , '/test.txt'];

            % Load training and validation sets
            [Xtr1, Ytr1, Xval1, Yval1] = ...
                collectORload_xy(XYtrain_dirpath, XYval_dirpath, save_Xtr, save_Xval, ext, XYtr_regpath, XYval_regpath, fc_dir);

            % Load test set
            [Xte, Yte] = collectORload_xy_test(XYtest_dirpath, save_Xte, ext, XYtest_regpath, fc_dir);
            
        otherwise
            error('dataset not recognized')
    end

    if ~strcmp(datasetName,'rgbd')
        
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
    else
        if applyPCA == 1
            
            warning('PCA not implemented for rgbd')
% 
%             X = [Xtr; Xval; Xte];
%             % Apply PCA to reduce d
%             [~, ~, X] = PCA(X, m);
%             Xtr = X(ds.trainIdx,:);
%             Xtr = X(ds.trainIdx,:);
%             Xte = X(ds.testIdx,:);
        end    
        
        % fill dataset size vars
        ntr1 = size(Xtr1,1);
        nval1 = size(Xval1,1);
        nte = size(Xte,1);
        d = size(Xtr1,2);
        t  = size(Ytr1,2);
%         p = ds.trainClassNum / ntr; % Class frequencies array

        switch coding
            case 'zeroOne'
                %%% Apply specified coding
                Ytr1 = (Ytr1 + 1) ./ 2;
                Yval1 = (Yval1 + 1) ./ 2;
                Yte = (Yte + 1) ./ 2;
            otherwise
                error('only zeroOne coding is supported');
        end

    end


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
        case 'rgbd'
            %  rgbd
            display('rgbd already split by default')

        otherwise
            error('dataset not recognized')
    end
    
    
    for imbClass = imbClassArr
        
        % Split training set in balanced (for pretraining) and imbalanced
        % (for incremental learning) subsets
        
        [tmp1,tmp2] = find(Ytr1 == 1);
        idx_bal = tmp1(tmp2 ~= imbClass);   % Compute indexes of balanced samples
        Xtr_bal = Xtr1(idx_bal , :);
        Ytr_bal = Ytr1(idx_bal , :);
        ntr_bal = size(Xtr_bal,1);
        
        idx_imbal = setdiff(1:ntr1 , idx_bal);   % Compute indexes of imbalanced samples
        Xtr_imbal = Xtr1(idx_imbal , :);
        Ytr_imbal = Ytr1(idx_imbal , :);
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
            
            sIdx = 1;
            for q = 1:ntr_imbal
                
                Xtr_tmp = [Xtr_tmp ; Xtr_imbal(q,:)];
                Ytr_tmp = [Ytr_tmp ; Ytr_imbal(q,:)];
                ntr_tmp = size(Xtr_tmp,1);
                
                
                if (sIdx <= numel(snaps)) && (q == snaps(sIdx))

                    tic 

                    % Compute p
                    % p: Relative class frequencies vector
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
                            Yval1pred = scoresToClasses( Yval1pred_raw , coding );
                            [currAcc , CM] = weightedAccuracy2( Yval1, Yval1pred , classes);
                        else

                            CM = confusionmat(Yval1,sign(Yval1pred_raw));
                            CM = CM ./ repmat(sum(CM,2),1,2);
                            currAcc = trace(CM)/2;                
                        end
                        results_bat.valAcc(k,imbClass,q,lidx) = currAcc;

                        if currAcc > bestAcc
                            bestAcc = currAcc;
                            bestCM = CM;
                            lstar = l;
                            wstar = w;
                        end
                    end

                    trainTime = trainTime + toc;
                    
                    if computeTestAcc == 1

                        % Predict validation labels
                        Ytepred_raw = Xte * wstar;

                        % Compute current validation accuracy

                        if t > 2
                            Ytepred = scoresToClasses( Ytepred_raw , coding );
                            [teAcc , CM] = weightedAccuracy2( Yte, Ytepred , classes);
                        else
                            CM = confusionmat(Yte,sign(Ytepred_raw));
                            CM = CM ./ repmat(sum(CM,2),1,2);
                            teAcc = trace(CM)/2;                
                        end        

                        results_bat.testAccBuf(k,imbClass,sIdx) = teAcc;
                        results_bat.testCM(k,imbClass,sIdx,:,:) = CM;
                    end                    

                    results_bat.ntr = ntr;
                    results_bat.nte = nte;
                    results_bat.bestValAccBuf(k,imbClass,sIdx) = bestAcc;
                    results_bat.bestCMBuf(k,imbClass,sIdx,:,:) = bestCM;
                    results_bat.trainTime(k,imbClass,sIdx) = trainTime;
  
                    sIdx = sIdx + 1;
                end
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

            sIdx = 1;
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
                wstar = zeros(d,t,numAlpha);              % Best w
                currAcc = zeros(1,numAlpha);            % Current accuracy
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

                    if (sIdx <= numel(snaps)) && (q == snaps(sIdx))

                        w0 = R_tmp{lidx} \ (R_tmp{lidx}' \ XtY_tmp);                    

                        for kk = 1:numAlpha

                            alpha = alphaArr(kk);       
                            % Training with specified alpha
                            w = w0 * (C ^ alpha);

                            % Predict validation labels
                            Yval1pred_raw = Xval1 * w;

                            % Compute current accuracy

                            if t > 2
                                Yval1pred = scoresToClasses( Yval1pred_raw , coding );
                                [currAcc(kk) , CM(:,:,kk)] = weightedAccuracy2( Yval1, Yval1pred , classes);
                            else
                                CM(:,:,kk) = confusionmat(Yval1,sign(Yval1pred_raw));
                                CM(:,:,kk) = CM(:,:,kk) ./ repmat(sum(CM(:,:,kk),2),1,2);
                                currAcc(kk) = trace(CM(:,:,kk))/2;                
                            end

                            results(kk).valAcc(k,imbClass,sIdx,lidx) = currAcc(kk);

                            if currAcc(kk) > bestAcc(kk)
                                bestAcc(kk) = currAcc(kk);
                                bestCM(:,:,kk) = CM(:,:,kk);
                                lstar(kk) = l;
                                wstar(:,:,kk) = w;
                            end

                            results(kk).ntr = ntr;
                            results(kk).nte = nte;
                            results(kk).bestValAccBuf(k,imbClass,sIdx) = bestAcc(kk);
                            results(kk).bestCMBuf(k,imbClass,sIdx,:,:) = bestCM(:,:,kk);
                            results(kk).bestLambdaBuf(k,imbClass,sIdx) = lstar(kk);     
                            
                        end
                    end
                end
                
                % Compute test accuracy
                if (computeTestAcc == 1) && (sIdx <= numel(snaps)) && (q == snaps(sIdx))
                    for kk = 1:numAlpha

                        % Predict validation labels
                        Ytepred_raw = Xte * wstar(:,:,kk);

                        % Compute current validation accuracy

                        if t > 2
                            Ytepred = scoresToClasses( Ytepred_raw , coding );
                            [teAcc , CM] = weightedAccuracy2( Yte, Ytepred , classes);
                        else
                            CM = confusionmat(Yte,sign(Ytepred_raw));
                            CM = CM ./ repmat(sum(CM,2),1,2);
                            teAcc = trace(CM)/2;
                        end        

                        results(kk).testAccBuf(k,imbClass,sIdx) = teAcc;
                        results(kk).testCM(k,imbClass,sIdx,:,:) = CM;
                    end
                end
                
                % update snapshot index
                if (sIdx < numel(snaps)) && (q == snaps(sIdx))
                    sIdx = sIdx + 1;
                end
            end
        end
    end
end



%% Plots

for c = imbClassArr

    % Validation error comparison plots
    
    if numrep == 1
        warning('Plots only for numrep > 1');
    else

        %% Overall Validation Accuracy
        
%         
        c1 = squeeze(results_bat.bestValAccBuf(:,c,:));
        c2 = squeeze(results(recod_alpha_idx).bestValAccBuf(:,c,:));
        c3 = squeeze(results(1).bestValAccBuf(:,c,:));
        
        m_bat_tot_acc = mean(c1,1);
        s_bat_tot_acc = std(c1,[],1);
        
        m_rec_tot_acc = mean(c2,1);
        s_rec_tot_acc = std(c2,[],1);
        
        m_nai_tot_acc = mean(c3,1);
        s_nai_tot_acc = std(c3,[],1);
        
        figure
        hold on
        box on
        grid on
        errorbar(snaps, ...
                m_bat_tot_acc,...
                s_bat_tot_acc);
        errorbar(snaps, ...
                m_rec_tot_acc,...
                s_rec_tot_acc);
        errorbar(snaps, ...
                m_nai_tot_acc,...
                s_nai_tot_acc);
        hold off        
        legend('Batch loss Rebalancing','Incremental Recoding', 'Incremental Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Overall Validation Accuracy')

        
        % Delta plots
        
        % Delta batch rebalancing
        d_bat_nai = c1 - c3;

        m_bat_nai_tot_del = mean(d_bat_nai,1);
        s_bat_nai_tot_del = std(d_bat_nai,[],1);
             
        % Delta incremental recoding
        d_rec_nai = c2 - c3;

        m_rec_nai_tot_del = mean(d_rec_nai,1);
        s_rec_nai_tot_del = std(d_rec_nai,[],1);
        
        figure
        hold on
        box on
        grid on
        errorbar(snaps, ...
                m_bat_nai_tot_del ,...
                s_bat_nai_tot_del );
        errorbar(snaps, ...
                m_rec_nai_tot_del,...
                s_rec_nai_tot_del);
        hold off        
        legend('Batch - Naive','Recoding - Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Overall Validation Accuracy \Delta')
        
%         
        %%% Imbalanced Validation Accuracy
        
        c1 = squeeze(results_bat.bestCMBuf(:,c,:, c, c));
        c2 = squeeze(results(recod_alpha_idx).bestCMBuf(:,c,:, c, c));
        c3 = squeeze(results(1).bestCMBuf(:,c,:, c, c));
        


        m_bat_imb_acc = mean(c1,1);
        s_bat_imb_acc = std(c1,[],1);
        
        
        m_rec_imb_acc = mean(c2,1);
        s_rec_imb_acc = std(c2,[],1);
        
            
        m_nai_imb_acc = mean(c3,1);
        s_nai_imb_acc = std(c3,[],1);
        
        figure
        hold on
        box on
        grid on
        errorbar(snaps, ...
                m_bat_imb_acc,...
                s_bat_imb_acc);
        errorbar(snaps, ...
                m_rec_imb_acc ,...
                s_rec_imb_acc);
        errorbar(snaps, ...
                m_nai_imb_acc ,...
                s_nai_imb_acc);
            
            
        hold off        
        legend('Batch loss Rebalancing','Incremental Recoding', 'Incremental Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Imbalanced Validation Accuracy')
        
        % Delta plots
        
        % Delta batch rebalancing
        d_bat_nai = c1 - c3;

        m_bat_nai_imb_del = mean(d_bat_nai,1);
        s_bat_nai_imb_del = std(d_bat_nai,[],1);
             
        % Delta incremental recoding
        d_rec_nai = c2 - c3;

        m_rec_nai_imb_del = mean(d_rec_nai,1);
        s_rec_nai_imb_del = std(d_rec_nai,[],1);
        
        figure
        hold on
        box on
        grid on
        errorbar(snaps, ...
                m_bat_nai_imb_del ,...
                s_bat_nai_imb_del );
        errorbar(snaps, ...
                m_rec_nai_imb_del,...
                s_rec_nai_imb_del);
        hold off        
        legend('Rebalancing vs. Naive','Recoding vs. Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Imb. Validation \Delta Acc.')
        set(gca,'xscale','log')
        
        
%         %%% Balanced Validation Accuracy
% 
        a1 = squeeze(results_bat.bestValAccBuf(:,c,:));
        b1 = squeeze(results_bat.bestCMBuf(:,c,:, c, c));
        c1 = (t*a1 - b1) / (t-1);

        a2 = squeeze(results(recod_alpha_idx).bestValAccBuf(:,c,:));
        b2 = squeeze(results(recod_alpha_idx).bestCMBuf(:,c,:, c, c));
        c2 = (t*a2 - b2) / (t-1);

        a3 = squeeze(results(1).bestValAccBuf(:,c,:));
        b3 = squeeze(results(1).bestCMBuf(:,c,:, c, c));
        c3 = (t*a3 - b3) / (t-1);

        m_bat_bal_acc = mean(c1,1);
        s_bat_bal_acc = std(c1,[],1); 
%         
        m_rec_bal_acc = mean(c2,1);
        s_rec_bal_acc = std(c2,[],1);  
%             
        m_nai_bal_acc = mean(c3,1);
        s_nai_bal_acc = std(c3,[],1);     
        
        
        figure
        hold on
        box on
        grid on


        
        errorbar(snaps, ...
                m_bat_bal_acc ,...
                s_bat_bal_acc );
        errorbar(snaps, ...
                m_rec_bal_acc,...
                s_rec_bal_acc);
        errorbar(snaps, ...
                m_nai_bal_acc,...
                s_nai_bal_acc);
            
            
        hold off        
        legend('Batch Loss Rebalancing','Incremental Recoding', 'Incremental Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Balanced Validation Accuracy')
        
        
        % Delta plots
        
        % Delta batch rebalancing
        d_bat_nai = c1 - c3;

        m_bat_nai_bal_del = mean(d_bat_nai,1);
        s_bat_nai_bal_del = std(d_bat_nai,[],1);
             
        % Delta incremental recoding
        d_rec_nai = c2 - c3;

        m_rec_nai_bal_del = mean(d_rec_nai,1);
        s_rec_nai_bal_del = std(d_rec_nai,[],1);
%         
        figure
        hold on
        box on
        grid on
        errorbar(snaps, ...
                m_bat_nai_bal_del ,...
                s_bat_nai_bal_del );
        errorbar(snaps, ...
                m_rec_nai_bal_del,...
                s_rec_nai_bal_del);
        hold off        
        legend('Rebalancing vs. Naive','Recoding vs. Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Balanced Validation Accuracy \Delta')
        set(gca,'xscale','log')
        
    end
    
    % Test error comparison plots
    
    if numrep == 1
        warning('Plots only for numrep > 1');
    else

        %% Overall Test Accuracy
        
        
        c1 = squeeze(results_bat.testAccBuf(:,c,:));
        c2 = squeeze(results(recod_alpha_idx).testAccBuf(:,c,:));
        c3 = squeeze(results(1).testAccBuf(:,c,:));
        
        m_bat_tot_acc_te = mean(c1,1);
        s_bat_tot_acc_te = std(c1,[],1);
        
        m_rec_tot_acc_te = mean(c2,1);
        s_rec_tot_acc_te = std(c2,[],1);
        
        m_nai_tot_acc_te = mean(c3,1);
        s_nai_tot_acc_te = std(c3,[],1);
        
        figure
        hold on
        box on
        grid on
        errorbar(snaps, ...
                m_bat_tot_acc,...
                s_bat_tot_acc);
        errorbar(snaps, ...
                m_rec_tot_acc,...
                s_rec_tot_acc);
        errorbar(snaps, ...
                m_nai_tot_acc,...
                s_nai_tot_acc);
        hold off        
        legend('Batch loss Rebalancing','Incremental Recoding', 'Incremental Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Overall Test Accuracy')

        
        % Delta plots
        
        % Delta batch rebalancing
        d_bat_nai_te = c1 - c3;

        m_bat_nai_tot_del_te = mean(d_bat_nai_te,1);
        s_bat_nai_tot_del_te = std(d_bat_nai_te,[],1);
             
        % Delta incremental recoding
        d_rec_nai_te = c2 - c3;

        m_rec_nai_tot_del_te = mean(d_rec_nai_te,1);
        s_rec_nai_tot_del_te = std(d_rec_nai_te,[],1);
        
        figure
        hold on
        box on
        grid on
        errorbar(snaps, ...
                m_bat_nai_tot_del ,...
                s_bat_nai_tot_del );
        errorbar(snaps, ...
                m_rec_nai_tot_del,...
                s_rec_nai_tot_del);
        hold off        
        legend('Batch - Naive','Recoding - Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Overall Test Accuracy \Delta')
        
        
        %%% Imbalanced Test Accuracy
        
        c1 = squeeze(results_bat.testCM(:,c,:, c, c));
        c2 = squeeze(results(recod_alpha_idx).testCM(:,c,:, c, c));
        c3 = squeeze(results(1).testCM(:,c,:, c, c));
        
        figure
        hold on
        box on
        grid on

        m_bat_imb_acc_te = mean(c1,1);
        s_bat_imb_acc_te = std(c1,[],1);
        
        errorbar(snaps, ...
                m_bat_imb_acc_te,...
                s_bat_imb_acc_te);
        

        m_rec_imb_acc_te = mean(c2,1);
        s_rec_imb_acc_te = std(c2,[],1);
        errorbar(snaps, ...
                m_rec_imb_acc_te ,...
                s_rec_imb_acc_te);
            
        m_nai_imb_acc_te = mean(c3,1);
        s_nai_imb_acc_te = std(c3,[],1);
        errorbar(snaps, ...
                m_nai_imb_acc_te ,...
                s_nai_imb_acc_te);
            
            
        hold off        
        legend('Batch loss Rebalancing','Incremental Recoding', 'Incremental Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Imbalanced Test Accuracy')
        
        % Delta plots
        
        % Delta batch rebalancing
        d_bat_nai_te = c1 - c3;

        m_bat_nai_imb_del_te = mean(d_bat_nai_te,1);
        s_bat_nai_imb_del_te = std(d_bat_nai_te,[],1);
             
        % Delta incremental recoding
        d_rec_nai_te = c2 - c3;

        m_rec_nai_imb_del_te = mean(d_rec_nai_te,1);
        s_rec_nai_imb_del_te = std(d_rec_nai_te,[],1);
        
        figure
        hold on
        box on
        grid on
        errorbar(snaps, ...
                m_bat_nai_imb_del_te ,...
                s_bat_nai_imb_del_te );
        errorbar(snaps, ...
                m_rec_nai_imb_del_te,...
                s_rec_nai_imb_del_te);
        hold off        
        legend('Rebalancing vs. Naive','Recoding vs. Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Imb. Test \Delta Acc.')
        set(gca,'xscale','log')
        
        
        %%% Balanced Test Accuracy

        a1 = squeeze(results_bat.testAccBuf(:,c,:));
        b1 = squeeze(results_bat.testCM(:,c,:, c, c));
        c1 = (t*a1 - b1) / (t-1);

        a2 = squeeze(results(recod_alpha_idx).testAccBuf(:,c,:));
        b2 = squeeze(results(recod_alpha_idx).testCM(:,c,:, c, c));
        c2 = (t*a2 - b2) / (t-1);

        a3 = squeeze(results(1).testAccBuf(:,c,:));
        b3 = squeeze(results(1).testCM(:,c,:, c, c));
        c3 = (t*a3 - b3) / (t-1);

        figure
        hold on
        box on
        grid on

        m_bat_bal_acc_te = mean(c1,1);
        s_bat_bal_acc_te = std(c1,[],1);
        
        errorbar(snaps, ...
                m_bat_bal_acc_te ,...
                s_bat_bal_acc_te );
        
        m_rec_bal_acc_te = mean(c2,1);
        s_rec_bal_acc_te = std(c2,[],1);
        errorbar(snaps, ...
                m_rec_bal_acc_te,...
                s_rec_bal_acc_te);
            
        m_nai_bal_acc_te = mean(c3,1);
        s_nai_bal_acc_te = std(c3,[],1);
        errorbar(snaps, ...
                m_nai_bal_acc_te,...
                s_nai_bal_acc_te);
            
            
        hold off        
        legend('Batch Loss Rebalancing','Incremental Recoding', 'Incremental Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Balanced Test Accuracy')
        
        
        % Delta plots
        
        % Delta batch rebalancing
        d_bat_nai_te = c1 - c3;

        m_bat_nai_bal_del_te = mean(d_bat_nai_te,1);
        s_bat_nai_bal_del_te = std(d_bat_nai_te,[],1);
             
        % Delta incremental recoding
        d_rec_nai_te = c2 - c3;

        m_rec_nai_bal_del_te = mean(d_rec_nai_te,1);
        s_rec_nai_bal_del_te = std(d_rec_nai_te,[],1);
        
        figure
        hold on
        box on
        grid on
        errorbar(snaps, ...
                m_bat_nai_bal_del_te ,...
                s_bat_nai_bal_del_te );
        errorbar(snaps, ...
                m_rec_nai_bal_del_te,...
                s_rec_nai_bal_del_te);
        hold off        
        legend('Rebalancing vs. Naive','Recoding vs. Naive', 'Location', 'southeast');
        xlabel('n_{imb}')
        ylabel('Balanced Test Accuracy \Delta')
        set(gca,'xscale','log')
        
    end    
    
end

%% Print ICRA results 

%% Total Validation Accuracy Table

display(' ');
% display('Overall Validation Accuracy Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
incremental_naive = cellstr([num2str(round(m_nai_tot_acc',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_nai_tot_acc',3))]);
incremental_recoding = cellstr([num2str(round(m_rec_tot_acc',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_tot_acc',3))]);
batch_rebalancing = cellstr([num2str(round(m_bat_tot_acc',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_tot_acc'),3)]);

% Create a table, |T|, as a container for the workspace variables. 
T_tot = table(n_imb,incremental_naive,incremental_recoding,batch_rebalancing);
T_tot.Properties.VariableNames{'n_imb'} = 'nImb';
T_tot.Properties.VariableNames{'incremental_naive'} = 'IncrementalNaive';
T_tot.Properties.VariableNames{'incremental_recoding'} = 'IncrementalRecoding';
T_tot.Properties.VariableNames{'batch_rebalancing'} = 'BatchRebalancing';

% Now use this table as input in our input struct:
input.data = T_tot;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Overall Validation Accuracy Table'];

% Now call the function to generate LaTex code:
latex_tot = latexTable(input);



%%% Balanced Validation Accuracy Table

display(' ');
% display('Balanced Validation Accuracy Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
incremental_naive = cellstr([num2str(round(m_nai_bal_acc',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_nai_bal_acc',3))]);
incremental_recoding = cellstr([num2str(round(m_rec_bal_acc',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_bal_acc',3))]);
batch_rebalancing = cellstr([num2str(round(m_bat_bal_acc',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_bal_acc',3))]);

% Create a table, |T|, as a container for the workspace variables. 
T_bal = table(n_imb,incremental_naive,incremental_recoding,batch_rebalancing);
T_bal.Properties.VariableNames{'n_imb'} = 'nImb';
T_bal.Properties.VariableNames{'incremental_naive'} = 'IncrementalNaive';
T_bal.Properties.VariableNames{'incremental_recoding'} = 'IncrementalRecoding';
T_bal.Properties.VariableNames{'batch_rebalancing'} = 'BatchRebalancing';

% Now use this table as input in our input struct:
input.data = T_bal;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Balanced Validation Accuracy Table'];

% Now call the function to generate LaTex code:
latex_bal = latexTable(input);



%%% Imbalanced Validation Accuracy Table

display(' ');
% display('Imbalanced Validation Accuracy Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
incremental_naive = cellstr([num2str(round(m_nai_imb_acc',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_nai_imb_acc',3))]);
incremental_recoding = cellstr([num2str(round(m_rec_imb_acc',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_imb_acc',3))]);
batch_rebalancing = cellstr([num2str(round(m_bat_imb_acc',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_imb_acc',3))]);

% Create a table, |T|, as a container for the workspace variables. 
T_imb = table(n_imb,incremental_naive,incremental_recoding,batch_rebalancing);
T_imb.Properties.VariableNames{'n_imb'} = 'nImb';
T_imb.Properties.VariableNames{'incremental_naive'} = 'IncrementalNaive';
T_imb.Properties.VariableNames{'incremental_recoding'} = 'IncrementalRecoding';
T_imb.Properties.VariableNames{'batch_rebalancing'} = 'BatchRebalancing';

% Now use this table as input in our input struct:
input.data = T_imb;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Imbalanced Validation Accuracy Table'];

% Now call the function to generate LaTex code:
latex_imb = latexTable(input);


% Total Test Accuracy Delta Table

display(' ');
% display('Overall Validation Accuracy Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
rebalancing_vs_naive = cellstr([num2str(round(m_bat_nai_tot_del',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_nai_tot_del',3))]);
incremental_vs_naive = cellstr([num2str(round(m_rec_nai_tot_del',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_nai_tot_del',3))]);

% Create a table, |T|, as a container for the workspace variables. 
T_tot_delta = table(n_imb,rebalancing_vs_naive,incremental_vs_naive);
T_tot_delta.Properties.VariableNames{'n_imb'} = 'nImb';
T_tot_delta.Properties.VariableNames{'rebalancing_vs_naive'} = 'RebalancingVsNaive';
T_tot_delta.Properties.VariableNames{'incremental_vs_naive'} = 'IncrementalVsNaive';

% Now use this table as input in our input struct:
input.data = T_tot_delta;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Overall Validation Accuracy Delta Table'];

% Now call the function to generate LaTex code:
latex_tot_delta = latexTable(input);



%%% Balanced Validation Accuracy Table


display(' ');
% display('Overall Validation Accuracy Delta Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
rebalancing_vs_naive = cellstr([num2str(round(m_bat_nai_bal_del',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_nai_bal_del',3))]);
incremental_vs_naive = cellstr([num2str(round(m_rec_nai_bal_del',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_nai_bal_del',3))]);

% Create a table, |T|, as a container for the workspace variables. 
T_bal_delta = table(n_imb,rebalancing_vs_naive,incremental_vs_naive);
T_bal_delta.Properties.VariableNames{'n_imb'} = 'nImb';
T_bal_delta.Properties.VariableNames{'rebalancing_vs_naive'} = 'RebalancingVsNaive';
T_bal_delta.Properties.VariableNames{'incremental_vs_naive'} = 'IncrementalVsNaive';

% Now use this table as input in our input struct:
input.data = T_bal_delta;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Balanced Validation Accuracy Delta Table'];

% Now call the function to generate LaTex code:
latex_bal_delta = latexTable(input);


%%% Imbalanced Test Accuracy Delta Table



display(' ');
% display('Imbalanced Validation Accuracy Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
rebalancing_vs_naive = cellstr([num2str(round(m_bat_nai_imb_del',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_nai_imb_del',3))]);
incremental_vs_naive = cellstr([num2str(round(m_rec_nai_imb_del',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_nai_imb_del',3))]);

% Create a table, |T|, as a container for the workspace variables. 
T_imb_delta = table(n_imb,rebalancing_vs_naive,incremental_vs_naive);
T_imb_delta.Properties.VariableNames{'n_imb'} = 'nImb';
T_imb_delta.Properties.VariableNames{'rebalancing_vs_naive'} = 'RebalancingVsNaive';
T_imb_delta.Properties.VariableNames{'incremental_vs_naive'} = 'IncrementalVsNaive';

% Now use this table as input in our input struct:
input.data = T_imb_delta;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Imbalanced Validation Accuracy Delta Table'];

% Now call the function to generate LaTex code:
latex_imb_delta = latexTable(input);



%% Total Test Accuracy Table

display(' ');
% display('Overall Test Accuracy Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
incremental_naive = cellstr([num2str(round(m_nai_tot_acc_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_nai_tot_acc_te',3))]);
incremental_recoding = cellstr([num2str(round(m_rec_tot_acc_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_tot_acc_te',3))]);
batch_rebalancing = cellstr([num2str(round(m_bat_tot_acc_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_tot_acc_te',3))]);

% Create a table, |T|, as a container for the workspace variables. 
T_tot = table(n_imb,incremental_naive,incremental_recoding,batch_rebalancing);
T_tot.Properties.VariableNames{'n_imb'} = 'nImb';
T_tot.Properties.VariableNames{'incremental_naive'} = 'IncrementalNaive';
T_tot.Properties.VariableNames{'incremental_recoding'} = 'IncrementalRecoding';
T_tot.Properties.VariableNames{'batch_rebalancing'} = 'BatchRebalancing';

% Now use this table as input in our input struct:
input.data = T_tot;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Overall Test Accuracy Table'];

% Now call the function to generate LaTex code:
latex_tot = latexTable(input);



%%% Balanced Test Accuracy Table

display(' ');
% display('Balanced Test Accuracy Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
incremental_naive = cellstr([num2str(round(m_nai_bal_acc_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_nai_bal_acc_te',3))]);
incremental_recoding = cellstr([num2str(round(m_rec_bal_acc_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_bal_acc_te',3))]);
batch_rebalancing = cellstr([num2str(round(m_bat_bal_acc_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_bal_acc_te',3))]);

% Create a table, |T|, as a container for the workspace variables. 
T_bal = table(n_imb,incremental_naive,incremental_recoding,batch_rebalancing);
T_bal.Properties.VariableNames{'n_imb'} = 'nImb';
T_bal.Properties.VariableNames{'incremental_naive'} = 'IncrementalNaive';
T_bal.Properties.VariableNames{'incremental_recoding'} = 'IncrementalRecoding';
T_bal.Properties.VariableNames{'batch_rebalancing'} = 'BatchRebalancing';

% Now use this table as input in our input struct:
input.data = T_bal;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Balanced Test Accuracy Table'];

% Now call the function to generate LaTex code:
latex_bal = latexTable(input);



%%% Imbalanced Test Accuracy Table

display(' ');
% display('Imbalanced Test Accuracy Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
incremental_naive = cellstr([num2str(round(m_nai_imb_acc_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_nai_imb_acc_te',3))]);
incremental_recoding = cellstr([num2str(round(m_rec_imb_acc_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_imb_acc_te',3))]);
batch_rebalancing = cellstr([num2str(round(m_bat_imb_acc_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_imb_acc_te',3))]);

% Create a table, |T|, as a container for the workspace variables. 
T_imb = table(n_imb,incremental_naive,incremental_recoding,batch_rebalancing);
T_imb.Properties.VariableNames{'n_imb'} = 'nImb';
T_imb.Properties.VariableNames{'incremental_naive'} = 'IncrementalNaive';
T_imb.Properties.VariableNames{'incremental_recoding'} = 'IncrementalRecoding';
T_imb.Properties.VariableNames{'batch_rebalancing'} = 'BatchRebalancing';

% Now use this table as input in our input struct:
input.data = T_imb;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Imbalanced Test Accuracy Table'];

% Now call the function to generate LaTex code:
latex_imb = latexTable(input);


%% Total Test Accuracy Delta Table

display(' ');
% display('Overall Test Accuracy Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
rebalancing_vs_naive = cellstr([num2str(round(m_bat_nai_tot_del_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_nai_tot_del_te',3))]);
incremental_vs_naive = cellstr([num2str(round(m_rec_nai_tot_del_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_nai_tot_del_te',3))]);

% Create a table, |T|, as a container for the workspace variables. 
T_tot_delta = table(n_imb,rebalancing_vs_naive,incremental_vs_naive);
T_tot_delta.Properties.VariableNames{'n_imb'} = 'nImb';
T_tot_delta.Properties.VariableNames{'rebalancing_vs_naive'} = 'RebalancingVsNaive';
T_tot_delta.Properties.VariableNames{'incremental_vs_naive'} = 'IncrementalVsNaive';

% Now use this table as input in our input struct:
input.data = T_tot_delta;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Overall Test Accuracy Delta Table'];

% Now call the function to generate LaTex code:
latex_tot_delta = latexTable(input);



%%% Balanced Test Accuracy Table


display(' ');
% display('Overall Test Accuracy Delta Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
rebalancing_vs_naive = cellstr([num2str(round(m_bat_nai_bal_del_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_nai_bal_del_te',3))]);
incremental_vs_naive = cellstr([num2str(round(m_rec_nai_bal_del_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_nai_bal_del_te',3))]);

% Create a table, |T|, as a container for the workspace variables. 
T_bal_delta = table(n_imb,rebalancing_vs_naive,incremental_vs_naive);
T_bal_delta.Properties.VariableNames{'n_imb'} = 'nImb';
T_bal_delta.Properties.VariableNames{'rebalancing_vs_naive'} = 'RebalancingVsNaive';
T_bal_delta.Properties.VariableNames{'incremental_vs_naive'} = 'IncrementalVsNaive';

% Now use this table as input in our input struct:
input.data = T_bal_delta;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Balanced Test Accuracy Delta Table'];

% Now call the function to generate LaTex code:
latex_bal_delta = latexTable(input);


%%% Imbalanced Test Accuracy Delta Table



display(' ');
% display('Imbalanced Test Accuracy Table');
display(' ');
n_imb = cellstr(num2str(snaps'));
rebalancing_vs_naive = cellstr([num2str(round(m_bat_nai_imb_del_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_bat_nai_imb_del_te',3))]);
incremental_vs_naive = cellstr([num2str(round(m_rec_nai_imb_del_te',3)) , repmat(' $\pm$ ',numSnaps,1) , num2str(round(s_rec_nai_imb_del_te',3))]);

% Create a table, |T|, as a container for the workspace variables. 
T_imb_delta = table(n_imb,rebalancing_vs_naive,incremental_vs_naive);
T_imb_delta.Properties.VariableNames{'n_imb'} = 'nImb';
T_imb_delta.Properties.VariableNames{'rebalancing_vs_naive'} = 'RebalancingVsNaive';
T_imb_delta.Properties.VariableNames{'incremental_vs_naive'} = 'IncrementalVsNaive';

% Now use this table as input in our input struct:
input.data = T_imb_delta;
input.dataFormat = {'%s'};
input.tableColumnAlignment = 'c';
input.tableBorders = 0;
input.makeCompleteLatexDocument = 0;
input.tableCaption = [ datasetName , ' Imbalanced Test Accuracy Delta Table'];

% Now call the function to generate LaTex code:
latex_imb_delta = latexTable(input);


%% Save figures

figsdir = [ resdir , '/figures/'];
mkdir(figsdir);
saveAllFigs;


%% Save workspace

if saveResult == 1

    save([resdir '/workspace.mat'] , '-v7.3');
end

% %%  Play sound
% 
% load gong;
% player = audioplayer(y, Fs);
% play(player);
