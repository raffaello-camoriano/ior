close all;
conf;

%% Set experimental results relative directory name

saveResult = 1;

customStr = '';
dt = clock;
dt = fix(dt); 	% Get timestamp
expDir = ['Exp_' , customStr , '_' , mat2str(dt)];

resdir = [resRoot , 'results/incremental_exp/' , expDir];
mkdir(resdir);

%% Save current script and loadExp in results
tmp = what;
[ST,I] = dbstack('-completenames');
copyfile([tmp.path ,'/', ST.name , '.m'],[ resdir ,'/', ST.name , '.m'])

%% Experiments setup
run_bat_rlsc_yesreb = 0;    % Batch RLSC with exact rebalancing
run_inc_rlsc_norec = 1;     % Naive incremental RLSC with no recoding
run_inc_rlsc_yesrec = 1;    % Incremental RLSC with recoding

trainPart = 0.8;


switch datasetName
    case 'MNIST'
        dataConf_MNIST;
    case 'iCub28'
        dataConf_iCub28;
    otherwise
        error('dataset not recognized')
end


% Tikhonov Parameter selection
numLambdas = 30;
minLambdaExp = -10;
maxLambdaExp = 10;
lrng = logspace(maxLambdaExp , minLambdaExp , numLambdas);


%% Alpha setting (only for recoding)

alphaArr = 0.1:0.1:1;
numAlpha = numel(alphaArr);
resultsArr = struct();

maxiter = 100;

numrep = 3;

% Instantiate storage structures
results.bat_rlsc_yesreb.testCM = zeros(numrep,numel(classes),1, numel(classes), numel(classes));
results.bat_rlsc_yesreb.bestValAccBuf = zeros(numrep,numel(classes),1);
results.bat_rlsc_yesreb.valAcc = zeros(numrep,numel(classes),1,numLambdas);
results.bat_rlsc_yesreb.teAcc = zeros(numrep,numel(classes),1,numLambdas);
results.bat_rlsc_yesreb.trainTime = zeros(numrep,numel(classes),1);

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
    
    imbClassArr = 10;
    
    for imbClass = imbClassArr
        
        % Split training set in balanced (for pretraining) and imbalanced
        % (for incremental learning) subsets
        
        [tmp1,tmp2] = find(Ytr == 1);
        idx_bal = tmp1(tmp2 ~= imbClass);
        Xtr_bal = Xtr(idx_bal , :);
        Ytr_bal = Ytr(idx_bal , :);
        ntr_bal = size(Xtr_bal,1);
        
        idx_imbal = setdiff(1:ntr , idx_bal);
        Xtr_imbal = Xtr(idx_imbal , :);
        Ytr_imbal = Ytr(idx_imbal , :);
%         ntr_imbal = size(Xtr_imbal,1);
        ntr_imbal = min([maxiter, numel(idx_imbal)]);
        
        
        % Pre-compute batch model on points not belonging to class j
        XtX = Xtr_bal'*Xtr_bal;
        XtY = Xtr_bal'*Ytr_bal;

        lstar = 0;      % Best lambda
        bestAcc = 0;    % Highest accuracy
        w = cell(1,numel(lrng));
        R = cell(1,numel(lrng));
        
        for lidx = 1:numel(lrng)

            l = lrng(lidx);

            % Train on TR1
%             w{lidx} = (XtX + ntr_bal * l * eye(d)) \ XtY;
            R{lidx} = chol(XtX + ntr_bal * l * eye(d), 'upper');  
%             R{lidx} = chol(XtX + l * eye(d), 'upper');  
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
                    results.bat_rlsc_yesreb.valAcc(k,imbClass,q,lidx) = currAcc;

                    if currAcc > bestAcc
                        bestAcc = currAcc;
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

                results.bat_rlsc_yesreb.ntr = ntr;
                results.bat_rlsc_yesreb.nte = nte;
                results.bat_rlsc_yesreb.testCM(k,imbClass,q,:,:) = CM;
                results.bat_rlsc_yesreb.bestValAccBuf(k,imbClass,q) = bestAcc;
                results.bat_rlsc_yesreb.trainTime(k,imbClass,q) = trainTime;
            end
        end



        %% Incremental RLSC, no recoding
        % Naive Linear Regularized Least Squares Classifier, 
        % with Tikhonov regularization parameter selection

        if run_inc_rlsc_norec == 1    
            
            R_tmp = cell(1,numLambdas);
            trainTime = 0;
            for q = 1:ntr_imbal

                if q == 1
                    % Compute cov mat and b
                    XtY_tmp = XtY;   
                    ntr_tmp = ntr_bal;
                end

                tic

                % Update XtY term
                XtY_tmp = XtY_tmp + Xtr_imbal(q,:)' * Ytr_imbal(q,:);
                ntr_tmp = ntr_tmp + 1;
                

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

                    results.inc_rlsc_norec.valAcc(k,imbClass,q,lidx) = currAcc;

                    if currAcc > bestAcc
                        bestAcc = currAcc;
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

                results.inc_rlsc_norec.ntr = ntr;
                results.inc_rlsc_norec.nte = nte;
                results.inc_rlsc_norec.testCM(k,imbClass,q,:,:) = CM;
                results.inc_rlsc_norec.bestValAccBuf(k,imbClass,q) = bestAcc;
                results.inc_rlsc_norec.trainTime(k,imbClass,q) = trainTime;
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
            
            for q = 1:ntr_imbal

%                 Xtr_tmp = [Xtr_tmp ; Xtr_imbal(q,:)];
%                 Ytr_tmp = [Ytr_tmp ; Ytr_imbal(q,:)];
                ntr_tmp = ntr_tmp + 1;
                Xtr_tmp(ntr_tmp,:) = Xtr_imbal(q,:);
                Ytr_tmp(ntr_tmp,:) = Ytr_imbal(q,:);
                
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
                    Gamma(i,i) = computeGamma(p,currClassIdx)^alpha;
                end
                
                % Compute b
                XtY_tmp = Xtr_tmp(1:ntr_tmp,:)' * (Gamma * Ytr_tmp(1:ntr_tmp,:));

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

                    results.inc_rlsc_yesrec.valAcc(k,imbClass,q,lidx) = currAcc;

                    if currAcc > bestAcc
                        bestAcc = currAcc;
                        lstar = l;
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
                end

                trainTime = trainTime + toc;

                results.inc_rlsc_yesrec.ntr = ntr;
                results.inc_rlsc_yesrec.nte = nte;
                results.inc_rlsc_yesrec.testCM(k,imbClass,q,:,:) = CM;
                results.inc_rlsc_yesrec.bestValAccBuf(k,imbClass,q) = bestAcc;
                results.inc_rlsc_yesrec.trainTime(k,imbClass,q) = trainTime;
            end
        end
        
        % Save workspace

        if saveResult == 1

            save([resdir '/workspace.mat'] , '-v7.3');
        end
        
    end
end

%% Print results

% clc
% display('Results');
% display(' ');
% 
% if run_bat_rlsc_yesreb == 1    
% 
%     display('Batch RLSC, exact rebalancing (sqrt(Gamma))');
%     best_val_acc_avg = mean(results.bat_rlsc_yesreb.bestValAccBuf);
%     best_val_acc_std = std(results.bat_rlsc_yesreb.bestValAccBuf,1);
% 
%     display(['Best validation accuracy = ', num2str(best_val_acc_avg) , ' +/- ' , num2str(best_val_acc_std)])
% 
%     test_acc_avg = mean(results.bat_rlsc_yesreb.testAccBuf);
%     test_acc_std = std(results.bat_rlsc_yesreb.testAccBuf,1);
% 
%     display(['Test accuracy = ', num2str(test_acc_avg) , ' +/- ' , num2str(test_acc_std)])
%     display(' ');
% end
% 
% if run_bat_rlsc_yesreb2 == 1
%     display('Batch RLSC, exact rebalancing (Gamma)');
%     best_val_acc_avg = mean(results.bat_rlsc_yesreb2.bestValAccBuf);
%     best_val_acc_std = std(results.bat_rlsc_yesreb2.bestValAccBuf,1);
% 
%     display(['Best validation accuracy = ', num2str(best_val_acc_avg) , ' +/- ' , num2str(best_val_acc_std)])
% 
%     test_acc_avg = mean(results.bat_rlsc_yesreb2.testAccBuf);
%     test_acc_std = std(results.bat_rlsc_yesreb2.testAccBuf,1);
% 
%     display(['Test accuracy = ', num2str(test_acc_avg) , ' +/- ' , num2str(test_acc_std)])
%     display(' ');    
% end
% 
% if run_bat_rlsc_noreb == 1    
% 
%     display('Batch RLSC, no rebalancing');
%     best_val_acc_avg = mean(results.bat_rlsc_noreb.bestValAccBuf);
%     best_val_acc_std = std(results.bat_rlsc_noreb.bestValAccBuf,1);
% 
%     display(['Best validation accuracy = ', num2str(best_val_acc_avg) , ' +/- ' , num2str(best_val_acc_std)])
% 
%     test_acc_avg = mean(results.bat_rlsc_noreb.testAccBuf);
%     test_acc_std = std(results.bat_rlsc_noreb.testAccBuf,1);
% 
%     display(['Test accuracy = ', num2str(test_acc_avg) , ' +/- ' , num2str(test_acc_std)]);
%     display(' ');
% end
% 
% if run_inc_rlsc_yesreb == 1    
% 
%     display('Incremental RLSC, with recoding (sqrt(Gamma))');
%     best_val_acc_avg = mean(results.inc_rlsc_yesreb.bestValAccBuf);
%     best_val_acc_std = std(results.inc_rlsc_yesreb.bestValAccBuf,1);
% 
%     display(['Best validation accuracy = ', num2str(best_val_acc_avg) , ' +/- ' , num2str(best_val_acc_std)])
% 
%     test_acc_avg = mean(results.inc_rlsc_yesreb.testAccBuf);
%     test_acc_std = std(results.inc_rlsc_yesreb.testAccBuf,1);
% 
%     display(['Test accuracy = ', num2str(test_acc_avg) , ' +/- ' , num2str(test_acc_std)]);
%     display(' ');
% end
% 
% 
% if run_inc_rlsc_yesreb2 == 1    
% 
%     display('Incremental RLSC, with recoding (Gamma)');
%     best_val_acc_avg = mean(results.inc_rlsc_yesreb2.bestValAccBuf);
%     best_val_acc_std = std(results.inc_rlsc_yesreb2.bestValAccBuf,1);
% 
%     display(['Best validation accuracy = ', num2str(best_val_acc_avg) , ' +/- ' , num2str(best_val_acc_std)])
% 
%     test_acc_avg = mean(results.inc_rlsc_yesreb2.testAccBuf);
%     test_acc_std = std(results.inc_rlsc_yesreb2.testAccBuf,1);
% 
%     display(['Test accuracy = ', num2str(test_acc_avg) , ' +/- ' , num2str(test_acc_std)]);
%     display(' ');
% end



%% Plots

% for c = 1:numel(classes)
% for c = t
for c = imbClassArr
    
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
        surf(squeeze(results.inc_rlsc_yesrec.valAcc(1,c,:,:)))

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
        
        figure
        hold on
        h1 = bandplot(1:min(maxiter,out(c,2)),squeeze(results.inc_rlsc_norec.bestValAccBuf(:,c,1:min(maxiter,out(c,2)))), ...
            'r' , 0.1 , 0 , 1 , '-');
        h2 = bandplot(1:min(maxiter,out(c,2)),squeeze(results.inc_rlsc_yesrec.bestValAccBuf(:,c,1:min(maxiter,out(c,2)))), ...
            'b' , 0.1 , 0 , 1 , '-');
        title(['Test Error for imbalanced class # ' , num2str(c)]);
        legend([h1,h2],'Naive RRLSC','Recoded RRLSC','Location','southeast')
        xlabel('n_{imb}')
        ylabel('Test Accuracy')
        hold off    

        figure
        surf(squeeze(results.inc_rlsc_yesrec.valAcc(1,c,:,:)))

        figure
        hold on
        h1 = bandplot(1:min(maxiter,out(c,2)),squeeze(results.inc_rlsc_norec.trainTime(:,c,1:min(maxiter,out(c,2)))), ...
            'r' , 0.1 , 0 , 1 , '-');
        h2 = bandplot(1:min(maxiter,out(c,2)),squeeze(results.inc_rlsc_yesrec.trainTime(:,c,1:min(maxiter,out(c,2)))), ...
            'b' , 0.1 , 0 , 1 , '-');
        title(['Training Time for imbalanced class # ' , num2str(c)]);
        legend([h1,h2],'Naive RRLSC','Recoded RRLSC','Location','southeast')
        xlabel('n_{imb}')
        ylabel('Training Time')
        hold off    
        
    
    end    
end
% 
% if run_bat_rlsc_yesreb == 1    
% 
%     % Batch RLSC, exact rebalancing
%     % figure
%     % hold on
%     % title({ 'Batch RLSC, exact rebalancing' ; ['Test accuracy over ' , num2str(numrep) , ' runs'] } );
%     % boxplot(results.bat_rlsc_yesreb.testAccBuf')
%     % hold off 
%     % 
%     % figure
%     % hold on
%     % title({ 'Batch RLSC, exact rebalancing' ; ['Validation accuracy over ' , num2str(numrep) , ' runs'] } );
%     % boxplot(results.bat_rlsc_yesreb.bestValAccBuf')
%     % hold off 
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Test accuracy vs \lambda'})
%     % contourf(lrng,1:numrep,results.bat_rlsc_yesreb.teAcc, 100, 'LineWidth',0);
%     % set(gca,'Xscale','log');
%     % xlabel('\lambda')
%     % ylabel('Repetition')
%     % colorbar
%     % hold off
% 
%     figure
%     hold on
%     title({'Batch RLSC, exact rebalancing  (sqrt(Gamma))' ; 'Test accuracy vs \lambda'})
%     bandplot( lrng , results.bat_rlsc_yesreb.teAcc , 'red' , 0.1 , 1 , 2, '-');
%     xlabel('\lambda')
%     ylabel('Test accuracy')
%     hold off
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Validation accuracy vs \lambda'})
%     % contourf(lrng,1:numrep,results.bat_rlsc_yesreb.valAcc, 100, 'LineWidth',0);
%     % set(gca,'Xscale','log');
%     % xlabel('\lambda')
%     % ylabel('Repetition')
%     % colorbar
%     % hold off
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Validation accuracy vs \lambda'})
%     % bandplot( lrng , results.bat_rlsc_yesreb.valAcc , 'red' , 0.1 , 1 , 2, '-');
%     % xlabel('\lambda')
%     % ylabel('Validation accuracy')
%     % hold off
% 
% end
% 
% if run_bat_rlsc_yesreb2 == 1    
% 
%     % Batch RLSC, exact rebalancing
%     % figure
%     % hold on
%     % title({ 'Batch RLSC, exact rebalancing' ; ['Test accuracy over ' , num2str(numrep) , ' runs'] } );
%     % boxplot(results.bat_rlsc_yesreb.testAccBuf')
%     % hold off 
%     % 
%     % figure
%     % hold on
%     % title({ 'Batch RLSC, exact rebalancing' ; ['Validation accuracy over ' , num2str(numrep) , ' runs'] } );
%     % boxplot(results.bat_rlsc_yesreb.bestValAccBuf')
%     % hold off 
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Test accuracy vs \lambda'})
%     % contourf(lrng,1:numrep,results.bat_rlsc_yesreb.teAcc, 100, 'LineWidth',0);
%     % set(gca,'Xscale','log');
%     % xlabel('\lambda')
%     % ylabel('Repetition')
%     % colorbar
%     % hold off
% 
%     figure
%     hold on
%     title({'Batch RLSC, exact rebalancing (Gamma)' ; 'Test accuracy vs \lambda'})
%     bandplot( lrng , results.bat_rlsc_yesreb2.teAcc , 'red' , 0.1 , 1 , 2, '-');
%     xlabel('\lambda')
%     ylabel('Test accuracy')
%     hold off
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Validation accuracy vs \lambda'})
%     % contourf(lrng,1:numrep,results.bat_rlsc_yesreb.valAcc, 100, 'LineWidth',0);
%     % set(gca,'Xscale','log');
%     % xlabel('\lambda')
%     % ylabel('Repetition')
%     % colorbar
%     % hold off
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Validation accuracy vs \lambda'})
%     % bandplot( lrng , results.bat_rlsc_yesreb.valAcc , 'red' , 0.1 , 1 , 2, '-');
%     % xlabel('\lambda')
%     % ylabel('Validation accuracy')
%     % hold off
% 
% end
% 
% if run_bat_rlsc_noreb == 1    
% 
%     % Batch RLSC, no rebalancing
%     % figure
%     % hold on
%     % title({ 'Batch RLSC, no rebalancing' ; ['Test accuracy over ' , num2str(numrep) , ' runs'] } );
%     % boxplot(results.bat_rlsc_noreb.testAccBuf')
%     % hold off 
%     % 
%     % figure
%     % hold on
%     % title({ 'Batch RLSC, no rebalancing' ; ['Validation accuracy over ' , num2str(numrep) , ' runs'] } );
%     % boxplot(results.bat_rlsc_noreb.bestValAccBuf')
%     % hold off 
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, no rebalancing' ; 'Test accuracy vs \lambda'})
%     % contourf(lrng,1:numrep,results.bat_rlsc_noreb.teAcc, 100, 'LineWidth',0);
%     % set(gca,'Xscale','log');
%     % xlabel('\lambda')
%     % ylabel('Repetition')
%     % colorbar
%     % hold off
% 
%     figure
%     hold on
%     title({'Batch RLSC, no rebalancing' ; 'Test accuracy vs \lambda'})
%     bandplot( lrng , results.bat_rlsc_noreb.teAcc , 'red' , 0.1 , 1 , 2, '-');
%     xlabel('\lambda')
%     ylabel('Test accuracy')
%     hold off
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, no rebalancing' ; 'Validation accuracy vs \lambda'})
%     % contourf(lrng,1:numrep,results.bat_rlsc_noreb.valAcc, 100, 'LineWidth',0);
%     % set(gca,'Xscale','log');
%     % xlabel('\lambda')
%     % ylabel('Repetition')
%     % colorbar
%     % hold off
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, no rebalancing' ; 'Validation accuracy vs \lambda'})
%     % bandplot( lrng , results.bat_rlsc_noreb.valAcc , 'red' , 0.1 , 1 , 2, '-');
%     % xlabel('\lambda')
%     % ylabel('Validation accuracy')
%     % hold off
% end
% 
% 
% 
% if run_inc_rlsc_yesreb == 1    
% 
%     % Batch RLSC, exact rebalancing
%     % figure
%     % hold on
%     % title({ 'Batch RLSC, exact rebalancing' ; ['Test accuracy over ' , num2str(numrep) , ' runs'] } );
%     % boxplot(results.bat_rlsc_yesreb.testAccBuf')
%     % hold off 
%     % 
%     % figure
%     % hold on
%     % title({ 'Batch RLSC, exact rebalancing' ; ['Validation accuracy over ' , num2str(numrep) , ' runs'] } );
%     % boxplot(results.bat_rlsc_yesreb.bestValAccBuf')
%     % hold off 
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Test accuracy vs \lambda'})
%     % contourf(lrng,1:numrep,results.bat_rlsc_yesreb.teAcc, 100, 'LineWidth',0);
%     % set(gca,'Xscale','log');
%     % xlabel('\lambda')
%     % ylabel('Repetition')
%     % colorbar
%     % hold off
% 
%     figure
%     hold on
%     title({'Incremental RLSC with recoding (sqrt(Gamma))' ; 'Test accuracy vs \lambda'})
%     bandplot( lrng , results.inc_rlsc_yesreb.teAcc , 'red' , 0.1 , 1 , 2, '-');
%     xlabel('\lambda')
%     ylabel('Test accuracy')
%     hold off
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Validation accuracy vs \lambda'})
%     % contourf(lrng,1:numrep,results.bat_rlsc_yesreb.valAcc, 100, 'LineWidth',0);
%     % set(gca,'Xscale','log');
%     % xlabel('\lambda')
%     % ylabel('Repetition')
%     % colorbar
%     % hold off
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Validation accuracy vs \lambda'})
%     % bandplot( lrng , results.bat_rlsc_yesreb.valAcc , 'red' , 0.1 , 1 , 2, '-');
%     % xlabel('\lambda')
%     % ylabel('Validation accuracy')
%     % hold off
% 
% end
% 
% 
% 
% 
% if run_inc_rlsc_yesrec2 == 1    
% 
%     % Batch RLSC, exact rebalancing
%     % figure
%     % hold on
%     % title({ 'Batch RLSC, exact rebalancing' ; ['Test accuracy over ' , num2str(numrep) , ' runs'] } );
%     % boxplot(results.bat_rlsc_yesreb.testAccBuf')
%     % hold off 
%     % 
%     % figure
%     % hold on
%     % title({ 'Batch RLSC, exact rebalancing' ; ['Validation accuracy over ' , num2str(numrep) , ' runs'] } );
%     % boxplot(results.bat_rlsc_yesreb.bestValAccBuf')
%     % hold off 
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Test accuracy vs \lambda'})
%     % contourf(lrng,1:numrep,results.bat_rlsc_yesreb.teAcc, 100, 'LineWidth',0);
%     % set(gca,'Xscale','log');
%     % xlabel('\lambda')
%     % ylabel('Repetition')
%     % colorbar
%     % hold off
% 
%     figure
%     hold on
%     title({'Incremental RLSC with recoding (Gamma)' ; 'Test accuracy vs \lambda'})
%     bandplot( lrng , results.inc_rlsc_yesrec.valAcc , 'red' , 0.1 , 1 , 2, '-');
%     xlabel('\lambda')
%     ylabel('Test accuracy')
%     hold off
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Validation accuracy vs \lambda'})
%     % contourf(lrng,1:numrep,results.bat_rlsc_yesreb.valAcc, 100, 'LineWidth',0);
%     % set(gca,'Xscale','log');
%     % xlabel('\lambda')
%     % ylabel('Repetition')
%     % colorbar
%     % hold off
% 
%     % figure
%     % hold on
%     % title({'Batch RLSC, exact rebalancing' ; 'Validation accuracy vs \lambda'})
%     % bandplot( lrng , results.bat_rlsc_yesreb.valAcc , 'red' , 0.1 , 1 , 2, '-');
%     % xlabel('\lambda')
%     % ylabel('Validation accuracy')
%     % hold off
% 
% end


% %%  Play sound
% 
% load gong;
% player = audioplayer(y, Fs);
% play(player);
