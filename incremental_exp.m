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
run_bat_rlsc_yesreb = 0;    % Batch with exact rebalancing
run_inc_rlsc_norec = 1;     % Naive incremental with no recoding
run_inc_rlsc_yesrec = 1;    % Incremental with recoding

retrain = 1;
trainPart = 0.8;

dataRoot =  '/home/kammo/Repos/ior/data/caffe_centralcrop_meanimagenet2012/';
trainFolder = {'lunedi22','martedi23','mercoledi24','venerdi26'};
testFolder = {'lunedi22','martedi23','mercoledi24','venerdi26'};

% ntr = [];
ntr = 1000;
nte = []; 

% classes = 1:28; % classes to be extracted
% classes = 1:4; % classes to be extracted
classes = 1:4:28; % classes to be extracted
% classes = [1 8]; % classes to be extracted
% classes = 0:9; % classes to be extracted

% Class frequencies for train and test sets
% trainClassFreq = [0.1 0.9];
% trainClassFreq = [ 0.1067*ones(1,9) 0.04];
% trainClassFreq = [ 200*ones(1,9) , 10] / ntr;
% trainClassFreq = [0.1658*ones(1,6) 0.005];
% trainClassFreq = [0.1658*ones(1,2) 0.005 0.1658*ones(1,4)];
% trainClassFreq = [0.1633*ones(1,2) 0.02 0.1633*ones(1,4)];
% trainClassFreq = [0.3250*ones(1,3) 0.025];
% trainClassFreq = [0.0369*ones(1,27) 0.0025];
trainClassFreq = [];
testClassFreq = [];

% Tikhonov Parameter selection
numLambdas = 10;
minLambdaExp = -5;
maxLambdaExp = 10;
lrng = logspace(maxLambdaExp , minLambdaExp , numLambdas);


% Recoding parameter
alpha = 1/2;

numrep = 1;

results.bat_rlsc_yesreb.testCM = zeros(numrep,numel(classes),1, numel(classes), numel(classes));
results.bat_rlsc_yesreb.bestValAccBuf = zeros(numrep,numel(classes),1);
results.bat_rlsc_yesreb.valAcc = zeros(numrep,numel(classes),1,numLambdas);
results.bat_rlsc_yesreb.teAcc = zeros(numrep,numel(classes),1,numLambdas);
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

%     if ~exist('ds','var')
%         ds = iCubWorld28(ntr , nte, 'plusMinusOne' , 1, 1, 0, {classes , trainClassFreq, testClassFreq, {}, trainFolder, testFolder});
        ds = iCubWorld28(ntr , nte, 'zeroOne' , 1, 1, 0, {classes , trainClassFreq, testClassFreq, {}, trainFolder, testFolder});
%         ds = MNIST(ntr , nte, 'zeroOne' , 0, 0, 0, {classes , trainClassFreq, testClassFreq});
%         mix up sampled points
        ds.mixUpTrainIdx;
        ds.mixUpTestIdx;
%     else
%         % mix up sampled points
%         ds.trainIdx = ds.trainIdx(randperm(numel(ds.trainIdx)));
%         ds.testIdx = ds.testIdx(randperm(numel(ds.testIdx)));
%     end
%     

    Xtr = ds.X(ds.trainIdx,:);
    Ytr = ds.Y(ds.trainIdx,:);
    Xte = ds.X(ds.testIdx,:);
    Yte = ds.Y(ds.testIdx,:);
    
    % Subsample features
%     Xtr = Xtr(:,1:4:end);
%     Xte = Xte(:,1:4:end);
    
    ntr = size(Xtr,1);
    nte = size(Xte,1);
    d = size(Xtr,2);
    t  = size(Ytr,2);
%     p = ds.trainClassNum / ntr;

    % Splitting
%     ntr1 = round(ntr*trainPart);
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
    
    
    for j = 1:t
        
        % Split training set in balanced (for pretraining) and imbalanced
        % (for incremental learning) subsets
        
        [~,tmp] = find(Ytr == 1);
        idx_bal = find(tmp ~= j);
        Xtr_bal = Xtr(idx_bal , :);
        Ytr_bal = Ytr(idx_bal , :);
        ntr_bal = size(Xtr_bal,1);
        
        idx_imbal = setdiff(1:ntr , idx_bal);
        Xtr_imbal = Xtr(idx_imbal , :);
        Ytr_imbal = Ytr(idx_imbal , :);
        ntr_imbal = size(Xtr_imbal,1);
        
        
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
            w{lidx} = (XtX + ntr_bal * l * eye(d)) \ XtY;
            R{lidx} = chol(XtX + ntr_bal * l * eye(d), 'upper');  
        end
        
    
        %% Batch RLSC, exact rebalancing
        % Naive Linear Regularized Least Squares Classifier, 
        % with Tikhonov regularization parameter selection

        if run_bat_rlsc_yesreb == 1
            
            Xtr_tmp = Xtr_bal;
            Ytr_tmp = Ytr_bal;
            
            for q = 1:ntr_imbal
                
                Xtr_tmp = [Xtr_tmp ; Xtr_imbal(q,:)];
                Ytr_tmp = [Ytr_tmp ; Ytr_imbal(q,:)];
                ntr_tmp = size(Xtr_tmp,1);
                
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
                        [currAcc , ~] = weightedAccuracy2( Yval1, Yval1pred , classes);
                    else

                        CM = confusionmat(Yval1,sign(Yval1pred_raw));
                        CM = CM ./ repmat(sum(CM,2),1,2);
                        currAcc = trace(CM)/2;                
                    end
                    results.bat_rlsc_yesreb.valAcc(k,j,q,lidx) = currAcc;

                    if currAcc > bestAcc
                        bestAcc = currAcc;
                        lstar = l;
                    end

                    % Compute current test accuracy
                    Ytepred_raw = Xte * w;

                    if t > 2
                        Ytepred = ds.scoresToClasses(Ytepred_raw);
                        [currAcc , CM] = weightedAccuracy2( Yte, Ytepred , classes);
                    else
                        CM = confusionmat(Yte,sign(Ytepred_raw));
                        CM = CM ./ repmat(sum(CM,2),1,2);
                        currAcc = trace(CM)/2;                
                    end
                    results.bat_rlsc_yesreb.teAcc(k,j,q,lidx) = currAcc;
                end

                results.bat_rlsc_yesreb.ntr = ntr;
                results.bat_rlsc_yesreb.nte = nte;
                results.bat_rlsc_yesreb.testCM(k,j,q,:,:) = CM;
                results.bat_rlsc_yesreb.bestValAccBuf(k,j,q) = bestAcc;
            end
        end



        %% Incremental RLSC, no recoding
        % Naive Linear Regularized Least Squares Classifier, 
        % with Tikhonov regularization parameter selection

        if run_inc_rlsc_norec == 1    
            
            R_tmp = cell(1,numLambdas);
            for q = 1:ntr_imbal

                if q == 1
                    % Compute cov mat and b
                    XtX_tmp = XtX;
                    XtY_tmp = XtY;   
                    ntr_tmp = size(XtX_tmp,1);
                end

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
                        [currAcc , ~] = weightedAccuracy2( Yval1, Yval1pred , classes);
                    else
                        CM = confusionmat(Yval1,sign(Yval1pred_raw));
                        CM = CM ./ repmat(sum(CM,2),1,2);
                        currAcc = trace(CM)/2;                
                    end

                    results.inc_rlsc_norec.valAcc(k,j,q,lidx) = currAcc;

                    if currAcc > bestAcc
                        bestAcc = currAcc;
                        lstar = l;
                    end

                    % Compute current test accuracy
                    Ytepred_raw = Xte * w;

                    if t > 2
                        Ytepred = ds.scoresToClasses(Ytepred_raw);
                        [currAcc , CM] = weightedAccuracy2( Yte, Ytepred , classes);
                    else
                        CM = confusionmat(Yte,sign(Ytepred_raw));
                        CM = CM ./ repmat(sum(CM,2),1,2);
                        currAcc = trace(CM)/2;                
                    end

                    results.inc_rlsc_norec.teAcc(k,j,q,lidx) = currAcc;
                end

                results.inc_rlsc_norec.ntr = ntr;
                results.inc_rlsc_norec.nte = nte;
                results.inc_rlsc_norec.testCM(k,j,q,:,:) = CM;
                results.inc_rlsc_norec.bestValAccBuf(k,j,q) = bestAcc;
            end
        end


        %% Incremental RLSC, recoding
        % Naive Linear Regularized Least Squares Classifier, 
        % with Tikhonov regularization parameter selection

        if run_inc_rlsc_yesrec == 1    
            
            Xtr_tmp = Xtr_bal;
            Ytr_tmp = Ytr_bal;
            R_tmp = cell(1,numLambdas);
            for q = 1:ntr_imbal

                Xtr_tmp = [Xtr_tmp ; Xtr_imbal(q,:)];
                Ytr_tmp = [Ytr_tmp ; Ytr_imbal(q,:)];
                ntr_tmp = size(Xtr_tmp,1);
                
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
                XtY_tmp = Xtr_tmp' * Gamma * Ytr_tmp;

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
                        [currAcc , ~] = weightedAccuracy2( Yval1, Yval1pred , classes);
                    else
                        CM = confusionmat(Yval1,sign(Yval1pred_raw));
                        CM = CM ./ repmat(sum(CM,2),1,2);
                        currAcc = trace(CM)/2;                
                    end

                    results.inc_rlsc_yesrec.valAcc(k,j,q,lidx) = currAcc;

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

                    results.inc_rlsc_yesrec.teAcc(k,j,q,lidx) = currAcc;
                end

                results.inc_rlsc_yesrec.ntr = ntr;
                results.inc_rlsc_yesrec.nte = nte;
                results.inc_rlsc_yesrec.testCM(k,j,q,:,:) = CM;
                results.inc_rlsc_yesrec.bestValAccBuf(k,j,q) = bestAcc;
            end
        end
        
        % Save workspace

        if saveResult == 1

            save([resdir '/workspace.mat']);
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

for c = 1:numel(classes)
    figure
    hold on
    plot(squeeze(mean(results.inc_rlsc_norec.bestValAccBuf(:,c,1:out(c,2)),1)))
    plot(squeeze(mean(results.inc_rlsc_yesrec.bestValAccBuf(:,c,1:out(c,2)),1)))
    title(['Test Error for imbalanced class # ' , num2str(c)]);
    hold off
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
% if run_inc_rlsc_yesreb2 == 1    
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
%     bandplot( lrng , results.inc_rlsc_yesreb2.teAcc , 'red' , 0.1 , 1 , 2, '-');
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


%%  Play sound

load gong;
player = audioplayer(y, Fs);
play(player);
