% Random dataset

ntr = 4000;     % Number of training samples
nte = 1000;     % Number of test samples
n = ntr + nte;  % Total number of examples
d = 100;        % Dimensionality
t = 2;          % Number of classes
classFrequenciesTrain = [0.2 0.8]; % Balance for each class in the training set
if sum(classFrequenciesTrain) ~= 1
    error('Sum of class frequencies must be = 1')
end
classFrequenciesTest = [0.5 0.5]; % Balance for each class in the training set
if sum(classFrequenciesTest) ~= 1
    error('Sum of class frequencies must be = 1')
end

if mod(n,t) >0
    error('n cannot be divided by t')
end

X = rand(n,d);
% tmp = randi(t,n,1);
% tmp = repmat(1:10,n/t,1);
% tmp = reshape(tmp,n,1);
% tmp = randperm(n);
%

% Training labels
tmp = [];
for i = 1:t
    tmp = [tmp ; i*ones(round(ntr*classFrequenciesTrain(i)),1)];
end
tmp = tmp(randperm(ntr),1);

if t  == 2
    Ytr = -ones(ntr,1);
    for i = 1:ntr
        if mod(tmp(i),2)==0
            Ytr(i,1) = 1;
        end
    end
else
    Ytr = zeros(ntr,t);
    for i = 1:ntr
        Ytr(i,tmp(i)) = 1;
    end
end

clear tmp;


% Test labels
tmp = [];
for i = 1:t
    tmp = [tmp ; i*ones(round(nte*classFrequenciesTest(i)),1)];
end
tmp = tmp(randperm(nte),1);

if t  == 2
    Yte = -ones(nte,1);
    for i = 1:nte
        if mod(tmp(i),2)==0
            Yte(i,1) = 1;
        end
    end
else
    Yte = zeros(nte,t);
    for i = 1:nte
        Yte(i,tmp(i)) = 1;
    end
end

clear tmp;


Xtr = X(1:ntr,:);
Xte = X(ntr+1:ntr+nte,:);
% Ytr = Y(1:ntr,:);
% Yte = Y(ntr+1:ntr+nte,:);

% 
% % Format dataset in class-specific cells
% 
% X_c = cell(1,t);
% Y_c = cell(1,t);    
% Xtr_c = cell(1,t);
% Ytr_c = cell(1,t);    
% Xte_c = cell(1,t);
% Yte_c = cell(1,t);
% 
% for sampleidx = 1:n
%     for classidx = 1:t
%         if Y(sampleidx,classidx) == 1
% %                 Y_c{classidx}(sampleidx,:) = Y(sampleidx,:);
% %                 X_c{classidx}(sampleidx,:) = X(sampleidx,:);
%             Y_c{classidx} = [ Y_c{classidx} ; Y(sampleidx,:)];
%             X_c{classidx} = [ X_c{classidx} ; X(sampleidx,:)];
% %                 break
%         end
%     end
% end
% 
% for classidx = 1:t
%     Xtr_c{classidx} = X_c{classidx}(1:(ntr/t),:);
%     Ytr_c{classidx} = Y_c{classidx}(1:(ntr/t),:);    
%     Xte_c{classidx} = X_c{classidx}((ntr/t)+1:(ntr+nte)/t,:);
%     Yte_c{classidx} = Y_c{classidx}((ntr/t)+1:(ntr+nte)/t,:);
% end
