data = load('mnist_all.mat');


%% Training

ntr = 5000*ones(1,10);
ntr(10) = 200;

ntrtot = sum(ntr);

tridx = cell(1,10);
Xtr = [];
Ytr = -ones(ntrtot,10);

for i = 0:9
    
    currentFieldStr = strcat('train' , num2str(i));
    tridx{i+1} = randperm(size(data.(currentFieldStr),1),ntr(i+1));
    
    Xtr = double([Xtr ; data.(currentFieldStr)(tridx{i+1},:)]);
    Ytr(sum(ntr(1:i))+1:sum(ntr(1:i+1)),i+1) = ones(numel(tridx{i+1}),1);

end

% Shuffle training set

idxRndTr = randperm(ntrtot);
Xtr = Xtr(idxRndTr,:);
Ytr = Ytr(idxRndTr,:);

% Compute class frequencies
gamma = ntr / ntrtot;


%% Test

nte = 800*ones(1,10);

ntetot = sum(nte);

teidx = cell(1,10);
Xte = [];
Yte = -ones(ntetot,10);

for i = 0:9
    
    currentFieldStr = strcat('test' , num2str(i));
    teidx{i+1} = randperm(size(data.(currentFieldStr),1),nte(i+1));
    
    Xte = double([Xte ; data.(currentFieldStr)(teidx{i+1},:)]);
    Yte(sum(nte(1:i))+1:sum(nte(1:i+1)),i+1) = ones(numel(teidx{i+1}),1);

end