data = load('mnist_all.mat');


%% Training

ntrc1 = 950;
ntrc2 = 50;
ntrtot = ntrc1 + ntrc2;

tridxc1 = randperm(size(data.train1,1),ntrc1);
tridxc2 = randperm(size(data.train8,1),ntrc2);

Xtr = data.train1(tridxc1,:);
Xtr = double([Xtr ; data.train8(tridxc2,:)]);

Ytr = ones(ntrc1,1);
Ytr = [Ytr ; -ones(ntrc2,1)];

idxRndTr = randperm(ntrtot);
Xtr = Xtr(idxRndTr,:);
Ytr = Ytr(idxRndTr,:);

%% Test

ntec1 = 500;
ntec2 = 500;
ntetot = ntec1 + ntec2;

tridxc1 = randperm(size(data.test1,1),ntec1);
tridxc2 = randperm(size(data.test8,1),ntec2);

Xte = data.test1(1:ntec1,:);
Xte = double([Xte ; data.test8(1:ntec2,:)]);

Yte = ones(ntec1,1);
Yte = [Yte ; -ones(ntec2,1)];

%% Compute frequencies

gammac1 = ntrc1 / ntrtot;
gammac2 = ntrc2 / ntrtot;
