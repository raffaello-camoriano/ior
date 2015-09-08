ICUBWORLDopts = ICUBWORLDinit('iCubWorld30');
obj_names = keys(ICUBWORLDopts.objects)';
dataroot = '/home/kammo/Repos/ior/data/caffe_centralcrop_meanimagenet2012/';
iorroot = '/home/kammo/Repos/ior/';

numClasses = numel(classes); % number of classes

if (numel(trainClassFreq) ~= numel(trainClassFreq))  || (numel(trainClassFreq) ~= numel(classes))
    error('Number of classes and class frequencies specifications differ')
end

dataset_train = Features.GenericFeature();
dataset_test = Features.GenericFeature();

train_root = '/home/kammo/Repos/ior/data/caffe_centralcrop_meanimagenet2012/train/lunedi22';
test_root = '/home/kammo/Repos/ior/data/caffe_centralcrop_meanimagenet2012/test/martedi23';

feat_ext = '.mat';

dataset_train.load_feat(train_root, [], feat_ext, [], []);
dataset_test.load_feat(test_root, [], feat_ext, [], []);

XtrTmp = dataset_train.Feat';
XteTmp = dataset_test.Feat';

YtrTmp = create_y(dataset_train.Registry, obj_names, []);
YteTmp = create_y(dataset_test.Registry, obj_names, []);

classIdxTr = cell(1,numClasses);
classIdxTe = cell(1,numClasses);
for i = 1:numClasses
    classIdxTr{i} = find(YtrTmp(:,classes(i)) == 1);
    classIdxTe{i} = find(YteTmp(:,classes(i)) == 1);
end

Xtr = [];
Xte = [];
Ytr = [];
Yte = [];

for i = 1:numClasses
    Xtr = [ Xtr ; XtrTmp(classIdxTr{i},:)];
    Xte = [ Xte ; XteTmp(classIdxTe{i},:)];
    Ytr = [ Ytr ; YtrTmp(classIdxTr{i},:)];
    Yte = [ Yte ; YteTmp(classIdxTe{i},:)];
end

% Apply training class frequencies
trainClassNum = [];
for i = 1:numClasses    
    trainClassNum = [trainClassNum , round((trainClassFreq(i) * numel(classIdxTr{i}))/max(trainClassFreq)) ];
end

nte = size(Xte,1);

% Shuffle idx
trainIdx = cell(1,numClasses);
for i = 1:numClasses
    trainIdx{i} = classIdxTr{i}(randperm(numel(classIdxTr{i}) , trainClassNum(i)));
end

testIdx = randperm(nte);

Xtrtmp = [];
Ytrtmp = [];
for i = 1:numClasses
    Xtrtmp = [Xtrtmp ; Xtr(trainIdx{i},:)];
    Ytrtmp = [Ytrtmp ; Ytr(trainIdx{i},:)];
end

Xtr = Xtrtmp;
Ytr = Ytrtmp;
d = size(Xtr,2);
t  = size(Ytr,2);

clear Xtrtmp Ytrtmp;
ntr = size(Xtr,1);
trainIdx2 = randperm(ntr);

Xtr = Xtr(trainIdx2,:);
Ytr = Ytr(trainIdx2,:);

Xte = Xte(testIdx,:);
Yte = Yte(testIdx,:);

