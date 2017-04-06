function phow_caltech101()
% PHOW_CALTECH101 Image classification in the Caltech-101 dataset
%   This program demonstrates how to use VLFeat to construct an image
%   classifier on the Caltech-101 data. The classifier uses PHOW
%   features (dense SIFT), spatial histograms of visual words, and a
%   Chi2 SVM. To speedup computation it uses VLFeat fast dense SIFT,
%   kd-trees, and homogeneous kernel map. The program also
%   demonstrates VLFeat PEGASOS SVM solver, although for this small
%   dataset other solvers such as LIBLINEAR can be more efficient.
%
%   By default 15 training images are used, which should result in
%   about 64% performance (a good performance considering that only a
%   single feature type is being used).
%
%   Call PHOW_CALTECH101 to train and test a classifier on a small
%   subset of the Caltech-101 data. Note that the program
%   automatically downloads a copy of the Caltech-101 data from the
%   Internet if it cannot find a local copy.
%
%   Edit the PHOW_CALTECH101 file to change the program configuration.
%
%   To run on the entire dataset change CONF.TINYPROBLEM to FALSE.
%
%   The Caltech-101 data is saved into CONF.CALDIR, which defaults to
%   'data/caltech-101'. Change this path to the desired location, for
%   instance to point to an existing copy of the Caltech-101 data.
%
%   The program can also be used to train a model on custom data by
%   pointing CONF.CALDIR to it. Just create a subdirectory for each
%   class and put the training images there. Make sure to adjust
%   CONF.NUMTRAIN accordingly.
%
%   Intermediate files are stored in the directory CONF.DATADIR. All
%   such files begin with the prefix CONF.PREFIX, which can be changed
%   to test different parameter settings without overriding previous
%   results.
%
%   The program saves the trained model in
%   <CONF.DATADIR>/<CONF.PREFIX>-model.mat. This model can be used to
%   test novel images independently of the Caltech data.
%
%     load('data/baseline-model.mat') ; # change to the model path
%     label = model.classify(model, im) ;
%

% Author: Andrea Vedaldi

% Copyright (C) 2011-2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

run('vlfeat-0.9.20/toolbox/vl_setup');

conf.calDir = 'data/imagenet' ;
conf.dataDir = 'data/' ;
conf.autoDownloadData = true ;
conf.numTrain = 15 ;
conf.numTest = 15 ;
conf.numClasses = 200 ;
conf.numWords = 600 ;
conf.numSpatialX = [2 4] ;
conf.numSpatialY = [2 4] ;
conf.quantizer = 'kdtree' ;
conf.svm.C = 10 ;

conf.svm.solver = 'sdca' ;
%conf.svm.solver = 'sgd' ;
%conf.svm.solver = 'liblinear' ;

conf.svm.biasMultiplier = 1 ;
conf.phowOpts = {'Step', 3} ;
conf.clobber = false ;
conf.tinyProblem = false ;
conf.prefix = ['baseline_' num2str(conf.numTrain) '_' num2str(conf.numTrain) ...
               '_'  num2str(conf.numWords)  '_x_'  sprintf('%d_', conf.numSpatialX) ...
               'y_', sprintf('%d_', conf.numSpatialY) num2str(conf.svm.C)];
conf.randSeed = 1 ;

if conf.tinyProblem
  conf.prefix = 'tiny' ;
  conf.numClasses = 5 ;
  conf.numSpatialX = 2 ;
  conf.numSpatialY = 2 ;
  conf.numWords = 300 ;
  conf.phowOpts = {'Verbose', 2, 'Sizes', 7, 'Step', 5} ;
end

conf.vocabPath = fullfile(conf.dataDir, [conf.prefix '-vocab.mat']) ;
conf.histPath = fullfile(conf.dataDir, [conf.prefix '-hists.mat']) ;
conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']) ;
conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']) ;

randn('state',conf.randSeed) ;
rand('state',conf.randSeed) ;
vl_twister('state',conf.randSeed) ;

% --------------------------------------------------------------------
%                                            Download Caltech-101 data
% --------------------------------------------------------------------

if ~exist(conf.calDir, 'dir') || ...
   (~exist(fullfile(conf.calDir, 'train', 'acorn'),'dir') && ...
    ~exist(fullfile(conf.calDir, 'imageNet200', 'train', 'acorn')))
  if ~conf.autoDownloadData
    error(...
      ['Imagenet data not found. ' ...
       'Set conf.autoDownloadData=true to download the required data.']) ;
  end
  vl_xmkdir(conf.calDir) ;
  calUrl = ['http://157.253.63.7/imageNet200.tar'] ;
  fprintf('Downloading Imagenet data to ''%s''. This will take a while.', conf.calDir) ;
  untar(calUrl, conf.calDir) ;
end

if ~exist(fullfile(conf.calDir, 'train', 'acorn'),'dir')
  conf.calDir = fullfile(conf.calDir, 'imageNet200') ;
end

% --------------------------------------------------------------------
%                                                           Setup data
% --------------------------------------------------------------------
traindir = fullfile(conf.calDir, 'train');
testdir = fullfile(conf.calDir, 'test');
classes = dir(traindir) ;
classes = classes([classes.isdir]) ;
classes = {classes(3:conf.numClasses+2).name} ;

selTrain = {} ;
trainLabels = {} ;
for ci = 1:length(classes)
  ims = dir(fullfile(traindir, classes{ci}, '*.JPEG'))' ;
  ims = vl_colsubset(ims, conf.numTrain);
  ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
  selTrain = {selTrain{:}, ims{:}} ;
  trainLabels{end+1} = ci * ones(1,length(ims)) ;
end
% selTrain = find(mod(0:length(images)-1, conf.numTrain) < conf.numTrain) ;
% selTest = setdiff(1:length(images), selTrain) ;
trainLabels = cat(2, trainLabels{:});

selTest = {};
testLabels = {};
for ci = 1:length(classes)
  ims = dir(fullfile(testdir, classes{ci}, '*.JPEG'))' ;
  ims = vl_colsubset(ims, conf.numTest);
  ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
  selTest = {selTest{:}, ims{:}} ;
  testLabels{end+1} = ci * ones(1,length(ims)) ;
end

testLabels = cat(2, testLabels{:});


model.classes = classes ;
model.phowOpts = conf.phowOpts ;
model.numSpatialX = conf.numSpatialX ;
model.numSpatialY = conf.numSpatialY ;
model.quantizer = conf.quantizer ;
model.vocab = [] ;
model.w = [] ;
model.b = [] ;
model.classify = @classify ;

% --------------------------------------------------------------------
%                                                     Train vocabulary
% --------------------------------------------------------------------

if ~exist(conf.vocabPath) || conf.clobber

  % Get some PHOW descriptors to train the dictionary
  selTrainFeats = vl_colsubset(selTrain, 30) ;
  descrs = {} ;
  %for ii = 1:length(selTrainFeats)
  parfor ii = 1:length(selTrainFeats)
    im = imread(fullfile(traindir, selTrain{selTrainFeats(ii)})) ;
    im = standarizeImage(im) ;
    [drop, descrs{ii}] = vl_phow(im, model.phowOpts{:}) ;
  end

  descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
  descrs = single(descrs) ;

  % Quantize the descriptors to get the visual words
  vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
  save(conf.vocabPath, 'vocab') ;
else
  load(conf.vocabPath) ;
end

model.vocab = vocab ;

if strcmp(model.quantizer, 'kdtree')
  model.kdtree = vl_kdtreebuild(vocab) ;
end

% --------------------------------------------------------------------
%                                           Compute spatial histograms
% --------------------------------------------------------------------

if ~exist(conf.histPath) || conf.clobber
  train_hists = {} ;
  parfor ii = 1:length(selTrain)
  % for ii = 1:length(selTrain)
    fprintf('Processing %s (%.2f %%)\n', selTrain{ii}, 100 * ii / length(selTrain)) ;
    im = imread(fullfile(traindir, selTrain{ii})) ;
    train_hists{ii} = getImageDescriptor(model, im);
  end

  train_hists = cat(2, train_hists{:});

  test_hists = {} ;
  parfor ii = 1:length(selTest)
  % for ii = 1:length(selTest)
    fprintf('Processing %s (%.2f %%)\n', selTest{ii}, 100 * ii / length(selTest)) ;
    im = imread(fullfile(testdir, selTest{ii})) ;
    test_hists{ii} = getImageDescriptor(model, im);
  end

  test_hists = cat(2, test_hists{:}) ;
  save(conf.histPath, 'train_hists', 'test_hists') ;
else
  load(conf.histPath) ;
end

% --------------------------------------------------------------------
%                                                  Compute feature map
% --------------------------------------------------------------------

psix_train = vl_homkermap(train_hists, 1, 'kchi2', 'gamma', .5) ;
psix_test = vl_homkermap(test_hists, 1, 'kchi2', 'gamma', .5) ;

% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------

if ~exist(conf.modelPath) || conf.clobber
  switch conf.svm.solver
    case {'sgd', 'sdca'}
      lambda = 1 / (conf.svm.C *  length(selTrain)) ;
      w = [] ;
      parfor ci = 1:length(classes)
        perm = randperm(length(selTrain)) ;
        fprintf('Training model for class %s\n', classes{ci}) ;
        y = 2 * (trainClasses == ci) - 1 ;
        [w(:,ci) b(ci) info] = vl_svmtrain(psix_train(:, perm), y(perm), lambda, ...
          'Solver', conf.svm.solver, ...
          'MaxNumIterations', 50/lambda, ...
          'BiasMultiplier', conf.svm.biasMultiplier, ...
          'Epsilon', 1e-3);
      end

    case 'liblinear'
      svm = train(imageClass(selTrain)', ...
                  sparse(double(psix(:,selTrain))),  ...
                  sprintf(' -s 3 -B %f -c %f', ...
                          conf.svm.biasMultiplier, conf.svm.C), ...
                  'col') ;
      w = svm.w(:,1:end-1)' ;
      b =  svm.w(:,end)' ;
  end

  model.b = conf.svm.biasMultiplier * b ;
  model.w = w ;

  save(conf.modelPath, 'model') ;
else
  load(conf.modelPath) ;
end

% --------------------------------------------------------------------
%                                                Test SVM and evaluate
% --------------------------------------------------------------------

% Estimate the class of the test images
scores = model.w' * psix_test + model.b' * ones(1,size(psix_test,2)) ;
[drop, imageEstClass] = max(scores, [], 1) ;
acc = mean(testLabels(:) == imageEstClass(:)) * 100
fprintf('Accuracy: %g', acc);
% Compute the confusion matrix
idx = sub2ind([length(classes), length(classes)], ...
              testLabels, imageEstClass) ;
confus = zeros(length(classes)) ;
confus = vl_binsum(confus, ones(size(idx)), idx) ;

% Plots
% figure('visible','off') ; clf;
% subplot(1,2,1) ;
% imagesc(scores(:,[selTrain selTest])) ; title('Scores') ;
% set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
% subplot(1,2,2) ;
% imagesc(confus) ;
% title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
%               100 * mean(diag(confus)/conf.numTest) )) ;
% print('-depsc2', [conf.resultPath '.ps']) ;
save([conf.resultPath '.mat'], 'confus', 'conf') ;

% -------------------------------------------------------------------------
function im = standarizeImage(im)
% -------------------------------------------------------------------------

im = im2single(im) ;
if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end

% -------------------------------------------------------------------------
function hist = getImageDescriptor(model, im)
% -------------------------------------------------------------------------

im = standarizeImage(im) ;
width = size(im,2) ;
height = size(im,1) ;
numWords = size(model.vocab, 2) ;

% get PHOW features
[frames, descrs] = vl_phow(im, model.phowOpts{:}) ;

% quantize local descriptors into visual words
switch model.quantizer
  case 'vq'
    [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
  case 'kdtree'
    binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                                  single(descrs), ...
                                  'MaxComparisons', 50)) ;
end

for i = 1:length(model.numSpatialX)
  binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
  binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

  % combined quantization
  bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  hists{i} = single(hist / sum(hist)) ;
end
hist = cat(1,hists{:}) ;
hist = hist / sum(hist) ;

% -------------------------------------------------------------------------
function [className, score] = classify(model, im)
% -------------------------------------------------------------------------

hist = getImageDescriptor(model, im) ;
psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5) ;
scores = model.w' * psix + model.b' ;
[score, best] = max(scores) ;
className = model.classes{best} ;
