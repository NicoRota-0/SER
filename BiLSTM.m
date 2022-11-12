% MACHINE LEARNING PROJECT: BiLSTM EMOTION CLASSIFICATION
clear all
clc
addpath(genpath(pwd));
disp(pwd);
DATASET_PATH = 'dataset/dataset_x_net';


% DATA LOADING*************************************************************
ads = audioDatastore(fullfile(DATASET_PATH));

filepaths = ads.Files;
emotionCodes = cellfun(@(x)x(end-5),filepaths,'UniformOutput',false);
emotions = replace(emotionCodes,{'W','S','E','A','F','T','N'}, ...
    {'angry','surprise','disgust','fear','happy','sad','neutral'});

speakerCodes = cellfun(@(x)x(end-4),filepaths,'UniformOutput',false);
labelTable = cell2table([speakerCodes,emotions],'VariableNames',{'Speaker','Emotion'});
labelTable.Emotion = categorical(labelTable.Emotion);
labelTable.Speaker = categorical(labelTable.Speaker);
summary(labelTable)
ads.Labels = labelTable;
fs = 16000;

% DATA AGUMENTATION********************************************************
%{
disp('agumenting data...');
numAugmentations = 4;
augmenter = audioDataAugmenter('NumAugmentations',numAugmentations, ...
    'TimeStretchProbability',0, ...
    'VolumeControlProbability',0, ...
    ...
    'PitchShiftProbability',0.5, ...
    ...
    'TimeShiftProbability',1, ...
    'TimeShiftRange',[-0.3,0.3], ...
    ...
    'AddNoiseProbability',1, ...
    'SNRRange', [-20,40]);

currentDir = pwd;
writeDirectory = fullfile(currentDir,'augmentedData');
mkdir(writeDirectory)

N = numel(ads.Files)*numAugmentations;

reset(ads)

numPartitions = 18;

tic
parfor ii = 1:numPartitions
    adsPart = partition(ads,numPartitions,ii);
    while hasdata(adsPart)
        [x,adsInfo] = read(adsPart);
        data = augment(augmenter,x,fs);

        [~,fn] = fileparts(adsInfo.FileName);
        for i = 1:size(data,1)
            augmentedAudio = data.Audio{i};
            augmentedAudio = augmentedAudio/max(abs(augmentedAudio),[],'all');
            augNum = num2str(i);
            if numel(augNum)==1
                iString = ['0',augNum];
            else
                iString = augNum;
            end
            audiowrite(fullfile(writeDirectory,sprintf('%s_aug%s.wav',fn,iString)),augmentedAudio,fs);
        end
    end
end

fprintf('Augmentation complete (%0.2f minutes).\n',toc/60)

adsAug = audioDatastore(writeDirectory);
adsAug.Labels = repelem(ads.Labels,augmenter.NumAugmentations,1);
%}
disp('feature extraction...');
% FEATURE EXTRACTION*******************************************************
win = hamming(round(0.03*fs),"periodic");
overlapLength = 0;

afe = audioFeatureExtractor( ...
    'Window',win, ...
    'OverlapLength',overlapLength, ...
    'SampleRate',fs, ...
    ...
    'gtcc',true, ...
    'gtccDelta',true, ...
    'mfccDelta',true, ...
    ...
    'SpectralDescriptorInput','melSpectrum', ...
    'spectralCrest',true);


% CROSS VALIDATION*********************************************************
speaker = ads.Labels.Speaker;
numFolds = numel(speaker);
summary(speaker)
disp('starting cross validation...');

% with data augmentation:
%[labelsTrue,labelsPred, emptyEmotions] = HelperTrainAndValidateNetwork(ads,adsAug,afe);

[labelsTrue,labelsPred, emptyEmotions] = HelperTrainAndValidateNetwork(ads,afe);


for ii = 1:numel(labelsTrue)
    foldAcc = mean(labelsTrue{ii}==labelsPred{ii})*100;
    fprintf('Fold %1.0f, Accuracy = %0.1f\n',ii,foldAcc);
end

labelsTrueMat = cat(1,labelsTrue{:});
labelsPredMat = cat(1,labelsPred{:});
figure
cm = confusionchart(labelsTrueMat,labelsPredMat);
valAccuracy = mean(labelsTrueMat==labelsPredMat)*100;
cm.Title = sprintf('Confusion Matrix for 10-Fold Cross-Validation\nAverage Accuracy = %0.1f',valAccuracy);
sortClasses(cm,categories(emptyEmotions))
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';


% FUNCTIONS****************************************************************

function [trueLabelsCrossFold,predictedLabelsCrossFold, emptyEmotions] = HelperTrainAndValidateNetwork(varargin)
    % Copyright 2019 The MathWorks, Inc.
    if nargin == 3
        ads = varargin{1};
        augads = varargin{2};
        extractor = varargin{3};
    elseif nargin == 2
        ads = varargin{1};
        augads = varargin{1};
        extractor = varargin{2};
    end
    speaker = categories(ads.Labels.Speaker);
    numFolds = numel(speaker);
    emptyEmotions = (ads.Labels.Emotion);
    emptyEmotions(:) = [];

    % Loop over each fold.
    trueLabelsCrossFold = {};
    predictedLabelsCrossFold = {};
    
    for i = 1:numFolds
        
        % 1. Divide the audio datastore into training and validation sets.
        % Convert the data to tall arrays.
        idxTrain           = augads.Labels.Speaker~=speaker(i);
        augadsTrain        = subset(augads,idxTrain);
        augadsTrain.Labels = augadsTrain.Labels.Emotion;
        tallTrain          = tall(augadsTrain);
        idxValidation        = ads.Labels.Speaker==speaker(i);
        adsValidation        = subset(ads,idxValidation);
        adsValidation.Labels = adsValidation.Labels.Emotion;
        tallValidation       = tall(adsValidation);

        % 2. Extract features from the training set. Reorient the features
        % so that time is along rows to be compatible with
        % sequenceInputLayer.
        tallTrain         = cellfun(@(x)x/max(abs(x),[],'all'),tallTrain,"UniformOutput",false);
        tallFeaturesTrain = cellfun(@(x)extract(extractor,x),tallTrain,"UniformOutput",false);
        tallFeaturesTrain = cellfun(@(x)x',tallFeaturesTrain,"UniformOutput",false);  
        featuresTrain = gather(tallFeaturesTrain); % Use evalc to suppress command-line output.
        tallValidation         = cellfun(@(x)x/max(abs(x),[],'all'),tallValidation,"UniformOutput",false);
        tallFeaturesValidation = cellfun(@(x)extract(extractor,x),tallValidation,"UniformOutput",false);
        tallFeaturesValidation = cellfun(@(x)x',tallFeaturesValidation,"UniformOutput",false); 
        featuresValidation = gather(tallFeaturesValidation); % Use evalc to suppress command-line output.

        % 3. Use the training set to determine the mean and standard
        % deviation of each feature. Normalize the training and validation
        % sets.
        allFeatures = cat(2,featuresTrain{:});
        M = mean(allFeatures,2,'omitnan');
        S = std(allFeatures,0,2,'omitnan');
        featuresTrain = cellfun(@(x)(x-M)./S,featuresTrain,'UniformOutput',false);
        for ii = 1:numel(featuresTrain)
            idx = find(isnan(featuresTrain{ii}));
            if ~isempty(idx)
                featuresTrain{ii}(idx) = 0;
            end
        end
        featuresValidation = cellfun(@(x)(x-M)./S,featuresValidation,'UniformOutput',false);
        for ii = 1:numel(featuresValidation)
            idx = find(isnan(featuresValidation{ii}));
            if ~isempty(idx)
                featuresValidation{ii}(idx) = 0;
            end
        end

        % 4. Buffer the sequences so that each sequence consists of twenty
        % feature vectors with overlaps of 10 feature vectors.
        featureVectorsPerSequence = 20;
        featureVectorOverlap = 10;
        [sequencesTrain,sequencePerFileTrain] = HelperFeatureVector2Sequence(featuresTrain,featureVectorsPerSequence,featureVectorOverlap);
        [sequencesValidation,sequencePerFileValidation] = HelperFeatureVector2Sequence(featuresValidation,featureVectorsPerSequence,featureVectorOverlap);

        % 5. Replicate the labels of the train and validation sets so that
        % they are in one-to-one correspondence with the sequences.
        labelsTrain = [emptyEmotions;augadsTrain.Labels];
        labelsTrain = labelsTrain(:);
        labelsTrain = repelem(labelsTrain,[sequencePerFileTrain{:}]);

        % 6. Define a BiLSTM network.
        dropoutProb1 = 0.3;
        numUnits     = 200;
        dropoutProb2 = 0.6;
        layers = [ ...
            sequenceInputLayer(size(sequencesTrain{1},1))
            dropoutLayer(dropoutProb1)
            bilstmLayer(numUnits,"OutputMode","last")
            dropoutLayer(dropoutProb2)
            fullyConnectedLayer(numel(categories(emptyEmotions)))
            softmaxLayer
            classificationLayer];

        % 7. Define training options.
        miniBatchSize       = 512;
        initialLearnRate    = 0.005;
        learnRateDropPeriod = 2;
        maxEpochs           = 4;
        options = trainingOptions("adam", ...
            "MiniBatchSize",miniBatchSize, ...
            "InitialLearnRate",initialLearnRate, ...
            "LearnRateDropPeriod",learnRateDropPeriod, ...
            "LearnRateSchedule","piecewise", ...
            "MaxEpochs",maxEpochs, ...
            "Shuffle","every-epoch", ...
            "Verbose",false);

        % 8. Train the network.
        net = trainNetwork(sequencesTrain,labelsTrain,layers,options);

        % 9. Evaluate the network. Call classify to get the predicted labels
        % for each sequence. Get the mode of the predicted labels of each
        % sequence to get the predicted labels of each file.
        predictedLabelsPerSequence = classify(net,sequencesValidation);
        trueLabels = categorical(adsValidation.Labels);
        predictedLabels = trueLabels;
        idx1 = 1;
        for ii = 1:numel(trueLabels)
            predictedLabels(ii,:) = mode(predictedLabelsPerSequence(idx1:idx1 + sequencePerFileValidation{ii} - 1,:),1);
            idx1 = idx1 + sequencePerFileValidation{ii};
        end
        trueLabelsCrossFold{i} = trueLabels; %#ok<AGROW>
        predictedLabelsCrossFold{i} = predictedLabels; %#ok<AGROW>
    end
end

function [sequences,sequencePerFile] = HelperFeatureVector2Sequence(features,featureVectorsPerSequence,featureVectorOverlap)
    % Copyright 2019 MathWorks, Inc.
    if featureVectorsPerSequence <= featureVectorOverlap
        error('The number of overlapping feature vectors must be less than the number of feature vectors per sequence.')
    end

    if ~iscell(features)
        features = {features};
    end
    hopLength = featureVectorsPerSequence - featureVectorOverlap;
    idx1 = 1;
    sequences = {};
    sequencePerFile = cell(numel(features),1);
    for ii = 1:numel(features)
        sequencePerFile{ii} = floor((size(features{ii},2) - featureVectorsPerSequence)/hopLength) + 1;
        idx2 = 1;
        for j = 1:sequencePerFile{ii}
            sequences{idx1,1} = features{ii}(:,idx2:idx2 + featureVectorsPerSequence - 1); %#ok<AGROW>
            idx1 = idx1 + 1;
            idx2 = idx2 + hopLength;
        end
    end
end
