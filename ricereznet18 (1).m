tic;
imds = imageDatastore('Rice_Image_Dataset','IncludeSubfolders',true,'LabelSource','foldernames');

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

net = resnet18;
inputSize = net.Layers(1).InputSize;
numClasses = numel(categories(imdsTrain.Labels));

% Create a layer graph from the network
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});

newLayers = [
    fullyConnectedLayer(numClasses, 'Name','FC', 'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer('Name', 'Real')
    classificationLayer('Name', 'Atletico')];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'FC');

% Data augmentation setup
pixelRange = [-40 40];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, 'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);


bestAccuracy = 0;

            fprintf('Training with LR: %f, BatchSize: %d, Epochs: %d\n', lr, batchSize, epoch);
            options = trainingOptions('adam', ...
                'MiniBatchSize', 64, ...
                'MaxEpochs', 6, ...
                'InitialLearnRate', 0,000100, ...
                'Shuffle', 'every-epoch', ...
                'ValidationData', augimdsValidation, ...
                'ValidationFrequency', 3, ...
                'Verbose', false, ...
                'Plots', 'none');
            
            netTransfer = trainNetwork(augimdsTrain, lgraph, options);
            [YPred, ~] = classify(netTransfer, augimdsValidation);
            YValidation = imdsValidation.Labels;
            acc = mean(YPred == YValidation);
            
            fprintf('Accuracy: %.2f%%\n', acc * 100);
            

toc;
