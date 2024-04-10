%% Classification of real and fake voice using DEEP LEARNING Technique
% Load the dataset
data = readtable('DATASET-balanced.csv');
% Extract features and labels
X = table2array(data(:, 1:end-1)); % Extract features
Y = data{:, end};                  % Extract labels
% Split data into training and testing sets (80% train, 20% test)
cv = cvpartition(size(X, 1), 'Holdout', 0.2);
XTrain = X(cv.training, :);
YTrain = categorical(Y(cv.training));
XTest = X(cv.test, :);
YTest = categorical(Y(cv.test));
% Normalize the features (optional but recommended)
XTrain = normalize(XTrain);
XTest = normalize(XTest);
% Create a neural network
net = patternnet(50); % 50 neurons in the hidden layer
% Train the neural network
net = train(net, XTrain', dummyvar(YTrain)');
%% test for prediction of voice signal
% Load the dataset
data = readtable('DATASET-balanced.csv');
% Extract features and labels
X = table2array(data(:, 1:end-1)); % Extract features
Y = data{:, end};                  % Extract labels
% Split data into training and testing sets (80% train, 20% test)
cv = cvpartition(size(X, 1), 'Holdout', 0.2);
XTrain = X(cv.training, :);
YTrain = categorical(Y(cv.training));
XTest = X(cv.test, :);
YTest = categorical(Y(cv.test));

% Normalize the features (optional but recommended)
XTrain = normalize(XTrain);
XTest = normalize(XTest);
% Load the features and labels
load('features_and_labels.mat');
% Call the deepnetmodel function
netmodel();
% Load and assign the audio file from net model of the neural network
% Feature extraction (MFCCs)
numCoeffs = 13;
frameDuration = 0.02;
hopDuration = 0.01; % Duration of the hop size in seconds
% Make predictions on the test set
YPred = net(XTest');
% Convert predicted probabilities to classes
YPredClass = vec2ind(YPred);
% Convert YTest to numerical array
YTestNumeric = double(YTest);
% Evaluate the performance
accuracy = sum(YPredClass == YTestNumeric) / numel(YTestNumeric);
disp(['Accuracy: ', num2str(accuracy)]);
% Compute confusion matrix
C = confusionmat(YTestNumeric, YPredClass);
% Compute precision, recall, and F1-score
precision = diag(C) ./ sum(C, 1)';
recall = diag(C) ./ sum(C, 2);
f1_score = 2 * (precision .* recall) ./ (precision + recall);
% Compute overall accuracy
overall_accuracy = sum(diag(C)) / sum(C, 'all');
% Display results
disp('Confusion Matrix:');
disp(C);
% Plot confusion matrix
figure;
confusionchart(C);
title('Confusion Matrix');
disp('Precision:');
disp(array2table(precision', 'VariableNames', {'FAKE', 'REAL'}));
disp('Recall:');
disp(array2table(recall, 'VariableNames', {'Recall'})); % Change 'VariableNames'
disp('F1-score:');
disp(array2table(f1_score', 'VariableNames', {'FAKE', 'REAL'}));
disp(['Overall Accuracy: ', num2str(overall_accuracy)]);