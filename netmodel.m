function netModel()
X=2;% FOR REAL
Y=5;% FOR FAKE
% Split data into training and testing sets (80% train, 20% test)
cv = cvpartition(size(X, 1), 'Holdout', 0.2);
XTrain = X(cv.training, :);
YTrain = categorical(Y(cv.training));
XTest = X(cv.test, :);
YTest = categorical(Y(cv.test));
% Normalize the features (optional but recommended)
XTrain = normalize(XTrain);
XTest = normalize(XTest)
% Create a neural network
net = patternnet(50); % 50 neurons in the hidden layer
% Train the neural network
net = train(net, XTrain', dummyvar(YTrain)');
%% Save features and labels
save('features_and_labels.mat', 'XTrain', 'YTrain', 'XTest', 'YTest');
% Load saved features and labels
load('features_and_labels.mat');
% Load and preprocess the audio file
audio_file = '*.*.wav'; % Provide the path to your audio file
[y, fs] = audioread(audio_file);
% Extract MFCC features
mfccs = mfcc(y, fs);
% Normalize features (if necessary)
mfccs_normalized = normalize(mfccs);
% Reshape to match input shape of the model
mfccs_reshaped = mfccs_normalized(:)';
% Check the size of mfccs_reshaped
expectedInputSize = size(XTrain, 2);  % Get the input size expected by the model
if numel(mfccs_reshaped) ~= expectedInputSize
    error('Input data size does not match expected input size of the model.');
end
% Make prediction using the trained model
YPred = net(mfccs_reshaped');
% Convert predicted probabilities to classes
YPredClass = vec2ind(YPred);
% Interpret prediction
if YPredClass == 1
    fprintf('The audio is classified as FAKE using Deep learning.\n');
else
    fprintf('The audio is classified as REAL using Deep learning.\n');
end