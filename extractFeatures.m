function features = extractFeatures(audioFile)
    [audioData, sampleRate] = audioread(audioFile);

    % Extract MFCC features
    mfccFeatures = mfcc(audioData, sampleRate);

    % Compute RMS (Root Mean Square) value
    rmsValue = rms(audioData);

    % If MFCC features have multiple frames, take the mean across frames
    if size(mfccFeatures, 2) > 1
        mfccFeatures = mean(mfccFeatures, 2);
    end

    % Ensure that RMS value is a scalar
    if numel(rmsValue) > 1
        rmsValue = mean(rmsValue);
    end

    % Reshape RMS value to match the dimension of MFCC features
    rmsValue = repmat(rmsValue, size(mfccFeatures, 1), 1);

    % Concatenate features into a single vector
    features = [mfccFeatures; rmsValue];
end
