function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1.0;
sigma = 0.10;
minError = 1; % maximum error (fraction of Xval predicted incorrectly)

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% candidateValues = [0.01 0.03 0.1 0.3 1 3 10 30];
% for sigmaCV = candidateValues
%     for CCV = candidateValues
        
%         % train model using current combination of sigma and C
%         model = svmTrain(X, y, CCV, @(x1, x2) gaussianKernel(x1, x2, sigmaCV));
        
%         % Check model using cross-validation set
%         predictions = svmPredict(model, Xval);
%         predictionsError = mean(double(predictions ~= yval));

%         fprintf('CCV %2.2f\tsigmaCV %2.2f\terror %.4f',CCV, sigmaCV, predictionsError);

%         % if lowest error, save sigma/C values and minError to check against others
%         if predictionsError < minError
%             fprintf('NEW BEST = CCV %2.2f\tsigmaCV %2.2f\terror %.4f',CCV, sigmaCV, predictionsError);
%             C = CCV;
%             sigma = sigmaCV;
%             minError = predictionsError;
%         end

%     endfor
% endfor





% =========================================================================

end
