clear ; close all; clc

s = RandStream.create('mt19937ar','seed',490);
RandStream.setGlobalStream(s);

fprintf('Loading and Visualizing Data ...\n')
load('ex3data1.mat');
yMat = getYMat(y);
m = size(X, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%% Select Training and Test Data %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
fprintf('Selecting training and testing data set ...\n')

rp = randperm(m);

xTrain = X(rp(1, 1:(0.8 * m)), :);
yTrain = y(rp(1, 1:(0.8 * m)), :);

xTest = X(rp(1, (0.8 * m)+1: m), :);
yTest = y(rp(1, (0.8 * m)+1: m), :); 

fprintf('Program paused. Press enter to continue.\n');
%pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Running PCA ...\n')

[X_train_red, eigenVect] = pca(xTrain, 0.1);
xTest_red = xTest * eigenVect;
fprintf('\nReduced to dimension %d.\n', size(X_train_red, 2) );


fprintf('\nProgram paused. Press enter to continue.\n');
%pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Linear Support Vector Machine %%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%By default, fitcecoc uses K(K ? 1)/2 binary support vector machine (SVM) models 
% using the one-versus-one coding design, where K is the number of unique class 
% labels (levels). Mdl is a ClassificationECOC model.

% we have 45 svms.

t = templateSVM('KernelFunction','linear', 'KernelScale','auto');
Mdl = fitcecoc(X_train_red,yTrain, 'Coding','onevsall', 'Learners', t)

SVMTrainingPred = predict(Mdl, X_train_red);
SVMTestPred = predict(Mdl, xTest_red);

fprintf('\nSVM Training Set Accuracy: %f\n', mean(double(SVMTrainingPred == yTrain)) * 100);
fprintf('\nSVM Test Set Accuracy: %f\n', mean(double(SVMTestPred == yTest)) * 100);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Gaussian Kernel Support Vector Machine %%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%By default, fitcecoc uses K(K ? 1)/2 binary support vector machine (SVM) models 
% using the one-versus-one coding design, where K is the number of unique class 
% labels (levels). Mdl is a ClassificationECOC model.

% we have 45 svms.

% gammaValues = [1,2,2.3, 2.6, 3];
% trainingAccuracies = zeros(size(gammaValues));
% testAccuracies = zeros(size(gammaValues));
% 
% for i = 1:size(gammaValues, 2)
% 
%     gamma = gammaValues(i);
%     t = templateSVM('KernelFunction','gaussian', 'KernelScale',gamma);
%     Mdl = fitcecoc(X_train_red,yTrain, 'Coding','onevsall', 'Learners', t)
% 
%     SVMTrainingPred = predict(Mdl, X_train_red);
%     SVMTestPred = predict(Mdl, xTest_red);
%     trainingAccuracies(i) = mean(double(SVMTrainingPred == yTrain)) * 100;
%     testAccuracies(i) = mean(double(SVMTestPred == yTest)) * 100;
%     
%     fprintf('\nGamma: %f\n', gamma);
%     fprintf('\nSVM Training Set Accuracy: %f\n', mean(double(SVMTrainingPred == yTrain)) * 100);
%     fprintf('\nSVM Test Set Accuracy: %f\n', mean(double(SVMTestPred == yTest)) * 100);
% end
% 
% plot(gammaValues,trainingAccuracies,'color','r'); hold on;
% plot(gammaValues,testAccuracies,'color','b');
% 
% pause;
% hold off;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Show Predictions %%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rp = randperm(size(xTest, 1));
for i = 1:size(rp, 2)
    % Display 
    dataNumber = rp(i);
    fprintf('\nDisplaying Example Image\n');
    displayData(xTest(dataNumber, :));

    xTest_red = xTest(dataNumber,:) * eigenVect;
    [label] = predict(Mdl, xTest_red);
    
    fprintf('\nSVM Prediction: %d (digit %d)\n', label , mod(yTest(rp(i)), 10));
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
    
end


