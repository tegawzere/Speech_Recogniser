NStates = 8;
statesExtended = NStates + 2; 
KFeatures = 13;
DIM=13;
N=NStates;
devFolder = "C:\Users\japau\Documents\MATLAB\hmm coursework\EEEM030cw2_DevelopmentSet";
allFiles = dir(fullfile(devFolder, '*.mp3'));
devFiles = [];
num_of_model=11;
% wordString = {"heed", "hid", "head", "had", "hard", "hud", "hod", "hoard", "hood", "who'd", "heard"};
% wordString = {'heed', 'hid', 'head', 'had', 'hard', 'hud', 'hod', 'hoard', 'hood', 'who'd', "heard"};
wordString = ["heed", "hid", "head", "had", "hard", "hud", "hod", "hoard", "hood", "whod", "heard"];
   
recognitionErrors = 0;
totalObservations = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  FOR FILES NEEDED %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%index for files 
index = 1; 

% Removes both say and again files 
for i = 1:length(allFiles)
    filename = allFiles(i).name;
    % devFiles = [devFiles, allFiles(i)];
    % index = index + 1;
    % Check if the filename does not contain 'say' or 'code'
    if ~contains(filename, 'say') && ~contains(filename, 'again')
        devFiles = [devFiles, allFiles(i)];
        index = index + 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  MEL-FREQUENCY FEATURE EXTRACTION %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Parameters as stated in document 
frameLength = 0.03; 
hopLength = 0.01;
numCoefficients = 13;

allMfccs = []; 
allObservations = {};%This..this might nor do anything but here we are

%Use this for the average duration for A matrix
totalFrameCount = 0;

for i =1: length(devFiles)
    devPath = fullfile(devFolder, devFiles(i).name);

    % Read audio file
    [devData, fs] = audioread(devPath);

    % Convert frame and hop length from time to samples
    frameLengthSamples = round(frameLength * fs);
    hopLengthSamples = round(hopLength * fs);

    % Extract MFCCs (no delta and delta-delta features)
    coefficients = mfcc(devData, fs, 'WindowLength', frameLengthSamples, 'OverlapLength', frameLengthSamples - hopLengthSamples, 'NumCoeffs', numCoefficients-1);
    
    %total frame count gathering 
    totalFrameCount = totalFrameCount + size(coefficients, 1);

    allMfccs = [allMfccs; coefficients];
    allObservations{end+1} = coefficients;
end

%Mean and Variance of the Coeffiecients(this is across all the sound files)
globalMean = mean(allMfccs, 1);
globalVariance = var(allMfccs, 0, 1);


% %Applying a variance floor because I hate the NaN bug 
% varianceFloorScaling = 0.1;%need to ask if this scaling factor is alright
% varianceFloor = globalVariance * varianceFloorScaling;
% globalVariance = globalVariance + varianceFloor;
covarianceMatrix = diag(globalVariance);


% covarianceMatrix = max(covarianceMatrix, diag(varianceFloor));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  A MATRIX AND B MATRIX (should have put the covariance one here too) %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
averageDuration = totalFrameCount / NStates;

%Transition probabilities calculations
selfLoopProbability = exp(-1 / (averageDuration - 1));
transitionToNextStateProbability = 1 - selfLoopProbability;

% Initialize and fill the transition matrix A
A = zeros(statesExtended, statesExtended);
for i = 2:(statesExtended - 1)
    if i < (statesExtended - 1)
        A(i, i) = selfLoopProbability;  % Self-loop
        A(i, i + 1) = transitionToNextStateProbability;  % Transition to next state
    else
        A(i, i) = selfLoopProbability;  % Last state transitions to itself
    end
end

% For the entry and making sure the last state is the transitionProbability
A(1, 2) = 1;  % Entry is 1 
A(statesExtended - 1, statesExtended) = transitionToNextStateProbability;


%B Matrix
% B = repmat(globalMean, statesExtended, 1);
% B = zeros(NStates, KFeatures);
floor= 10^(-10);
B = zeros(statesExtended, size(allMfccs, 2));

% Calculate Gaussian probabilities for each state and observation
for state = 1:statesExtended
    for obs = 1:size(allMfccs, 2)
        B(state, obs) = mvnpdf(allMfccs(obs, :), globalMean, covarianceMatrix);
        B(state, obs) = max(floor,B(state,obs));
    end
end

%Pi
Pi = zeros(1, statesExtended);
Pi(2) = 1;
initPi=Pi;
initB=B;
initA=A;
% Pi=rand(1,statesExtended);
% Pi=Pi/sum(Pi);
% 
% Pi=rand(1,statesExtended-1);
% Pi=[Pi,0,1-sum(Pi)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  TRAINING TIME %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize accumulators
% N = statesExtended;
% accumulatorGamma = zeros(N, KFeatures);
% accumulatorXi = zeros(N, N);
% totalGamma = zeros(N, 1);
num_of_state= NStates;
HMM.mean = zeros(DIM, num_of_state,num_of_model);
HMM.var  = zeros(DIM, num_of_state,num_of_model);
HMM.Aij  = zeros(num_of_state+2, num_of_state+2,num_of_model);
% generate initial HMM
HMM = hmm_init(HMM, devFiles,devFolder, DIM, num_of_state, num_of_model);


num_of_iteration = 15;
log_likelihood_iter = zeros(1, num_of_iteration);
likelihood_iter = zeros(1, num_of_iteration);

countIterations = 15;

likelihoodarr=[];
loglikelihoodarr=[];
jude_likelihoodarr=[];
jude_likelihoodrow=[];
log_likelihood = 0;
likelihood = 0;
for iteration = 1:countIterations
    fprintf("Training Iteration Num is => %d\n", iteration);
    sum_mean_numerator = zeros(DIM, num_of_state,num_of_model);
    sum_var_numerator = zeros(DIM, num_of_state,num_of_model);
    sum_aij_numerator = zeros(num_of_state, num_of_state,num_of_model);
    sum_denominator = zeros(num_of_state,num_of_model);
    
    log_likelihood_i = 0;
    likelihood_i = 0;
    accumulatorGamma = zeros(length(allObservations), KFeatures);
    accumulatorXi = zeros(statesExtended, statesExtended);

    accumulatorM = zeros(statesExtended, KFeatures);
    accumulatorV = zeros(statesExtended, KFeatures);
    totalGamma = zeros(statesExtended, 1);
    % Reset accumulators for each iteration
    % accumulatorGamma(:) = 0;
    % accumulatorXi(:) = 0;
    % totalGamma(:) = 0;
    jude_likelihoodrow=[];

    for idx = 1:length(allObservations)
        observations = allObservations{idx};

        ground_word=getTrainGroundTruthWord(devFiles(idx),wordString);
                % Original word in single quotes
        % wordInSingleQuotes = 'word';
        
        % Remove single quotes and add double quotes
        % ground_word = strcat('"', strrep(ground_word, '', ''' ), '"');
        
        % Display the result
        % disp(wordInDoubleQuotes);
        % charwordstring=char(wordString);
        k = find(strcmpi(wordString, string(ground_word)));

        % k=getClassLabel(idx);
        % T = size(observations, 1); % Number of observations
        % accumulatorGamma = zeros(T, KFeatures);
        obs=observations';

        meang=HMM.mean(:,:,k);
        varg= HMM.var(:,:,k);
        aij= HMM.Aij(:,:,k);
        [dim,T] = size(obs); 

        meang = [NaN(dim,1) meang NaN(dim,1)];
        varg = [NaN(dim,1) varg NaN(dim,1)];
        dim=13;
   
        aij(end,end) = 1;
        N=10;
        log_alpha = -Inf(N, T+1); 
        log_beta = -Inf(N, T+1);  
        
        for i = 1:N 
            log_alpha(i,1) = log(aij(1,i)) + logGaussian(meang(:,i),varg(:,i),obs(:,1)); 
        end
        
        for t = 2:T 
            for j = 2:N-1    
                log_alpha(j,t) = log_sum_alpha(log_alpha(2:N-1,t-1),aij(2:N-1,j)) + logGaussian(meang(:,j),varg(:,j),obs(:,t));
            end
        end
        
        log_alpha(N,T+1) = log_sum_alpha(log_alpha(2:N-1,T),aij(2:N-1,N)); 
        
        log_beta(:,T) = log(aij(:,N));
        for t = (T-1):-1:1 
            for i = 2:N-1
                log_beta(i,t) = log_sum_beta(aij(i,2:N-1),meang(:,2:N-1),varg(:,2:N-1),obs(:,t+1),log_beta(2:N-1,t+1));
            end
        end
        log_beta(N,1) = log_sum_beta(aij(1,2:N-1),meang(:,2:N-1),varg(:,2:N-1),obs(:,1),log_beta(2:N-1,1));
        
        log_Xi = -Inf(N,N,T);
        for t = 1:T-1
            for j = 2:N-1
                for i = 2:N-1
                    log_Xi(i,j,t) = log_alpha(i,t) + log(aij(i,j)) + logGaussian(meang(:,j),varg(:,j),obs(:,t+1)) + log_beta(j,t+1) - log_alpha(N,T+1);
                end
            end
        end
        for i = 1:N
            log_Xi(i,N,T) = log_alpha(i,T) + log(aij(i,N)) - log_alpha(N, T+1);
        end

        log_gamma = -inf(N,T);
        for t = 1:T
            for i = 2:N-1
                log_gamma(i,t) = log_alpha(i,t) + log_beta(i,t) - log_alpha(N,T+1);
            end
        end
        gamma = exp(log_gamma);
        
        mean_numerator = zeros(dim,N); 
        var_numerator = zeros(dim,N);
        denominator = zeros(N,1);
        aij_numerator = zeros(N,N);
        for j = 2:N-1
            for t = 1:T
                mean_numerator(:,j) = mean_numerator(:,j) + gamma(j,t)*obs(:,t);
                var_numerator(:,j) = var_numerator(:,j)+ gamma(j,t)*(obs(:,t)).*(obs(:,t));
                denominator(j) = denominator(j) + gamma(j,t);
            end  
        end
        for i = 2:N-1
            for j = 2:N-1
                for t = 1:T
                    aij_numerator(i,j) = aij_numerator(i,j) + exp(log_Xi(i,j,t));
                end
            end
        end
        log_likelihood_i = log_alpha(N,T+1);
        likelihood_i = exp(log_alpha(N,T+1));
        jude_likelihood=log_likelihood_i*-1*10^-4;
        % loglikelihoodarr=[loglikelihoodarr  log_likelihood_i];
        % likelihoodarr=[likelihoodarr likelihood_i];
        % jude_likelihoodrow=[jude_likelihoodrow jude_likelihood];
        
    % end
    % jude_likelihoodarr=[jude_likelihoodarr; jude_likelihoodrow];
    sum_mean_numerator(:,:,k) = sum_mean_numerator(:,:,k) + mean_numerator(:,2:end-1);
    sum_var_numerator(:,:,k) = sum_var_numerator(:,:,k) + var_numerator(:,2:end-1);
    sum_aij_numerator(:,:,k) = sum_aij_numerator(:,:,k) + aij_numerator(2:end-1,2:end-1);
    sum_denominator(:,k) = sum_denominator(:,k) + denominator(2:end-1);

    log_likelihood = log_likelihood + log_likelihood_i;
    likelihood = likelihood + likelihood_i;
    end
    for k=1:num_of_model
        for n = 1:num_of_state
            HMM.mean(:,n,k) = sum_mean_numerator(:,n,k) / sum_denominator (n,k);
            HMM.var (:,n,k) = sum_var_numerator(:,n,k) / sum_denominator (n,k) -  HMM.mean(:,n,k).* HMM.mean(:,n,k);
        end
    end
    for k=1:num_of_model
        for i = 2:num_of_state+1
            for j = 2:num_of_state+1
                HMM.Aij (i,j,k) = sum_aij_numerator(i-1,j-1,k) / sum_denominator (i-1,k);
            end
        end
    
    HMM.Aij (num_of_state+1,num_of_state+2,k) = 1 - HMM.Aij (num_of_state+1,num_of_state+1,k);

    HMM.Aij (num_of_state+2,num_of_state+2,k) = 1;
    log_likelihood_iter(iteration) = log_likelihood;
    likelihood_iter(iteration) = likelihood;
    end
end
figure();
plot(log_likelihood_iter,'-*');
xlabel('iterations'); ylabel('log likelihood');
title(['number of states: ', num2str(num_of_state)]);

save HMM_models_vit_test;

num_of_model = 11;
num_of_error = 0;
num_of_testing = 0;

% load (testing_file_list, 'testingfile');
num_of_test = 55;
% num_of_test=11;
testFolder = "C:\Users\japau\Documents\MATLAB\hmm coursework\EEEM030cw2_EvaluationSet";
% testFolder="C:\Users\japau\Documents\MATLAB\hmm coursework\tega_sound_clips";
alltestFiles = dir(fullfile(testFolder, '*.mp3'));
testFiles = [];

% wordString = {"heed", "hid", "head", "had", "hard", "hud", "hod", "hoard", "hood", "who'd", "heard"};
% wordString = {"heed", "hid", "head", "had", "hard", "hud", "hod", "hoard", "hood", "whod", "heard"};    
wordString = ["heed", "hid", "head", "had", "hard", "hud", "hod", "hoard", "hood", "whod", "heard"];    

recognitionErrors = 0;
totalObservations = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  FOR FILES NEEDED %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%index for files 
index = 1; 

% Removes both say and again files 
for i = 1:length(alltestFiles)
    filename = alltestFiles(i).name;
    % devFiles = [devFiles, allFiles(i)];
    % index = index + 1;
    % Check if the filename does not contain 'say' or 'code'
    if ~contains(filename, 'say') && ~contains(filename, 'again')
        testFiles = [testFiles, alltestFiles(i)];
        index = index + 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  MEL-FREQUENCY FEATURE EXTRACTION %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Parameters as stated in document 
frameLength = 0.03; 
hopLength = 0.01;
numCoefficients = 13;

alltestMfccs = []; 
alltestObservations = {};%This..this might nor do anything but here we are

%Use this for the average duration for A matrix
totalFrameCount = 0;

for i =1: length(testFiles)
    testPath = fullfile(testFolder, testFiles(i).name);

    % Read audio file
    [testData, fs] = audioread(testPath);

    % Convert frame and hop length from time to samples
    frameLengthSamples = round(frameLength * fs);
    hopLengthSamples = round(hopLength * fs);

    % Extract MFCCs (no delta and delta-delta features)
    testcoefficients = mfcc(testData, fs, 'WindowLength', frameLengthSamples, 'OverlapLength', frameLengthSamples - hopLengthSamples, 'NumCoeffs', numCoefficients-1);
    
    totalFrameCount = totalFrameCount + size(testcoefficients, 1);

    alltestMfccs = [alltestMfccs; testcoefficients];
    alltestObservations{end+1} = testcoefficients;
end
for u=1:num_of_test
    
end
confusionm=zeros(11,11); % 
target=[];
output=[];
for u = 1:num_of_test

    features=alltestObservations{u};
    num_of_testing = num_of_testing + 1;
    fopt_max = -Inf; word = 'unknown';
    for p=1:num_of_model
        fopt = viterbi_dist_FR(HMM.mean(:,:,p), HMM.var(:,:,p), HMM.Aij(:,:,p), features); 

        if fopt > fopt_max
            word = wordString{p};
            fopt_max = fopt;
        end
    end
    % 
    % [maxLikelihood, maxIndex] = max(f);
    % fprintf('Maximum Likelihood: %f\n', maxLikelihood);
    % fprintf('Corresponding Index: %d\n', maxIndex);
    % fprintf("fopt %d ",fopt);
    % fprintf("word %s ",word);
    groundTruth_word=getGroundTruthWord(testFiles(u),wordString);
    groundTruth_word=string(groundTruth_word);
    if strcmp(word, groundTruth_word) % testing
        num_of_error = num_of_error + 1;
    end
    wi=find(strcmpi(wordString,string(word)));
    wu=find(strcmpi(wordString,string(groundTruth_word)));
    target=[target wu];
    output=[output wi];

    % end
    
end
accuracy_rate = (num_of_testing - num_of_error)*100/num_of_testing;
figure;
confusionchart(output,target);

function log_b = logGaussian (mean_i, var_i, o_i)
dim = length(var_i);
log_b = -1/2*(dim*log(2*pi) + sum(log(var_i)) + sum((o_i - mean_i).*(o_i - mean_i)./var_i));
end

function logsumalpha = log_sum_alpha(log_alpha_t,aij_j)
len_x = size(log_alpha_t,1);
y = -Inf(1,len_x);
ymax = -Inf;
for i = 1:len_x
    y(i) = log_alpha_t(i) + log(aij_j(i));
    if y(i) > ymax
        ymax = y(i);
    end
end
if ymax == Inf
    logsumalpha = Inf;
else
    sum_exp = 0;
    for i = 1:len_x
        if ymax == -Inf && y(i) == -Inf
            sum_exp = sum_exp + 1;
        else
            sum_exp = sum_exp + exp(y(i) - ymax);
        end
    end
    logsumalpha = ymax + log(sum_exp);
end
end

function logsumbeta = log_sum_beta(aij_i,mean,var,obs,beta_t1)
len_x = size(mean,2);
y = -Inf(1,len_x);
ymax = -Inf;
for j = 1:len_x
    y(j) = log(aij_i(j)) + logGaussian(mean(:,j),var(:,j),obs) + beta_t1(j);
    if y(j) > ymax
        ymax = y(j);
    end
end
if ymax == Inf
    logsumbeta = Inf;
else
    sum_exp = 0;
    for i = 1:len_x
        if ymax == -Inf && y(i) == -Inf
            sum_exp = sum_exp + 1;
        else
            sum_exp = sum_exp + exp(y(i) - ymax);
        end
    end
    logsumbeta = ymax + log(sum_exp);
end
end


function [fopt,optimal_chain,f,optimal_index] = viterbi_dist_FR(meang, varg, aij, obs)
obs=obs';
[dim, t_len] = size(obs);

% Define the floor value
floorValue = 10e-10;

% Replace 0 values with the floor value
% varg = varg;
varg(varg == 0) = floorValue;

% fprintf("dim %d", dim);
% fprintf("meang %d", meang);
% fprintf("varg %d", varg);
aij(end,end) = 1;
timing = 1:t_len+1;
matrix_size = size(meang, 2);
% disp(matrix_size);
vit_matrix = -Inf(matrix_size, t_len);
vit_chain = cell(matrix_size, t_len);

delta_time = timing(1);
for j=2:matrix_size-1 % 2->14
    % fprintf("1 obs %d %d \n",obs(:,1),size(obs(:,1)));
    % fprintf("1 meag %d %d \n",meang(:,j),size(meang(:,j)));
    % fprintf("1 varg %d %d \n",varg(:,j),size(varg(:,j)));
    % disp(j);
    % disp(size(obs(:,1)));
    % disp(size(varg(:,j)));
    % disp(size(meang(:,j)));
    % fprintf("1 obs %d \n",size(obs(:,1)));
    % fprintf("1 obs %d \n",(obs));
    % fprintf("1 aij %d \n",(aij));
    % 
    % fprintf("1 meag %d \n",size(meang(:,j)));
    % fprintf("1 varg %d \n",size(varg(:,j)));
    log_b=logGaussian_v(meang(:,j),varg(:,j),obs(:,1));
    vit_matrix(j,1) = log(aij(1,j)) + log_b;
    if vit_matrix(j,1) > -Inf
        vit_chain{j,1} = [1 j];
    end
end

for t=2:t_len
    delta_time = timing(t)-timing(t-1);
    for j=2:matrix_size-1 
        f_max = -Inf;
        index_max = -1;
        f = -Inf;
        for i=2:j
            if(vit_matrix(i,t-1) > -Inf)
                % fprintf("obs %d",obs(:,t));
                % fprintf("obs %d",meang(:,j));
                % fprintf("obs %d",varg(:,j));
                f = vit_matrix(i,t-1) + log(aij(i,j)) + logGaussian_v(meang(:,j),varg(:,j),obs(:,t));
            end
            if f > f_max 
                f_max = f;
                index_max = i;
            end
        end
        if index_max ~= -1
            vit_chain{j,t} = [vit_chain{index_max,t-1} j];
            vit_matrix(j,t) = f_max;
        end
    end
end
delta_time = timing(end) - timing(end - 1);
fopt = -Inf;
optimal_index = -1;
for i=2:matrix_size-1
    f = vit_matrix(i, t_len) + log(aij(i, matrix_size));
    if f > fopt
        fopt = f;
        optimal_index = i;
    end
end

if optimal_index ~=-1
    optimal_chain = [vit_chain{optimal_index,t} matrix_size];
end
end
%         % Forward procedure with covariance matrix
%         % [logAlpha, scalingFactor] = forwardProcedure(observations, A, B, Pi, covarianceMatrix);
%         [logAlpha] = new_forward_1(observations, A, B, Pi);
% 
%         % Backward procedure with covariance matrix
%         % logBeta = backwardProcedure(observations, A, B, covarianceMatrix, scalingFactor);
%         [logBeta] = new_backward_1(observations, A, B);
%         % logAlpha=logAlpha*10^9;
%         % logBeta=logBeta*10^9;
%         % Calculate occupation and transition likelihoods with covariance matrix
%         [gamma, xi,accumulatorGamma] = new_likelihood(logAlpha, logBeta, A, B, observations,accumulatorGamma);
% 
%         % [accumulatorGamma,accumulatorXi,totalGamma]=accumulator(gamma,xi);
%         % accumulatorGamma = accumulatorGamma + sum(gamma, 1); % Sum over time
%         accumulatorXi = accumulatorXi + sum(xi, 1); % Sum over time and states
%         % totalGamma = totalGamma + sum(gamma, 2); % Sum over states
%         % accumulatorGamma=accumulatorGamma + sum(gamma(idx,:));
%         % accumulatorGamma = accumulatorGamma + repmat(sum(gamma, 1)', 1, KFeatures); % Sum over time and replicate for each feature
%         % accumulatorXi = accumulatorXi + squeeze(sum(xi, 1)); % Sum over time and states
%         totalGamma = totalGamma + sum(gamma, 1)'; % Sum over states
%         % accumulatorM=accumulatorM + gamma(1,idx)*observations;
%         % accumulatorV=accumulatorV + gamma(1,idx)*(observations-globalMean(1,idx))*(observations-globalMean(1,idx))';
%     end
% 
% 
%     % Re-estimate A, B, and Pi
%     for i = 1:statesExtended
%         for j = 1:statesExtended
%             if totalGamma(i)==0
%                 A(i,j)=0;
%             else
%                 A(i, j) = sum(accumulatorXi(:, i, j)) / totalGamma(i);
%             % B(i, j) = sum(accumulatorGamma(:,j) )/ totalGamma(i);
%             end
% 
%         end
%     end
% 
%     % for j = 1:statesExtended
%     %     B(j, :) = accumulatorGamma(j, :) / totalGamma(j);
%     % end
%     B_new = zeros(statesExtended, KFeatures);
% 
%     for i=1:statesExtended
%         for j=1:KFeatures
%             expected_count=sum(accumulatorGamma(i,:));
%                         % Normalize the expected count by the total gamma for state i
%             if totalGamma(i) > 0
%                 B_new(i, j) = expected_count / totalGamma(i);
%             else
%                 % Handle the case where totalgamma(i) is zero (to avoid division by zero)
%                 B_new(i, j) = 0;
%             end
%         end
%     end
%     B=B_new;
%     Pi = gamma(1, :); % Initial state probabilities
% end

function log_b = logGaussian_v (mean_i, var_i, o_i)
dim = length(var_i);
log_b = -1/2*(dim*log(2*pi) + sum(log(var_i)) + sum((o_i - mean_i).*(o_i - mean_i)./var_i));
end

function groundTruthWord = getTrainGroundTruthWord(audioFile, wordString)

    audioFile=audioFile.name;
    % Check if the filename contains an underscore followed by a word in wordString
    underscoreIndex = strfind(audioFile, '_');
    underscore_parts=strsplit(audioFile,'_');
    
    wordAfterUnderscore=underscore_parts{3};
    dotIndex = strfind(wordAfterUnderscore, '.');
    if ~isempty(dotIndex)
        wordAfterUnderscore = wordAfterUnderscore(1:dotIndex-1);
    end
    groundTruthWord=wordAfterUnderscore;
  
end
function groundTruthWord = getGroundTruthWord(audioFile, wordString)

    audioFile=audioFile.name;
    underscoreIndex = strfind(audioFile, '_');
    underscore_parts=strsplit(audioFile,'_');

    wordAfterUnderscore=underscore_parts{2};
    dotIndex = strfind(wordAfterUnderscore, '.');
    if ~isempty(dotIndex)
        wordAfterUnderscore = wordAfterUnderscore(1:dotIndex-1);
    end
    groundTruthWord=wordAfterUnderscore;
  
end
function label_id=  getClassLabel(idx)

end

 

function HMM = hmm_init(HMM,devFiles,devFolder, DIM, num_of_state, num_of_model)
sum_of_features = zeros(DIM,1);
sum_of_features_square = zeros(DIM, 1);
num_of_feature = 0;
frameLength = 0.03; 
hopLength = 0.01;
numCoefficients = 13;

for i =1: length(devFiles)
    devPath = fullfile(devFolder, devFiles(i).name);

    % Read audio file
    [devData, fs] = audioread(devPath);

    % Convert frame and hop length from time to samples
    frameLengthSamples = round(frameLength * fs);
    hopLengthSamples = round(hopLength * fs);

    % Extract MFCCs (no delta and delta-delta features)
    coefficients = mfcc(devData, fs, 'WindowLength', frameLengthSamples, 'OverlapLength', frameLengthSamples - hopLengthSamples, 'NumCoeffs', numCoefficients-1);
    features = coefficients';
    
    sum_of_features = sum_of_features + sum(features, 2); 
    sum_of_features_square = sum_of_features_square + sum(features.^2, 2);
    num_of_feature = num_of_feature + size(features,2); 

end

HMM = hmm_init_components(HMM, num_of_state, num_of_model, sum_of_features, sum_of_features_square, num_of_feature);
end



function HMM = hmm_init_components(HMM, num_of_state, num_of_model, sum_of_features, sum_of_features_square, num_of_feature)

for k = 1:num_of_model
    for m = 1:num_of_state
        HMM.mean(:,m,k) = sum_of_features/num_of_feature;
        HMM.var(:,m,k) = sum_of_features_square/num_of_feature - HMM.mean(:,m,k).*HMM.mean(:,m,k);
    end
    for i = 2:num_of_state+1
        HMM.Aij(i,i+1,k) = 0.28;
        HMM.Aij(i,i,k) = 1-HMM.Aij(i,i+1,k);
        
    end
    HMM.Aij(1,2,k) = 1;
end
end
