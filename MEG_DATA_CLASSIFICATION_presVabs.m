%%% MEG DATA CLASSIFICATION

trigData_sz = size(trigData); % (time points (ms) x channels x trials)
maxTime = trigData_sz(1); % ms in trial
nChannels = trigData_sz(2); % MEG channels
nTrials = trigData_sz(3); % number of trials
nCond = 4; % number of conditions

responseData = behav.responseData_all(:,13:14); % stimulus presence & orientation
responseLabels = behav.responseData_labels; % data labels

%% DATA ORGANIZATION

% logic index vectors, giving trial locations in the complete data set 
t1p_t2aTrials = ~isnan(responseData(:,1)) & isnan(responseData(:,2)); % cond1 = T1 pres T2 abs
t1a_t2pTrials = isnan(responseData(:,1)) & ~isnan(responseData(:,2)); % cond2 = T1 abs T2 pres
t1p_t2pTrials = ~isnan(responseData(:,1)) & ~isnan(responseData(:,2)); % cond3 = T1 pres T2 pres
t1a_t2aTrials = isnan(responseData(:,1)) & isnan(responseData(:,2)) & behav.responseData_all(:,4) ~= 1; % cond4 = T1 abs T2 abs

% extract data for each condition
t1p_t2aData = trigData(:,:, t1p_t2aTrials==1); % data from cond1
t1a_t2pData = trigData(:,:, t1a_t2pTrials==1); % "" cond2
t1p_t2pData = trigData(:,:, t1p_t2pTrials==1); % "" cond3
t1a_t2aData = trigData(:,:, t1a_t2aTrials==1); % "" cond4

nt1p_t2aTrials = size(t1p_t2aData,3);
nt1a_t2pTrials = size(t1a_t2pData,3);
nt1p_t2pTrials = size(t1p_t2pData,3);
nt1a_t2aTrials = size(t1a_t2aData,3);

%% Trial Labels

% stimulus present or absent labels
t1p_t2aLabels = t1p_t2aTrials(t1p_t2aTrials == 1); 
t1a_t2pLabels= t1a_t2pTrials(t1a_t2pTrials == 1); 
t1p_t2pLabels = t1p_t2pTrials(t1p_t2pTrials == 1);
t1a_t2aLabels = zeros(length(t1a_t2aTrials(t1a_t2aTrials == 1)),1);

% convert trial label from logical -> num -> character
t1p_t2aTrialLabel_num = double(t1p_t2aLabels);
t1p_t2aTrialLabel_char = char(zeros(size(t1p_t2aTrialLabel_num)));
t1p_t2aTrialLabel_char(t1p_t2aTrialLabel_num==0) = 'A'; t1p_t2aTrialLabel_char(t1p_t2aTrialLabel_num==1) = 'P';

t1a_t2pTrialLabel_num = double(t1a_t2pLabels);
t1a_t2pTrialLabel_char = char(zeros(size(t1a_t2pTrialLabel_num)));
t1a_t2pTrialLabel_char(t1a_t2pTrialLabel_num==0)='A'; t1a_t2pTrialLabel_char(t1a_t2pTrialLabel_num==1) = 'P';

t1p_t2pTrialLabel_num = double(t1p_t2pLabels);
t1p_t2pTrialLabel_char = char(zeros(size(t1p_t2pTrialLabel_num)));
t1p_t2pTrialLabel_char(t1p_t2pTrialLabel_num==0)='A'; t1p_t2pTrialLabel_char(t1p_t2pTrialLabel_num==1) = 'P';

t1a_t2aTrialLabel_num = double(t1a_t2aLabels);
t1a_t2aTrialLabel_char = char(zeros(size(t1a_t2aTrialLabel_num)));
t1a_t2aTrialLabel_char(t1a_t2aTrialLabel_num==0)='A'; t1a_t2aTrialLabel_char(t1a_t2aTrialLabel_num==1) = 'P';

%% empty matrices to place accuracy scores for each time point from trials

% [svm crossval]
t1p_t2aAccScores = nan(maxTime,2);
t1a_t2pAccScores = nan(maxTime,2);
t1p_t2pAccScores = nan(maxTime,2);

% channelsRanked

%% CLASSIFICATION & ACCURACY %%
disp('Condition 1 Classification Time')
tic
    %%% CONDITION 1 %%%
for i=1:20:maxTime 
    tic
    % CLASSIFICATION %
    % SVM %
    X = [squeeze(t1p_t2aData(i,channelsRanked(1:10),:))'; squeeze(t1a_t2aData(i,channelsRanked(1:10),:))'];
    Y = [t1p_t2aTrialLabel_char; t1a_t2aTrialLabel_char];
    svm_t1pt2a = fitcsvm(X,Y);
    
    % Cross Validation %
    crossval_t1pt2a = crossval(svm_t1pt2a,'KFold',5);
    
    % ACCURACY %
    % SVM Accuracy %
    [label, scores] = predict(svm_t1pt2a, X);

    % Cross Validation Accuracy
    t1p_t2aAccScores(i,2) = 1-crossval_t1pt2a.kfoldLoss;
    toc
end
eltime = toc;
eltime = eltime/60;
disp(['Elapsed time is ' num2str(eltime) ' minute(s).'])
%%
disp('Condition 2 Classification Time')
tic
    %%% CONDITION 2 %%%
for i=1:10:maxTime
    % CLASSIFICATION %
    % SVM %
    X = squeeze(t1a_t2pData(i,channelsRanked(1:10),:))';
    Y = t1a_t2aTrialLabel_char';
    svm_t1at2p= fitcsvm(X,Y);
    
    % Cross Validation %
    crossval_t1at2p = crossval(svm_t1at2p,'KFold',5);
    
    % ACCURACY %    
    % Cross Validation Accuracy 
    t1a_t2pAccScores(i,2) = 1-crossval_t1at2p.kfoldLoss;
end
eltime = toc;
eltime = eltime/60;
disp(['Elapsed time is ' num2str(eltime) ' minute(s).'])
%%
disp('Condition 3 Classification Time')
tic
    %%% CONDITION 3 %%%
for i=1:10:maxTime 
    % CLASSIFICATION %
    % SVM %
    X = squeeze(t1p_t2pData(i,channelsRanked(1:10),1:nt1p_t2pTrials))';
    Y = t1a_t2aTrialLabel_char';
    svm_t1pt2p= fitcsvm(X,Y);
    
    % Cross Validation %
    crossval_t1pt2p = crossval(svm_t1pt2p,'KFold',5);
    
    % ACCURACY %
    
    % Cross Validation Accuracy
    t1p_t2pAccScores(i,2) = 1-crossval_t1pt2p.kfoldLoss;
end
eltime = toc;
eltime = eltime/60;
disp(['Elapsed time is ' num2str(eltime) ' minute(s).'])
%% PLOTTING FIGURES
