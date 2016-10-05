%%% Classification on fake MEG data set for a single time point

%%% Generate random MEG data: single time point, 2 channels (to 100 channels
%%% total), 2 conditions/classes (vertical(V)/horiztonal(H)), 100 trials per condition (200
%%% trials total. (mean and SD for each condition can overlap) 
%%% Classifier distinguishes condition for each trial

%Parameters
nCond = 2; %number of conditions
nChan = 2; %number of channels
nTrialsPerCond = 100; %number of trials per condition
nTrials = nTrialsPerCond*nCond; %total number of trials

a = 1;b=2; %mean and sd interval
mu_mat = a + (a-b)*randn(nCond,nChan); % mean matrix; row = condition, col = channel
sd_mat = a + ((a-b)*randn(nCond,nChan))./10; %standard deviation matrix; row = condition, col = channel
respVarName = [repmat('H', nTrialsPerCond,1); repmat('V', nTrialsPerCond,1)]; %condition label matrix: horizontal/vertical

%Generate data
data_mat = nan(nTrialsPerCond*nCond, nChan); %empty data matrix; row = trial col = channel

curr_cond = 1;

% Ways of generating random data: 
% 1. vector of random values drawn from a normal distribution with  mu=500 and SD=5.
%    y = sd*randn() + mean; 
% 2. vector of uniformly distributed numbers in the interval (-5,5) a=-5, b=5.
%    r = -5 + (5+5)*rand(10,1);

for i= 1:nChan % for each column (channel)
    for j=1:nTrials % and for each row (trial)
        data_mat(j,i) = sd_mat(curr_cond, i)*randn + mu_mat(curr_cond,i); %fill in data matrix for condition 1
        if j >= nTrialsPerCond + 1
            data_mat(j,i) = sd_mat(curr_cond+1,i)*randn + mu_mat(curr_cond+1, i); %fill in data matrix for condition 2 
        end
    end
end


% SVM
Mdl1 = fitcsvm(data_mat, respVarName); %figure out outputs

% calculate accuracy 


% cross validation 

CrossValMdl = crossval(Mdl1);

% Plot figures / Visualize data
% figure
X = data_mat;
Y = respVarName;

d = 0.02; % Step size of the grid
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];        % The grid
[~,scores1] = predict(Mdl1,xGrid); % The scores

figure;
h(1:2) = gscatter(X(:,1),X(:,2),Y);
hold on
h(3) = plot(X(Mdl1.IsSupportVector,1),...
    X(Mdl1.IsSupportVector,2),'ko','MarkerSize',10);
    % Support vectors
contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[0 0],'k');
    % Decision boundary
title('Scatter Diagram with the Decision Boundary')
legend({'-1','1','Support Vectors'},'Location','Best');
hold off
