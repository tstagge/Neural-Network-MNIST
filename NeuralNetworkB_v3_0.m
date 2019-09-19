%% NEURAL NETWORK B, version 1.0
%  Written by Tyler J. Stagge, 2019.07
%  Purdue University, School of Mechanical Engineering

% [NETWORK OBJECTIVE]
%  A => The objective of this network is, essentially, to add 3 numbers.
%       This is an intentially simplistic use case for the purposes of
%       developing all the necessary functions to create and train a neural
%       network
%  B => This network is designed to recognize hand-written digits from the
%       MNIST database

% [VERSION CHANGELOG]
% [A1.0] (2019.07.03?-10)
%   -Ok, so it currently initializes the network with random weights and
%    biases, takes in training data inputs, computes test outputs, and 
%    calculates an average cost using the training data outputs
%   -I just barely started to implement Gradient Descent; starting that
%    made clear the need for a v1_1 that builds cost calculation into the
%    nnIO() function
% [A2.0] (2019.07.10-11)
%   -I just discovered MATLAB classes and objects, so I'm implementing
%    OOP here
%   -Created classes {nncLayer, nncNetwork}
%   -Notation change: what I previously called "levels" are now,
%    appropriately, "layers"
%   -Implemented GRADIENT DESCENT (currently the number of steps is just
%    hard-coded
%   -renamed nncNetwork's networkIO() to networkTrainIO and created a
%    separate networkTestIO that just takes an input (no expected output)
%    and identifies the output layer neuron with the max activation (no
%    cost calculation)
% [A2.1] (2019.07.11-13)
%   -Renamed the training case struct "IO Case Struct" and added a
%    'purpose' field (Note: this is in anticipation of NNB
% [B1.0] (2019.07.13-
%   -I jankily added a second network in parallel here
% [B1.1]
%   -I reverted back to a single network here: either random or loaded
% [B2.0]
%   -This is a parallel version to B1.1; this can run n randomized
%    networks, that way I can find one that looks like it will learn the
%    best
% [B3.0] (2019.07.15-
%   -This is an an attempt to combine the features of B1.1 and B2.0 via the
%    "MODE_NETS" variable
%   -I separated all the versions out into different folders; I can now
%    make more substantial edits to the classes
%   -Renamed "nncNetwork" to "nnNetwork"
%       >Eliminated the ununsed "DIRECTORY_NAME" variable
%       >Made storeNetwork & loadNetwork functions work properly
%       >FIXME: still working on the momentumGDS() function

%% clear et al.
clear; clc; close all;

%% NETWORK PARAMETERS
numLevels = 5;
numNeurons = [784, 256, 64, 25, 10]; %Number of neurons in each level
numGDS = 10; %How many (additional) GD Steps will be completed upon this execution of the program
gdsTestInc = 5;

MODE_NETS = 'SINGLE'; %{'SINGLE', 'SCATTER', 'PARALLEL'}
numNetworks = 1;
net_initRandom = true;
net_reloadFiles(1:numNetworks) = struct('filename', []);
net_reloadFiles(1) = 'NNB_r5_GDS385.mat'; %'NNB_GDS20.mat';


%% LOAD TRAINING & TEST DATA
load('NNB_TrainingData.mat'); %Variable 'TRAINING_CASES' is exported from 'NNB_Preprocessing.m'
numTrainingCases = length(TRAINING_CASES);
load('NNB_TestData.mat');
numTestCases = length(TEST_CASES);

%% INITIALIZE or RELOAD NETWORK(S)
NETWORKS = [];
switch(MODE_NETS)
    case 'SINGLE'
        if(net_initRandom)
            NETWORKS = nnNetwork(numLevels, numNeurons, true, 'dummy');
        else
            for n = 1:numNetworks
                NETWORKS(n) = nnNetwork(0,0,false,net_reloadFiles.filename);
            end
            %load(net_reloadFiles);
            %NETWORKS = NETWORK; NETWORK = [];
        end
    case 'SCATTER' %Multiple networks starting at diff. initial conditions
        for n = 1:numNetworks
            NETWORKS = [NETWORKS; nncNetwork(numLevels, numNeurons, true, DIRECTORY_NAME)];
        end
    case 'PARALLEL' %Multiple networks starting at same initial conditions
        NETWORKS(1:numNetworks) = nnNetwork(numLevels, numNeurons, true, DIRECTORY_NAME);
end

%% STOCHASTIC GRADIENT DESCENT
tic;
batchSize = 5000; numBatches = numTrainingCases/batchSize;
start = batchSize*(mod(1:numGDS, numBatches)) + 1;
stop = start + batchSize - 1;

gdsAvgBatchCost = zeros(numGDS+1,numNetworks); %On training data batches
%gdsAvgBatchCostInc = zeros(numGDS/gdsTestInc,numNetworks);
%gdsAvgTestCostInc = zeros(numGDS/gdsTestInc,numNetworks);
i = 1; j = 1;
stopGDS = false;

while((i <= numGDS) & (~stopGDS))
    if(strcmp(MODE_NETS,'SINGLE') | strcmp(MODE_NETS, 'SCATTER'))
        fprintf('Avg (batch) cost of network: ');
        for n = 1:numNetworks
            [NETWORKS(n), gdsAvgBatchCost(i,n)] = gradientDescentStep(NETWORKS(n), TRAINING_CASES(start:stop));
            fprintf('{N%d(GDS %d) = %.5f}\t', n, NETWORKS(n).GDS-1, gdsAvgBatchCost(i,n));
        end
        fprintf('(%.4f)\n', toc);
    elseif(strcmp(MODE_NETS,'PARALLEL'))
        [NETWORKS(1), gdsAvgBatchCost(i,1)] = gradientDescentStep(NETWORKS(1), TRAINING_CASES(start:stop));
        [NETWORKS(2), gdsAvgBatchCost(i,2)] = momentumGDS(NETWORKS(2), TRAINING_CASES(start:stop));
        fprintf('Avg (batch) cost of network: {N1(GDS %d) = %.5f}{N2(GDS %d) = %.5f}\n',...
            NETWORKS(1).GDS-1,gdsAvgBatchCost(i,1), NETWORKS(2).GDS-1,gdsAvgBatchCost(i,2));
    end
%     if(mod(i,gdsTestInc) == 0) %Every 5th GDS
%         gdsAvgBatchCostInc(j) = gdsAvgBatchCost(i);
%         tempCost = zeros(numTestCases,1);
%         for t = 1:numTestCases
%             [dump1, dump2, tempCost(t)] = networkTrainIO(NETWORK, TEST_CASES(t));
%         end
%         gdsAvgTestCostInc(j) = mean(tempCost);
%         
%         if(j > 5)
%             movingAvgTrain = mean(gdsAvgBatchCostInc(j-5:j));
%             movingAvgTest = mean(gdsAvgTestCostInc(j-5:j));
%             if(movingAvgTest > movingAvgTrain) %If network is memorizing, stop GDS
%                 stopGDS = true;
%             end
%         end
%         j = j + 1;
%     end
    i = i + 1;
end

%% TEST LAST ITERATION OF NETWORK
costs_lastNet = zeros(numTrainingCases,numNetworks);
fprintf('Avg (total) cost of network: ');
for n = 1:numNetworks
    for i = 1:numTrainingCases
        [dump1, dump2, costs_lastNet(i,n)] = networkTrainIO(NETWORKS(n), TRAINING_CASES(i));
    end
    gdsAvgBatchCost(end,n) = mean(costs_lastNet(:,n));
    fprintf('{N%d(GDS %d) = %.5f}\t', n, NETWORKS(n).GDS, gdsAvgBatchCost(end,n));
end
fprintf('\n');

 
fprintf('Time to complete %d x %d steps: %.4f\n', numGDS, numNetworks, toc);

%% PLOTS

figure; hold on;
for n = 1:numNetworks
    plot((NETWORKS(n).GDS-numGDS):(NETWORKS(n).GDS), gdsAvgBatchCost(:,n));
end
legend('N1', 'N2', 'N3', 'N4', 'N5');
xlabel('Gradient Descent Step');
ylabel('Average Cost');
title('Network Gradient Descent');
grid on;

