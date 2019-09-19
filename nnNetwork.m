% Written for "NeuralNetworkA_v2_0.m"

classdef nnNetwork
    properties
        %Lamda Functions:
        sig = @(x) 1./(1+exp(-x));              %Sigmoid function
        sig_d = @(x) exp(-x)./((1+exp(-x)).^2); %Derivative of the sigmoid fn
        
        numLayers = -1;
        numNeurons = []; %Number of neurons in each layer
        LAYERS = [];     %List of the objects for each layer
        
        GDS = 0;  %The number of steps this network has taken since being randomly initialized
        gdsK = 1; %Gradient descent step proportionality constant
    end
    
    methods
        %% INITIALIZATION fn.s
        % CONSTRUCTOR/INIT METHOD
        function self = nnNetwork(nL, nN, initRand, loadfile)
            if(initRand)
                self.numLayers = nL;
                self.numNeurons = nN;
                for l = 1:self.numLayers %Create Layer (nncLayer) Objects
                    newType = 0; %'type' of new
                    switch(l)
                        case 1              %Input Layer (type -1)
                            newType = -1;
                            newNLM1 = [];
                        case self.numLayers %Output Layer (type 1)
                            newType = 1;
                            newNLM1 = nN(l-1);
                        otherwise           %Hidden Layer (type 0)
                            newNLM1 = nN(l-1);
                    end
                    newLayer = nnLayer(l, nN(l), newNLM1, newType); %Auto inits Z,A,W,B to zero
                    self.LAYERS = [self.LAYERS; newLayer];
                end
                self = initRandParams(self);
                fprintf('RANDOM NETWORK INITIALIZED\n');
            else
                self = loadNetwork(self, loadfile);
                fprintf('NETWORK LOADED (%s)\n', loadfile);
            end
        end
        
        function self = initRandParams(self)
            for l = 1:self.numLayers
                n_l = self.LAYERS(l).n_l;     %Temporary alias
                n_lm1 = self.LAYERS(l).n_lm1; %Temporary alias
                if(l>1) %Layer 1 (input layer) lacks both W and B
                    self.LAYERS(l).W = randn(n_l, n_lm1);
                    self.LAYERS(l).B = randn(n_l, 1);
                end
            end
            GDS = 0; %New network (effectively)
        end
        
        %% NETWORK I/O and COST
        function [self, outputVec, cost] = networkTrainIO(self, trCaseStr)
            % i) self = the object of the nncNetwork class representing the complete network
            % i) trCaseStr = the training case struct w/ the I/O info
            % o) outputVec = A vector of the output layer
            % o) cost = the network's cost for this training case
            self.LAYERS(1).A = trCaseStr.inputVec;
            for l = 2:self.numLayers
                self.LAYERS(l).Z = self.LAYERS(l).W*self.LAYERS(l-1).A + self.LAYERS(l).B; %[n_l x 1]
                self.LAYERS(l).A = self.sig(self.LAYERS(l).Z);
            end
            outputVec = self.LAYERS(end).A;
            cost = sum( (outputVec - trCaseStr.outputVec).^2 );
        end
        
        function outputI = networkTestIO(self, input)
            % i) self = the object of the nncNetwork class representing the complete network
            % i) input = EITHER just a column vector ='ing the inputv layer's activations
            %            OR a ioCase struct -- either way, its just the test case
            % o) outputI = the index of the output layer with the maximum activation
            
            if(isstruct(input)) %input is a training case struct (but used for a test case)
                self.LAYERS(1).A = input.inputVec;
            else %input is just a column vector
                self.LAYERS(1).A = input;
            end
            for l = 2:self.numLayers %Compute output
                self.LAYERS(l).Z = self.LAYERS(l).W*self.LAYERS(l-1).A + self.LAYERS(l).B; %[n_l x 1]
                self.LAYERS(l).A = self.sig(self.LAYERS(l).Z);
            end
            maxOutActivation = 0;
            i_maxOutNeuron = 0;
            for i = 1:self.LAYERS(end).n_l %Indentify output neuron
                if(self.LAYERS(end).A(i) > maxOutActivation)
                    maxOutActivation = self.LAYERS(end).A(i);
                    i_maxOutNeuron = i;
                end
            end
            outputI = i_maxOutNeuron;
        end
        
        function accuracy = getNetworkAccuracy(self, ioCases)
            nTC = length(ioCases);
            numCorrect = 0;
            for t = 1:nTC
                if(ioCases(t).outputI == networkTestIO(self, ioCases(t)))
                    numCorrect = numCorrect + 1;
                end
            end
            accuracy = numCorrect/nTC;
            %fprintf('Accuracy: %.3f %%\n', 100*accuracy);
        end
        
        %% GRADIENT DESCENT
        function [self, avgCost] = gradientDescentStep(self, trCaseBatch)
            % i/o) self = the network (object of nncNetwork class)
            % i) trCaseBatch = the list of all training case structs to be
            %    used for this GDS
            % o) avgCost = average cost across the batch with the previous
            %              step's W,B
            cost_sum = 0;
            nTCB = length(trCaseBatch); %number of training cases in this batch
            for t = 1:nTCB
                %(1) Calculate network output for given training input
                %(2) Calculate C0 (cost for current training case)
                [self, outVec, C0] = networkTrainIO(self, trCaseBatch(t));
                cost_sum = cost_sum + C0;
                
                %(3:CALCULUS!) Calculate and sum dC0's for each training case
                for l = flip(2:self.numLayers) %Needs to start with output layer
                    % Calcualte dC0's
                    if(l == self.numLayers) %Output Layer
                        self.LAYERS(l).dC0_dBl = (2*(self.LAYERS(l).A - trCaseBatch(t).outputVec).*self.sig_d(self.LAYERS(l).Z));
                    else                    %Hidden Layer
                        self.LAYERS(l).dC0_dBl = self.LAYERS(l+1).dC0_dAlm1.*self.sig_d(self.LAYERS(l).Z);
                    end
                    self.LAYERS(l).dC0_dWl = self.LAYERS(l).dC0_dBl*transpose(self.LAYERS(l-1).A); %[n_l x n_lm1]
                    self.LAYERS(l).dC0_dAlm1 = zeros(self.LAYERS(l).n_lm1, 1);
                    for k =1:self.LAYERS(l).n_lm1
                        self.LAYERS(l).dC0_dAlm1(k) = sum(self.LAYERS(l).dC0_dBl .* self.LAYERS(l).W(:,k));
                    end
                    
                    % Sum dC0's
                    self.LAYERS(l).sum_dC0_dWl = self.LAYERS(l).sum_dC0_dWl + self.LAYERS(l).dC0_dWl;
                    self.LAYERS(l).sum_dC0_dBl = self.LAYERS(l).sum_dC0_dBl + self.LAYERS(l).dC0_dBl;
                    %self.LAYERS(l).sum_dC0_dAlm1 = self.LAYERS(l).sum_dC0_dAlm1 + self.LAYERS(l).dC0_dAlm1;
                end
            end
            
            for l = 2:self.numLayers
                %fprintf('Modifying layer %d\n',l);
                %(4) Average dC's across all training cases
                dC_dWl = (self.LAYERS(l).sum_dC0_dWl)./nTCB; %Temporary values
                dC_dBl = (self.LAYERS(l).sum_dC0_dBl)./nTCB;
                %(5) Modify W,B parameters using dC's
                self.LAYERS(l).W = self.LAYERS(l).W - self.gdsK*dC_dWl;
                self.LAYERS(l).B = self.LAYERS(l).B - self.gdsK*dC_dBl;
            end
            % (6) Reset the sums to 0 before next gradient descent step
            self.LAYERS(l).sum_dC0_dWl = 0;
            self.LAYERS(l).sum_dC0_dBl = 0;
            avgCost = cost_sum/nTCB;
            self.GDS = self.GDS + 1;
        end
        
        function [self, batchCost] = momentumGDS(self, trCaseBatch)
            
        end
        
        %% FILE I/O
        function void = storeNetwork(NETWORK, filename)
            save(filename, 'NETWORK'); %Every network always gets saved as "NETWORK"
        end
        function self = loadNetwork(self, filename)
            load(filename);
            self = NETWORK;
        end
    end    
end