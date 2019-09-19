% Written for "NeuralNetworkA_v2_0.m"

classdef nnLayer
    properties
        layer = []; %(l) Index of current level
        n_l = [];   %Number of neurons in level l (current level)
        n_lm1 = []; %Number of neurons in level l-1 (previous level)
        type = [];  %[Currently initialized but never utilized] {-1,0,1} = {input, hidden, output}

        Z = []; %[n_l x 1]     A, but before the sig() fn.
        A = []; %[n_l x 1]     Activations of this level's neurons
        W = []; %[n_l x n_lm1] Weights between this and prev. level's neurons
        B = []; %[n_l x 1]     Biases of this level's neurons

        currGDS = 0;        %[Not currently used at all] Current Gradient Descent Step
        currGDS_numTEs = 0; %[Not currently used at all] Number of training examples (TEs) so far checked for this gradient descent step
        dC0_dWl = [];    %[n_l x n_lm1] part. der. of costOfNetwork(currentTrainingExample) w.r.t. W(l)
        dC0_dBl = [];    %[n_l x 1]     part. der. of costOfNetwork(currentTrainingExample) w.r.t. B(l)
        dC0_dAlm1 = [];  %[n_lm1 x 1]   part. der. of costOfNetwork(currentTrainingExample) w.r.t. A(l-1)
        sum_dC0_dWl = 0;   %[n_l x n_lm1]  sum of 'dC0_dWl' across every trainingExample so far checked on this GDStep
        sum_dC0_dBl = 0;   %[n_l x 1]      sum of 'dC0_dBl' across every trainingExample so far checked on this GDStep
    end
    
    methods
       % CONSTRUCTOR/INIT METHOD
        function obj = nnLayer(l, nl, nlm1, typ)
            obj.layer = l;
            obj.n_l = nl;
            obj.n_lm1 = nlm1;
            obj.type = typ;
            
            % Initialize Z,A,W,B to zero matrices of the appropriate size
            obj.Z = zeros(obj.n_l, 1);
            obj.A = obj.Z;
            if(l>1) %Layer 1 (input layer) lacks both W and B
                obj.W = zeros(obj.n_l, obj.n_lm1);
                obj.B = zeros(obj.n_l, 1);
            end
        end
       
       % I don't currently have any plans for additional functions; just
       %  about everything seems to fit more cleanly in the network class
    end
end