function [] = train_single(l1,l2,frac,batchsize,datafile,output,scriptname)
addpath(genpath('../../single_W_code/'));
addpath(genpath('../code/'));
config1
l1
l2
frac
batchsize
datafile
output
%run(scriptname);
%hiddenSize = 25;
%params.etat=.05;
%params.etaw=.5;
params.lambda_t=l1;
params.lambda_w=l2;
%params.margin = 1;
hiddenSize=rnnoptions.hiddenSize;
params.etat=rnnoptions.etat;
params.etaw=rnnoptions.etaw;
params.margin=rnnoptions.margin;

temp = strcat(rnnoptions.output,'_');
temp = strcat(temp,num2str(l1));
temp = strcat(temp,'_');
temp = strcat(temp,num2str(l2));
temp = strcat(temp,'_');
temp = strcat(temp,num2str(batchsize));
temp = strcat(temp,'_');
temp = strcat(temp,output);
output=temp;

load('../../core_data/skipwiki25.mat');
load(rnnoptions.wordfile);
load(rnnoptions.init);
load(datafile);

train_data = [train_data test_data valid_data];

%use adagrad
%params.epochs = 20;
%params.save = 1;
params.epochs=rnnoptions.epochs;
params.save=rnnoptions.save;
params.quiet=rnnoptions.quiet;
params.data= train_data(1:round(frac*length(train_data)));
fprintf('Training on %i data using %f and %f\n',length(params.data),l1,l2);

params.batchsize = batchsize;
fprintf('Training on %d instances.\n',length(params.data));

AGParaWords(theta, params, hiddenSize, words, We_orig, output);
end