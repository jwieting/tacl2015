
%AG
%                                                                                                                                                                                                                                                                                                     
clear;
addpath('../core');
addpath('../obj');
wef = '../../../core_data/skipwiki25.mat';
initv = '../../../core_data/theta_init_25.mat';
dataf='../../../core_data/play_data.mat';
hiddenSize=25;
load(initv);
load(wef);
load(dataf);

p = randperm(length(play_data));
play_data=play_data(p);
sample=play_data;

sample = feedForwardTrees(sample, theta, hiddenSize, We_orig);

params.lambda_t=0.0000000000001;
params.lambda_w=0.0000000000001;
params.margin=1;
params.data = sample(1:1000);
params.batchsize = 5;
params.epochs = 5;
params.etat=0.05;
params.etaw=0.5;
params.save=0;
tic
AGWords(theta, params, hiddenSize, words, We_orig, 'temp');
toc