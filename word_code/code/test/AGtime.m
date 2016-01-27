%AG
%250s gives 14.9s. Improved to 9.9s
clear;
addpath('../core');
addpath('../obj');
wef = '../../../core_data/skipwiki25.mat';
dataf='../../../../Paraphrase_Project_data/word_data_done/words.txt';
hiddenSize=25;
load(wef);

fid = fopen(dataf,'r');
data = textscan(fid,'%s%s', 'delimiter','\t');
play_data = {};
for i=1:1:length(data{1})
    play_data{end+1} = {data{1}{i} data{2}{i}};
end

global wordMap;
wordMap = containers.Map(words,1:length(words));

temp={};
for i=1:1:length(play_data)
    temp{end+1}=[WordLookup(play_data{i}{1}) WordLookup(play_data{i}{2})];
end
play_data=temp;

p = randperm(length(play_data));
play_data=play_data(p);
sample = play_data(1:250);

params.lambda_t=0;
params.lambda_w=0.0000000000001;
params.margin=1;
params.data = sample;
params.batchsize = 5;
params.epochs = 5;
params.etat=0.05;
params.etaw=0.5;
params.quiet=0;
params.evaluate=0;
params.save=0;
tic
AGWords(params, hiddenSize, words, We_orig, 'temp');
toc