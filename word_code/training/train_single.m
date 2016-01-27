function [] = train_single(l1,l2,frac,output,datafile,batchsize,scriptname)
addpath(genpath('../code/'));
%run(scriptname);
config1 %hack
%hiddenSize = 25;
%params.etat=.05;
%params.etaw=.5;
l2
frac
output
datafile
batchsize
params.constraints=rnnoptions.constraints;
params.lambda_t=l1;
params.lambda_w=l2;
%params.margin = 1;
hiddenSize=rnnoptions.hiddenSize;
params.etat=rnnoptions.etat;
params.etaw=rnnoptions.etaw;
params.margin=rnnoptions.margin;

temp = strcat(rnnoptions.output,'_');
temp = strcat(temp,num2str(l2));
temp = strcat(temp,'_');
temp = strcat(temp,num2str(batchsize));
temp = strcat(temp,'_');
temp = strcat(temp,output);
output=temp;

load(rnnoptions.wordfile);
load('../../core_data/wordmap.mat');
global wordmap;

fid = fopen(datafile,'r');
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
train_data=play_data(p);

%use adagrad
%params.epochs = 20;
%params.save = 1;
params.epochs=rnnoptions.epochs;
params.save=rnnoptions.save;
params.quiet=rnnoptions.quiet;
params.data= train_data(1:round(frac*length(train_data)));
fprintf('Training on %i data using %f and %f\n',length(params.data),l1,l2);

params.batchsize = batchsize;
params.constraints=rnnoptions.constraints;
global params;
fprintf('Training on %d instances.\n',length(params.data));

AGWords(params, hiddenSize, words, We_orig, output);
end