clear;
addpath('../core');
addpath('../obj');
wef = '../../../core_data/skipwiki25.mat';
dataf='../../../../train_data/words.txt';
hiddenSize=25;
load(wef);
global params
params.constraints=0;

load('../../../core_data/wordmap.mat');

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
sample = play_data(1:20);

[We_new, sample_new, wordsT] = limitWords_justwords(sample, We_orig, words);
pairs = getPairs(sample_new,wordsT,We_orig);

%word derivatives
disp('testing word derivative');
Wefortraining = reshape(We_new,numel(We_new),1);
Wefortraining_m = We_new;
objectiveWordDerNew(hiddenSize, sample_new, pairs, wordsT, We_new, Wefortraining_m, .5, .5, 1,1);
tic;
numgrad_We = computeNumericalGradient( @(x) objectiveWordDerNew(hiddenSize, sample_new,pairs, wordsT, We_new, x,.5,.5,1,0), Wefortraining);
num1=toc;
tic;
numgrad_test = computeGradNewObjWords(sample_new, pairs, hiddenSize, wordsT, We_new, We_new, .5, 1);
num2=toc;
numgrad_test = reshape(numgrad_test, [numel(numgrad_test) 1]);
diff=scoreGradient(numgrad_We, numgrad_test);
fprintf('(Word) Num grad took %d s, computed grad took %d s, difference is %d\n',num1,num2,diff);

%AG test
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
sample = play_data(1:20);

[We_new, sample_new, wordsT] = limitWords_justwords(sample, We_orig, words);
pairs = getPairs(sample_new,wordsT,We_orig);

params.lambda_t=0; %just to prevent changing code
params.lambda_w=0.5;
params.margin=1;
params.data = sample_new;
params.batchsize = 5;
params.epochs = 5;
params.etat=0.05;
params.etaw=0.5;

AGWords_test(params, hiddenSize, wordsT, We_new);


%AG
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
sample = play_data(1:5000);

params.lambda_t=0;
params.lambda_w=0.0000000000001;
params.margin=1;
params.data = sample;
params.batchsize = 5;
params.epochs = 50;
params.etat=0.05;
params.etaw=0.5;
params.quiet=0;
params.evaluate=0;
params.save=0;
AGWords(params, hiddenSize, words, We_orig, 'temp');