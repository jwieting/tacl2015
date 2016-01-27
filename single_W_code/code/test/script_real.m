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
sample = play_data(1:10);

sample = feedForwardTrees(sample, theta, hiddenSize, We_orig);

[We_new, sample_new, wordsT] = limitWords(sample, We_orig, words);
pairs = getPairs(sample_new,wordsT);
[pairs, sample_new] = getPairsBatch(sample_new,wordsT,100);

%test matrix derivative
disp('testing matrix derivative');
thetafortraining = theta;
Wefortraining = reshape(We_new,numel(We_new),1);
Wefortraining_m = We_new;
objectiveWordDerNew(thetafortraining, hiddenSize, sample_new, pairs, wordsT, We_new, Wefortraining_m, .5, .5, 1,1);
tic;
numgrad_theta = computeNumericalGradient( @(x) objectiveWordDerNew(x, hiddenSize, sample_new,pairs, wordsT, We_new, We_new,.5,.5,1,0), thetafortraining);
num1 = toc;
tic;
numgrad_test = computeGradNewObj(sample_new, pairs, thetafortraining, hiddenSize, wordsT, We_new, .5, 1);
numgrad_test = numgrad_test(:);
num2 = toc;
diff = scoreGradient(numgrad_theta, numgrad_test);
fprintf('(Matrix) Num grad took %d s, computed grad took %d s, difference is %d\n',num1,num2,diff);

%word derivatives
disp('testing word derivative');
thetafortraining = theta;
Wefortraining = reshape(We_new,numel(We_new),1);
Wefortraining_m = We_new;
objectiveWordDerNew(thetafortraining, hiddenSize, sample_new, pairs, wordsT, We_new, Wefortraining_m, .5, .5, 1,1);
tic;
numgrad_We = computeNumericalGradient( @(x) objectiveWordDerNew(thetafortraining, hiddenSize, sample_new,pairs, wordsT, We_new, x,.5,.5,1,0), Wefortraining);
num1=toc;
tic;
numgrad_test = computeGradNewObjWords(sample_new, pairs, thetafortraining, hiddenSize, wordsT, We_new, We_new, .5, 1);
num2=toc;
numgrad_test = reshape(numgrad_test, [numel(numgrad_test) 1]);
diff=scoreGradient(numgrad_We, numgrad_test);
fprintf('(Word) Num grad took %d s, computed grad took %d s, difference is %d\n',num1,num2,diff);

%AG test
load(dataf);

p = randperm(length(play_data));
play_data=play_data(p);

sample = play_data(1:10);

sample = feedForwardTrees(sample, theta, hiddenSize, We_orig);

[We_new, sample_new, wordsT] = limitWords(sample, We_orig, words);
pairs = getPairs(sample_new,wordsT);
[pairs, sample_new] = getPairsBatch(sample_new,wordsT,100);

params.lambda_t=0.5;
params.lambda_w=0.5;
params.margin=1;
params.data = sample_new;
params.batchsize = 5;
params.epochs = 5;
params.etat=0.05;
params.etaw=0.5;
AGWords_test(theta, params, hiddenSize, wordsT, We_new);


%AG
load(dataf);

p = randperm(length(play_data));
play_data=play_data(p);
sample=play_data;

sample = feedForwardTrees(sample, theta, hiddenSize, We_orig);

params.lambda_t=0.0000000000001;
params.lambda_w=0.0000000000001;
params.margin=1;
params.data = sample;
params.batchsize = 5;
params.epochs = 5;
params.etat=0.05;
params.etaw=0.5;
params.save=0;
params.quiet=1;
AGWords(theta, params, hiddenSize, words, We_orig, 'temp');