#alias matlab=/Applications/MATLAB_R2014a.app/bin/matlab

echo "Training bigram model"
cd ../bigram_code/training/
matlab -nodisplay -nodesktop -nojvm -nosplash -r "train_single(0, 0.01, 0.05, 100, '../../../train_data/adj-noun-base-xl.txt.mat', 'adj-noun-xl', 'config1.m');quit"

cd ../..
