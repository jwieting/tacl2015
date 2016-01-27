#alias matlab=/Applications/MATLAB_R2014a.app/bin/matlab

echo "Training phrase model"
cd ../phrase_code/training/
matlab -nodisplay -nodesktop -nojvm -nosplash -r "train_single(0, 0.01, 0.05, 100, '../../../train_data/phrase_training_data_60k.txt.mat', 'phrase_60k', 'config1.m');quit"

cd ../..
