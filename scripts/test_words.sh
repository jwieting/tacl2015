#alias matlab=/Applications/MATLAB_R2014a.app/bin/matlab

echo "Training word model"
cd ../word_code/training/
matlab -nodisplay -nodesktop -nojvm -nosplash -r "train_single(0, 0.0001, 0.05, 'words-xl', '../../../train_data/words.txt', 100, 'config1.m');quit"

cd ../../scripts
