#$1 lam 1
#$2 lam 2
#$3 fraction of training data to use
#$4 output
#$5 data

matlab -nodisplay -nodesktop -nojvm -nosplash -r "train_single($1,$2,$3,'$4','$5');quit" &