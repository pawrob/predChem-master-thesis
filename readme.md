# PredChem

Long story short, there are chemical particles that can combine with albumin serum and stop some diseases like
parkinson. The problem is that searching for those particles is time-consuming and need some laboratory work, so
searching for them with neural network can be beneficial. Chemical databases contains only information about particles
that successfully can combine with albumin, that's the reason for one Class classifier im trying to implement.

## Requirements

* Python 3.10.11
* Pytorch 2.0.1
* Torch-geometric 2.3.1
* torchvision 0.15.2
* matplotlib 3.7.1
* numpy 1.24.3
* Scikit-learn 1.2.2
* pytorch-model-summary 0.1.2
* tabulate 0.8.9

## While using conda

````
conda install pytorch torchvision -c pytorch matplotlib scikit-learn
conda install -c conda-forge pytorch-model-summary

python main.py --network=[ocgnn/binary_classification] --plot=[save/show]
````

