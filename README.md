# nova
This is a simple support research library based in PyTorch.

It is totally based in PyTorch, get some ideas from fast.ai courses and aim is to create code based in papers and new ideias and eventually be useful to others.

Idea is to keep simple.

It is totally based in callbaks to insert new features like AUC accounting and plots for instance.

There is available an example of Base callback usage. More examples and features will be added soon.

## Clone the repository and setup
 ```
 git clone https://github.com/dpetrini/nova.git
 ```
To run using other folders for your code, set PYTHON_PATH to the location you downloaded Nova:
 ```
export PYTHONPATH="/home/.../nova"
 ```

 # Available features
  - Base Callback - run basic training and generates models, plots
  - AUC Callback - Calculates AUC for classification tasks, generates best model, plots
  - More to come...

# Usage of Base Callback

 - Declare the callbacks with this statement:
 ```
 cb = CallbackHandler([BaseCB('project-name')])
 ```
By doing this you assure the accounting of accuracy, create a name for project and showing training progress.

## Features of Base Callback

 - Train and save last and best models
 - Create Loss and Accuracy plots
 - Insert date and time in generate file names
 - Create default location for generated models and plots, both will be placed in 'models_project-name' and 'plots_project-name'.

## Example

This example is available in the repository and shows the features of Base Callback.

- Dataset preparation

In the file example_train.py you find a sample on how to use this library. In order to run we use a subset of cats & dogs dataset available from Kaggle, you can download it from:

- https://www.kaggle.com/c/dogs-vs-cats/data 

It has a total of 25k images, split equally between both categories. Please create a subset of 2k of each category. Create a top level folder 'data_cats_dogs' and divide the data in internal folders like this:

 - train/cats: 1000 samples of cats
 - train/dogs: 1000 samples of dogs
 - val/cats: 500 samples of cats
 - val/dogs: 500 samples of dogs
 - test/cats: 500 samples of cats
 - test/dogs: 500 samples of dogs

 Obs. Alternatively you can download this prepared dataset from here: https://drive.google.com/file/d/1Uya18oAg2nTHURXm_4t6ODw8dDR49n6L/view?usp=sharing .

 - Running train
 
 To run the training execute:

 ```
 python3 exemple_train.py
 ```

 - Summary of results:

| Number | Feature       | Accuracy      | Setup          |
| ------ | ------------- | ------------- | -------------- |
| 1 | Basic Resnet  | 0.7140  |  50 epochs, optim = Adam, LR = 3e-3, Init = Kaiming    |
| 2 | Basic Resnet pre-trained on imagenet |   |       |

- Generated artifacts

Models and plots will be stored in folders 'models_example' and 'plots_example'.

- Plots

| Accuracy | Loss |
|----------|------|
| ![Alt text](images/2020-07-12-08h28m_acc_curve.png?raw=true "Accuracy Curve") | ![Alt text](images/2020-08-12-08h28m_loss_curve.png?raw=true "Loss Curve") |

- Training progress output example
 ```
Epoch: 6/50
▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░
Epoch: 006, Train: Loss: 0.6864, Acc: 58.20%, Val: Loss: 0.6684, Acc: 59.00%, Time: 15.89s

Epoch: 7/50
▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░
Epoch: 007, Train: Loss: 0.6669, Acc: 58.20%, Val: Loss: 0.7107, Acc: 58.00%, Time: 16.11s

Epoch: 8/50
▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░ |------>  Best Val Acc model now 0.6310
Epoch: 008, Train: Loss: 0.6598, Acc: 59.15%, Val: Loss: 0.6377, Acc: 63.10%, Time: 15.96s
 ```

 Yes nice and simple output in base configuration.




