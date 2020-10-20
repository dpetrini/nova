# nova
This is a simple support research library based in PyTorch.

It is totally based in PyTorch, get some ideas from fast.ai courses and aim is to create code based in papers and new ideias and eventually be useful to others.

Idea is to keep simple and provide a simple entrance to PyTorch to a beginner researcher.

You configure the features you need in a simple structure (dictionary) and the engine will enable the resources easily. New features are added in this way, making compatibility among upgrades.

If you want to add new features, like a scheduler for instance, just look in the code and change according to your needs. The proposal is to keep code simple and readable to make customisations easier.

There is an example available. More examples and features will be added soon.

## Clone the repository and setup
 ```
 git clone https://github.com/dpetrini/nova.git
 ```
To run using other folders for your code, set PYTHON_PATH to the location you downloaded Nova:
 ```
export PYTHONPATH="/home/.../nova"
 ```

 # Available features
  - Base Training - run basic training and generates models, plots, etc.
  - Nice naming of generated models and plots making easy to track progress, automatically:
  ![Alt text](images/naming_features.png?raw=true "Naming features")
  - AUC Stats - Calculates AUC for classification tasks, generates best auc model and plots.
  - More to come...

# Configuration of Trainer

By default, training session will have accuracy statistics, last and best models and plots saved in 'models_example' and 'plots_example' folders respectively (config 'name' = 'example', as below).

 - Select your training configuration:
 ```
    train_config = {
        'num_epochs': NUM_EPOCHS,
        'batch_size': MINI_BATCH,
        'name': 'example',
        'title': 'Cats & Dogs Classifier',
        'features': ['auc', 'lr_warmup_cyc_cos'],   # add stats, schedulers
        'save_last': True,          # optional: Save last model (default=False)
        'save_best': True,          # optional: Save best model (default=True)
        'save_path': folder/,       # if want to save artifacts in other place (default=current)
        'show_plots': False,        # if want to show plots opening window
        'make_plots': False,        # if want to disable plots
    }
 ```

The 'features' list is optional. It contains additional features to be considered in training like 'auc' statistics collection and ploting.
 - Load the model, configure the dataloaders (according to PyTorch defaults and in exemples here), set loss_func, optimizers and start training session by instantiating Trainer object:

 ```
    session = Trainer(model, train_dataloader, val_dataloader, loss_func,
                      optimizer, optim_args, device, train_config)
```

 - Then call train_and_validate to start training:
```
    # train the model
    session.train_and_validate()
 ```
 - Additionally you can run test_dataloader aginst the last and best models:
 ```
    session.run_test(test_dataloader, "normal")
    session.run_test(test_dataloader, "best")
  ```
  and get the result:
  ```
    Model: normal - Test accuracy : 0.626 Test loss : 0.699
    Model: best - Test accuracy : 0.672 Test loss : 0.626
  ```
## Features of Base Config

 - Train and save last and best models
 - Create Loss and Accuracy plots
 - Insert date and time in generate file names
 - Create default location for generated models and plots, both will be placed in 'models_project-name' and 'plots_project-name'.

## Additional Features

 - AUC, stands for Area Under ROC (Receiver Operating Characteristic Curve) Curve, it considers the true positive rate (TPR) against the false positive rate (FPR) at various thresholds. When enabled it calculates the AUC value for each training epoch and plots the result for whole training when finishes.
 Output per epoch:
 ```
[Train] AUC: 0.539 [Val] AUC: 0.585 |------>  Best Val Auc model now 0.5853
 ```
You can also generate and save the AUC curve of test dataset with:
```
session.run_test_auc(test_dataloader, "best")
```

## Example

This example is available in the repository and shows the features of Base Config.

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

| Number | Feature       | Accuracy % (validation)     | Setup          |
| ------ | ------------- | ------------- | -------------- |
| 1 | Basic Resnet  | 71.40  |  50 epochs, optim = Adam, LR = 3e-3, Init = Kaiming    |
| 2 | Basic Resnet pre-trained on imagenet | 81.10  | 50 epochs, optim = Adam, LR = 3e-3, AMP      |

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

 - AUC Example generation: 
By default will calculate AUC of second category of binary inputs. In our example cats is first category (0), dogs is the second category (1), so the AUC is considering correct dog predictions.
 Steps:
 Make sure to include following line in train_config dictionary:
 ```
  'features': ['auc'],
 ```
 It will deliver per epoch stats. Also request AUC curve for test set placing the following line after training:
```
  session.run_test_auc(test_dataloader, "best")
```
AUC Plots:
| Per epoch AUC value | ROC Curve & AUC for test dataset |
|----------|------|
| ![Alt text](images/2020-09-22-17h35m_AUC_curve_AUC_08559.png?raw=true "Training AUC") | ![Alt text](images/2020-09-22-17h35m_Category_1_ROC.png?raw=true "Test AUC Curve") |

Please check this example in file example_train.py.






