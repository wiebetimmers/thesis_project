# thesis_project

Update 15-5:
- implemented pickle for better model / results saving
- implemented more distributions for random perturbation, mu and sigma are now tunable in the paremeters section
- plotting the results from the pickle files are now in a new file: main_digits_plot.py , saves time (first we had to run the complete model before plotting every time)
- plots and models are saved in their *new* respective directories. 

Update 4-5:
- 2 main files, for easier processing the digits or mnist experiments
- Changed all notebooks to .py files, processing time improved a lot. Added a Parameters.py file.
- Now also possible to do mutation and crossover on the bias weights


main_digits.py and main_mnist.py includes 3 cases so far:
- Baseline RNN, with full backpropagation
- Reservoir RNN, only output is trained, no evolutionairy optimization is used
- Reservoir RNN with evolution optimization: the output is trained for x amount of epochs, after that the EA is used to optimize either accuracy or loss  -> we only plot the best performing model in the population afterwards. 


### Still working on:
- Proper saving of the models
- Additional mutation, crossover, selection methods
- A situation in which we keep the output layer constant after a few training cycles, then only perform EA optimization on the non-output layers. 




