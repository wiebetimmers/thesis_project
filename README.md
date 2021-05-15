# thesis_project

Update 15-5:
- implemented pickle for better model / results saving
- implemented more distributions for random perturbation, mu and sigma are now tunable in the paremeters section
- plotting the results from the pickle files are now in a new file: main_digits_plot.py , saves time (first we had to run the complete model before plotting every time)
- plots and models are saved in their *new* respective directories. 

run main_digits.py in order to train the 3 models
run main_digits_plot.py to plot 3 models (trained models already present for plotting in the plots directory) 


### Still working on:
- Keeping the best performing model in the baseline and no evo reservoir model. 
- Additional mutation, crossover, selection methods
- A situation in which we keep the output layer constant after a few training cycles, then only perform EA optimization on the non-output layers. 
- Experimenting with a dynamical perturb rate 
- Experimenting with different mu's and sigma's for some random perturbation methods




