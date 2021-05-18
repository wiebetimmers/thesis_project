# thesis_project

Update 18-5:
- dynamic perturb rate implemented
- run Experiment1_distributions.py to run a first experiment on the distributions used for mutations (only for the EA Reservoir RNN). In a later stage, this experiment will be conducted for various perturb rate decays and various variances. (on Digits dataset) 
- run main_digits.py to run 3 models: baseline, Reservoir without EA optimization and Reservoir with EA optimization. Run main_digits_plot.py to access the results / plots. 

### Working on:
- Keeping the best performing model in the baseline and no evo reservoir model. 
- Additional selection methods
- A situation in which we keep the output layer constant after a few training cycles, then only perform EA optimization on the non-output layers. 
- Experimental setup / code to automate grid search. 




