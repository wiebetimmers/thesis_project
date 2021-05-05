# thesis_project

Update 4-5:
- 2 main files, for easier processing the digits or mnist experiments
- Changed all notebooks to .py files, processing time improved a lot. Added a Parameters.py file.
- Now also possible to do mutation and crossover on the bias weights


main.py includes 3 experiments so far:
- Baseline RNN, with full backpropagation
- Reservoir RNN, only output is trained, no evolutionairy optimization is used
- Reservoir RNN with evolution optimization: the output is trained for x amount of epochs, after that the EA is used to optimize either accuracy or loss  -> we only plot the best performing model in the population afterwards. 


### Still working on:

Proper saving of the models
Additional mutation, crossover, selection methods
A situation in which we keep the output layer constant after a few training cycles, then only perform EA optimization on the non-output layers. 




