# thesis_project

Update 3-5:
Fixed some errors regarding the mutation functions.
Changed from accuracy to classification error, objective function is now to minimize the classification error. 

Update 2-5:
Added a plot to visualize the final population. 

Update 1-5: 

run main.ipynb to import the datasets, neural networks, ea and performing operations (like training and evaluation). 

main.ipynb includes 3 experiments so far:
- Baseline RNN, with full backpropagation
- Reservoir RNN, only output is trained, no evolutionairy optimization is used
- Reservoir RNN with evolution optimization: the output is trained for x amount of epochs, after that the EA is used to optimize either accuracy or loss  -> we only plot the best performing model in the population afterwards. 


### Still working on:

Finding a well performing EA (still looking into the literature)
- What mutation to use
- What recombination to use
- What selection mechanism to use
- How to asses fitness. (minimize loss? or maximize accuracy? ) 

Proper saving of the models



