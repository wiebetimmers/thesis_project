# thesis_project

Update 4-5:
- Changed all notebooks to .py files, processing time improved a lot. Added a Parameters.py file.
- Would really like to run it on the compute.vu.nl service, but my student disk space is full (max 1000 mb for students at the VU) 

main.py includes 3 experiments so far:
- Baseline RNN, with full backpropagation
- Reservoir RNN, only output is trained, no evolutionairy optimization is used
- Reservoir RNN with evolution optimization: the output is trained for x amount of epochs, after that the EA is used to optimize either accuracy or loss  -> we only plot the best performing model in the population afterwards. 


### Still working on:

Finding a well performing EA (still looking into the literature)
- What mutation to use
- What recombination to use
- What selection mechanism to use
- How to asses fitness. (minizing classification error )

Proper saving of the models



