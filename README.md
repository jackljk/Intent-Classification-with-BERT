# PA4-Instructions
1. To run the model, cd to the folder where the model is located then write `bash run.sh` 
2. The default hyperparameters for the model are in arguments.py
3. To change these hyperparameters to your own liking, create a new line in run.sh starting with `python main.py`
4. Let's say for example you wanted to train a custom model with the number of reinitialization layers as 4 and number of epochs as 10. In that case you would make a new line in run.sh and write `python main.py --n-epochs 10 --do-train --task custom --reinit_n_layers 4`
5. Comment out all the other lines besides the line you want to run in run.sh
6. If you want to generate a plot, scroll to the bottom of main.py and set the hyperparameter `plot = True` for the task you are running.

Our team name is cse151b251b-wi24-pa4-ado2 <br />
Our github repo is linked [here](https://github.com/cse151bwi24/cse151b251b-wi24-pa4-ado2)

