##################################################
# LIBRARIES AND CONSTANTS
##################################################
import numpy as np
from neurons import Neurons

n_input_neurons = 100
n_hidden_neurons = 20
n_output_neurons = 10
weights_in_hi_filename = "input_weights.txt"
weights_hi_out_filename = "hidden_weights.txt"


##################################################
# MAIN
##################################################
print("###################################################")
print("#                                                 #")
print("#            AI FOR NUMBER RECOGNITION            #")
print("#                                                 #")
print("###################################################\n")

isActive = True
while isActive: 
    
    user_input = "input_file.txt"
    n = 1
    for i in range(n):
        with open(user_input, "r") as input_file:
            x_input = np.loadtxt(input_file)
            x_input = x_input.flatten()
            print("\nInput values:")
            print(x_input)
        
        Neurons(n_input_neurons, n_hidden_neurons, n_output_neurons, weights_in_hi_filename, weights_hi_out_filename, x_input)

        
        
        
    """
    # Saving input-hidden weights to a txt file
    print("\n--------------------Saving New Weights--------------------")
    np.savetxt(input_weights_filename, weights_in_hi, fmt="%.5f")"""
    
    
    # Want ot continue?
    wantToContinue = input("Do you want to try again this simple AI? (y/n) ")
    isActive = wantToContinue.lower() not in ["n", "no"]