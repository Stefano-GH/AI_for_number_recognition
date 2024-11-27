##################################################
# LIBRARIES AND CONSTANTS
##################################################
import os
import numpy as np

n_input_neurons = 100
n_hidden_neurons = 20
n_output_neurons = 10
input_weights_filename = "input_weights.txt"
hidden_weights_filename = "hidden_weights.txt"

# Functions definition
def generate_gaussian_matrix(n, m, mean=0, st_dev=0.1):
    """Genera una matrice casuale con distribuzione gaussiana"""
    random_matrix = np.random.normal(loc=mean, scale=st_dev, size=(n, m))
    return random_matrix

def relu(x):
    """Funzione di attivazione ReLU"""
    return np.maximum(0, x)

def softmax(x):
    """Funzione di attivazione softmax per il livello di output"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


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
    
    print("\n--------------------Reading Existing Weights--------------------")
    # Reading / generating input weights
    try:
        with open(input_weights_filename, "r") as input_weights_file:
            weights_in_hi = np.loadtxt(input_weights_file)
        
        if weights_in_hi.shape != (n_hidden_neurons, n_input_neurons):
            weights_in_hi = generate_gaussian_matrix(n_hidden_neurons, n_input_neurons)
            
        print("Input-Hidden weights: ")
        print(weights_in_hi)
    except:
        weights_in_hi = generate_gaussian_matrix(n_hidden_neurons, n_input_neurons)
        print("Input-Hidden weights: ")
        print(weights_in_hi)
    
    # Reading / generating hidden weights
    try:
        with open(hidden_weights_filename, "r") as hidden_weights_file:
            weights_hi_out = np.loadtxt(hidden_weights_file)
        
        if weights_hi_out.shape != (n_output_neurons, n_hidden_neurons):
            weights_hi_out = generate_gaussian_matrix(n_output_neurons, n_hidden_neurons)
            
        print("Hidden-Output weights: ")
        print(weights_hi_out)
    except:
        weights_hi_out = generate_gaussian_matrix(n_output_neurons, n_hidden_neurons)
        print("Hidden-Output weights: ")
        print(weights_hi_out)
    
    
        
    # Ask to the user for input signal
    user_input = None
    while user_input == None:
        user_input_methods = input("\nChoose the method for input signal. Enter the filename or 'M' to the massive input: ")
        if user_input_methods == "M":
            print("Not yet implemented...")
        elif user_input_methods == "-1":
            user_input = "input_file.txt"
            print("Test method selected!")
            
            with open(user_input, "r") as input_file:
                x_input = np.loadtxt(input_file)
                x_input = x_input.flatten()
                print("\nInput values:")
                print(x_input)
                
    
    
    # Reading input signal
    print("\n--------------------Forward Propagation--------------------")
    # Input --> Hidden
    h_input = np.dot(weights_in_hi, x_input)    
    h_act = relu(h_input)
    print("\nHidden Activation:")
    print(h_act)
    
    # Hidden --> Output
    o_input = np.dot(weights_hi_out, h_act)
    o_act = softmax(o_input)
    print("\nOutput Activation:")
    print(o_act)
    
    # Prediction
    y_pred = np.argmax(o_act)
    print(f"\nPredicted number: {y_pred}")
        
        
        
        
        
    # Saving input-hidden weights to a txt file
    print("\n--------------------Saving New Weights--------------------")
    np.savetxt(input_weights_filename, weights_in_hi, fmt="%.5f")
    
    
    # Want ot continue?
    wantToContinue = input("Do you want to try again this simple AI? (y/n) ")
    isActive = wantToContinue.lower() not in ["n", "no"]