import numpy as np


class Neurons:

    def __init__(self,
    n_input_neurons,
    n_hidden_neurons,
    n_output_neurons,
    weights_in_hi_filename,
    weights_hi_out_filename,
    x_input):
        
        self.n_input_neurons = n_input_neurons
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output_neurons = n_output_neurons
        self.weights_in_hi_filename = weights_in_hi_filename
        self.weights_hi_out_filename = weights_hi_out_filename
        self.x_input = x_input


        print("\n--------------------Reading Existing Weights--------------------")
        # Reading / generating input weights
        try:
            with open(self.weights_in_hi_filename, "r") as weight_file:
                weights_in_hi = np.loadtxt(weight_file)
            
            if weights_in_hi.shape != (self.n_hidden_neurons, self.n_input_neurons):
                weights_in_hi = self.generate_gaussian_matrix(self.n_hidden_neurons, self.n_input_neurons)
                
            print("Input-Hidden weights stored:")
            print(weights_in_hi)
        except Exception as e:
            weights_in_hi = self.generate_gaussian_matrix(self.n_hidden_neurons, self.n_input_neurons)
            print("Input-Hidden weights generated. Here's the error: ${e}")
            print(weights_in_hi)
        
        # Reading / generating hidden weights
        try:
            with open(self.weights_hi_out_filename, "r") as weight_file:
                weights_hi_out = np.loadtxt(weight_file)
            
            if weights_hi_out.shape != (self.n_output_neurons, self.n_hidden_neurons):
                weights_hi_out = self.generate_gaussian_matrix(self.n_output_neurons, self.n_hidden_neurons)
                
            print("Hidden-Output weights stored:")
            print(weights_hi_out)
        except Exception as e:
            weights_hi_out = self.generate_gaussian_matrix(self.n_output_neurons, self.n_hidden_neurons)
            print("Hidden-Output weights generated. Here's the error: ${e}")
            print(weights_hi_out)
        

        print("\n--------------------Forward Propagation--------------------")
        # Input --> Hidden
        h_input = np.dot(weights_in_hi, self.x_input)    
        h_act = self.relu(h_input)
        print("\nHidden Activation:")
        print(h_act)
        
        # Hidden --> Output
        o_input = np.dot(weights_hi_out, h_act)
        o_act = self.softmax(o_input)
        print("\nOutput Activation:")
        print(o_act)
        
        # Prediction
        y_pred = np.argmax(o_act)
        print(f"\nPredicted number: {y_pred}")
        y = int(input("\nEnter the correct value: "))
        target = self.one_hot_encode(y)
        print(o_act)
        print(target)


        print("\n--------------------Backward Propagation--------------------")
        # Compute the error
        loss = self.cross_entropy_loss(o_act, target)
        output_error = loss - target          # Derivata di Cross-Entropy
        delta_hi_out = np.outer(output_error, h_act)
        print("\nGradient for Hidden-Output Weights:")
        print(delta_hi_out)

        hidden_error = np.dot(weights_hi_out.T, output_error) * (h_input > 0)
        delta_in_hi = np.outer(hidden_error, self.x_input)
        print("\nGradient for Input-Hidden Weights:")
        print(delta_in_hi)



    

    ########################################
    # USEFUL FUNCTIONS
    ########################################
    def generate_gaussian_matrix(self, n, m, mean=0, st_dev=0.4):
        """Genera una matrice casuale con distribuzione gaussiana"""
        random_matrix = np.random.normal(loc=mean, scale=st_dev, size=(n, m))
        return random_matrix
    
    def relu(self, x):
        """Funzione di attivazione ReLU"""
        return np.maximum(0, x)
    
    def softmax(self, x):
        """Funzione di attivazione softmax per il livello di output"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def one_hot_encode(self, number, num_classes=10):
        """Converte un numero in un array one-hot (2 -> [0, 0, 1, 0, ...])"""
        one_hot = np.zeros(num_classes)
        one_hot[number] = 1
        return one_hot
    
    def cross_entropy_loss(self, predicted, target):
        """Calcola la loss usando Cross-Entropy"""
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)  # Limita il range [epsilon, 1-epsilon]
        return -np.sum(target * np.log(predicted))