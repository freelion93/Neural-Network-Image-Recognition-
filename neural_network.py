import numpy as np
import mnist
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(8)

if __name__ == "__main__":

    #load the data
    train_data_size = 6000 # Only a subset of training data is utilized, dont change for submission
    train_data = mnist.load_data("train-images-idx3-ubyte.gz")[0:train_data_size]
    train_labels = mnist.load_labels("train-labels-idx1-ubyte.gz")[0:train_data_size]
    test_data = mnist.load_data("t10k-images-idx3-ubyte.gz")
    test_labels = mnist.load_labels("t10k-labels-idx1-ubyte.gz")
    
    #normalizing by RGB to grayscale
    X_train = train_data / 255
    X_test = test_data /255
    
    #transforming examples by flatting
    X_train = X_train.reshape(X_train.shape[0], -1).T 
    X_test = X_test.reshape(X_test.shape[0], -1).T 
    
    #sizes of the sets
    items = 10
    m = train_data_size
    m_test = test_data.shape[0]
        
    #we want our labels matrices to be size of (items x examples in set), e.g. 10 x 6000
    def yshape (Y):
        
        labels = Y.shape[0]
        Y = Y.reshape(1, labels)
        
        Y_shaped = np.eye(items)[Y.astype('int32')]
        Y_shaped = Y_shaped.T.reshape(items, labels)
        
        return Y_shaped
    
    Y_train = yshape(train_labels)
    Y_test =  yshape(test_labels)
        
    def layer_size(X,Y):
        n_x = X.shape[0] # size of input layer
        n_h = 160        # size of hidden layer
        n_y = Y.shape[0] # size of output layer
        return (n_x, n_h, n_y)
    
    #initializing the parameters using a normal distribution with µ = 0 and σ = 0.05
    def initialize_parameters(n_x, n_h, n_y, m, s):
        
        W0 = np.random.normal(m, s,(n_h, n_x))
        b0 = np.random.normal(m, s,(n_h, 1))
        W1 = np.random.normal(m, s,(n_y, n_h))
        b1 = np.random.normal(m, s,(n_y, 1))
        
        parameters = {"W0": W0,
                      "b0": b0,
                      "W1": W1,
                      "b1": b1}
        
        return parameters
    
    def sigmoid(z):
        s = 1 / (1 + np.exp(-z))
        return s
    
    def softmax(z):
        s = np.exp(z) / np.sum(np.exp(z), axis=0)
        return s
    
    #forward propagation
    def forw_prop(X, W0, W1, b0, b1):
        
        Z1 = np.dot(W0,X) + b0
        #I'could use  np.tahn there because of tanh has a steeper slope so it has a higher derivative than sigmoid, which will make learning faster.
        #https://www.quora.com/What-are-the-benefits-of-a-tanh-activation-function-over-a-standard-sigmoid-activation-function-for-artificial-neural-nets-and-vice-versa
        A1 = sigmoid(Z1) 
        Z2 = np.dot(W1,A1) + b1
        A2 = softmax(Z2)
        
        forw_params = {"Z1": Z1,
                       "A1": A1,
                       "Z2": Z2,
                       "A2": A2}
        
        return forw_params
    
    #back propagation
    def back_prop(W0, W1, A1, A2, Z1, X, Y):
   
        m = X.shape[1]
        
        dZ2 = A2-Y
        dW1 = (1./m) * np.dot(dZ2, A1.T)
        db1 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)
    
        dA1 = np.dot(W1.T, dZ2)
        dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
        dW0 = (1./m) * np.dot(dZ1, X.T)
        db0 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)
        
        grads = {"dW0": dW0,
                 "db0": db0,
                 "dW1": dW1,
                 "db1": db1}
        
        return grads
    
    def update_parameters(W0, W1, b0, b1, dW0, dW1, db0, db1, learn_rate):
    
        W0 = W0-learn_rate*dW0
        b0 = b0-learn_rate*db0
        W1 = W1-learn_rate*dW1
        b1 = b1-learn_rate*db1
    
        parameters = {"W0": W0,
                      "b0": b0,
                      "W1": W1,
                      "b1": b1}
    
        return parameters
        
    def compute_cost(Y, A2):
        csum = np.sum(np.multiply(Y, np.log(A2)))
        m = Y.shape[1]
        cost = -(1/m) * csum
        return cost
    
    def model (X, Y, learning_rate, iterations, m, s):
        n_x = layer_size(X, Y)[0]
        n_h = layer_size(X, Y)[1]
        n_y = layer_size(X, Y)[2]
        
        parameters = initialize_parameters(n_x, n_h, n_y, m, s)
        W0 = parameters["W0"]
        b0 = parameters["b0"]
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        
        for i in range(iterations):
            
            forw_params = forw_prop(X, W0, W1, b0, b1)
            A1 = forw_params["A1"]
            A2 = forw_params["A2"]
            Z1 = forw_params["Z1"]

            cost = compute_cost(Y, A2)
        
            grads = back_prop(W0, W1, A1, A2, Z1, X, Y)
            dW0 = grads["dW0"]
            dW1 = grads["dW1"]
            db0 = grads["db0"]
            db1 = grads["db1"]
            
            up_parameters = update_parameters(W0, W1, b0, b1, dW0, dW1, db0, db1, learning_rate)
            W0 = up_parameters["W0"]
            b0 = up_parameters["b0"]
            W1 = up_parameters["W1"]
            b1 = up_parameters["b1"]
    
            if(i % 10 == 0 and i != 100):
                print("Cost on "+str(i)+"-th iteration: "+str(cost))
            
        print("Final cost:", cost)
        
        train_result = {"W0": W0,
                      "b0": b0,
                      "W1": W1,
                      "b1": b1}
        
        return train_result
        
    train_result = model(X_train, Y_train, 0.35, 100, 0, 0.05)
    
    def test_on_test (X, Y, train_result):
    
        print ("")
        print ("***********Testing on test dataset***********")
        W0 = train_result["W0"]
        b0 = train_result["b0"]
        W1 = train_result["W1"]
        b1 = train_result["b1"]
        
        #repeating forward propogation
        Z1 = np.dot(W0, X) + b0
        A1 = sigmoid(Z1)
        Z2 = np.dot(W1, A1) + b1
        A2 = softmax(Z2)
        
        predictions = np.argmax(A2, axis=0)
        labels = np.argmax(Y, axis=0)
        
        print(confusion_matrix(predictions, labels))
        print ("")
        print(classification_report(predictions, labels))
    
    test_on_test(X_test, Y_test, train_result)
