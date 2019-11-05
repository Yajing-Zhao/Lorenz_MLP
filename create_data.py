import numpy as np
import pickle
from scipy.integrate import odeint
"""
This file will create the training data sets and the testing data sets. The training data set contains two datasets, one is the targets data of size TOTAL_SAMPLES**3 which are the values of TOTAL_SAMPLES groups of randomly drawed parameters [sigma, rho, beta] , and it is saved as train_inputs.pickle. The other data set is the inpus data set of size TOTAL_SAMPLES*4000*3 which are the solutions of the parameters in the targets data set, and it is saved as train_inputs.pickle. Similar creation for the testing data set. 
"""
TOTAL_SAMPLES =5000 # number of samoles

# the function to create training data set
def dataset(total_samples):
    def f(state,t):
        x, y, z = state
        sigma, rho, beta = para
        return sigma * (y-x), x*(rho-z)-y, x*y-beta*z

    data = []
    targets = []
    for i in range(total_samples):
        para = [np.random.uniform(0,20), np.random.uniform(10,30), np.random.uniform(0,10)]
        t_seq = np.arange(0, 40, 0.01)
        state0 = [1.0, 1.0, 1.0]
        states = odeint(f, state0, t_seq)
        data.append(states)
        targets.append(para)
    return np.array(data), np.array(targets)

train_inputs, train_targets = dataset(TOTAL_SAMPLES)

# Save training data set as pickle files.
with open('train_inputs.pickle', 'wb') as f:
    pickle.dump(train_inputs, f)
with open('train_targets.pickle', 'wb') as f:
    pickle.dump(train_targets, f)

# the function to create training data set
def test_dataset(total_samples):
    def f(state,t):
        x, y, z = state
        sigma, rho, beta = para
        return sigma * (y-x), x*(rho-z)-y, x*y-beta*z

    data = []
    targets = []
    for i in range(total_samples):
        para = [np.random.uniform(0,20), np.random.uniform(10,30), np.random.uniform(0,10)]
        t_seq = np.arange(0, 40, 0.01)
        state0 = [1.0, 1.0, 1.0]
        states = odeint(f, state0, t_seq)
        data.append(states)
        targets.append(para)
    return np.array(data), np.array(targets)

test_inputs, test_targets = test_dataset(TOTAL_SAMPLES) 
# Save training data set as pickle files.
with open('test_inputs.pickle', 'wb') as f:
    pickle.dump(test_inputs, f)
with open('test_targets.pickle', 'wb') as f:
    pickle.dump(test_targets, f)
