import numpy as np
import pickle
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
import os
import torch 
import torch.nn as nn
from torch.autograd import Variable
my_path = '/home/peiguo/lorenz-cur/training_fig'
my_results_path_epoch100 = '/home/peiguo/lorenz-cur/results/epoch100' 
my_results_path_epoch200 = '/home/peiguo/lorenz-cur/results/epoch200' 
total_samples = 5000

with open('pickle/train_inputs.pickle', 'rb') as f:
    train_inputs = pickle.load(f)
with open('pickle/train_targets.pickle', 'rb') as f:
    train_targets = pickle.load(f)
train_inputs = Variable(torch.from_numpy(train_inputs).float()).view(total_samples, -1).cuda()
train_targets = Variable(torch.from_numpy(train_targets).float()).view(total_samples, -1).cuda()

with open('pickle/test_inputs.pickle', 'rb') as f:
    test_inputs = pickle.load(f)
with open('pickle/test_targets.pickle', 'rb') as f:
    test_targets = pickle.load(f)
test_inputs = Variable(torch.from_numpy(test_inputs).float()).view(total_samples, -1).cuda()
test_targets = Variable(torch.from_numpy(test_targets).float()).view(total_samples, -1).cuda()
batch_size = 8

# define the neural network
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.mlp = nn.Sequential(
        nn.Linear(12000, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 3),
        )
    def forward(self, x):
        para_pred = self.mlp(x)
        return para_pred
class MSRELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7
        self.weight = torch.Tensor([1,1/2,1]).cuda()
    def forward(self, input, target, size_average=True):
        return torch.mean(self.weight[None,:] * (input - target)**2) if size_average else torch.sum(self.weight[None,:] * (input - target)**2)
# define the model
model = MLP().cuda()
# difine the loss function
# using MSELOSS: 
#loss_fn = nn.MSELoss(reduction='sum').cuda()
# using mean squared relative error loss function(MSRELoss)
loss_fn = MSRELoss().cuda()

# define the hyperparameters
learning_rate = 1e-5
#learning_rate = 1e-3

EPOCH = 100

# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

error_all = []
error_sigma = []
error_rho = []
error_beta = []
error_all_lastep = []
error_sigma_lastep = []
error_rho_lastep = []
error_beta_lastep = []

rela_error_all = []
rela_error_sigma = []
rela_error_rho = []
rela_error_beta = []
rela_error_all_lastep = []
rela_error_sigma_lastep = []
rela_error_rho_lastep = []
rela_error_beta_lastep = []

predictions_lastep = []

for t in range(EPOCH):
    #training
    batch_num = int(np.ceil(total_samples / batch_size))
    shuffled = np.arange(total_samples)
    np.random.shuffle(shuffled)
    total_loss = 0
    all_data = 0
    for i in range(batch_num):
        start = i*batch_size
        end = min(total_samples, (i+1)*batch_size)
        inputs = train_inputs[shuffled[start:end]]
        targets_train = train_targets[shuffled[start:end]]
        preds_train = model(inputs)
        loss = loss_fn(preds_train, targets_train)
        total_loss += loss.item() * (end - start)
        all_data += end - start

        print("train epoch: {:4d}, iter: {:4d}, loss: {:5.2f}, avg loss: {:5.2f}".format(
            t, i, loss.item(), total_loss/all_data))
        
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    # testing
    total_loss = 0
    all_data = 0
    total_error_allpara = 0
    total_error_sigma = 0
    total_error_rho = 0
    total_error_beta = 0
    total_rela_error_allpara = 0
    total_rela_error_sigma = 0
    total_rela_error_rho = 0
    total_rela_error_beta = 0
    target_rho = []
    pred_rho = []

    for i in range(batch_num):
        start = i*batch_size
        end = min(total_samples, (i+1)*batch_size)
        inputs = test_inputs[start:end]
        targets = test_targets[start:end]
        preds = model(inputs)
        loss = loss_fn(preds, targets)

        total_loss += loss.item() * (end - start)
        all_data += end - start

        print("test epoch: {:4d}, iter: {:4d}, loss: {:5.2f}, "
                "avg loss: {:5.2f} target: {:5.2f} {:5.2f} {:5.2f} " 
                "pred: {:5.2f} {:5.2f} {:5.2f}".format(
            t, i, loss.item(), total_loss/all_data, 
            targets[0,0], targets[0,1], targets[0,2], 
            preds[0,0], preds[0,1], preds[0,2]))
        a = targets - preds
        error = torch.abs(a)
        rela_error = torch.abs(a/targets)

        if t == EPOCH - 1:
            target_rho.extend(targets[:,1])
            pred_rho.extend(preds[:,1])

            predictions_lastep.extend(preds.tolist())
            error_sigma_lastep.extend(error[:,0].tolist())
            error_rho_lastep.extend(error[:,1].tolist())
            error_beta_lastep.extend(error[:,2].tolist())
            error_all_lastep.extend((error[:,0]+error[:,1]+error[:,2]).tolist())

            rela_error_sigma_lastep.extend(rela_error[:,0].tolist())
            rela_error_rho_lastep.extend(rela_error[:,1].tolist())
            rela_error_beta_lastep.extend(rela_error[:,2].tolist())
            rela_error_all_lastep.extend((rela_error[:,0]+rela_error[:,1]+rela_error[:,2]).tolist())

        total_error_sigma += (torch.sum(error[:,0])).item()
        total_error_rho += (torch.sum(error[:,1])).item()
        total_error_beta += (torch.sum(error[:,2])).item()

        total_rela_error_sigma += (torch.sum(rela_error[:,0])).item()
        total_rela_error_rho += (torch.sum(rela_error[:,1])).item()
        total_rela_error_beta += (torch.sum(rela_error[:,2])).item()
    total_rela_error_allpara = total_rela_error_sigma + total_rela_error_rho + total_rela_error_beta
    total_error_allpara = total_error_sigma + total_error_rho + total_error_beta

    #visualize the result:
    mean_error_sigma = total_error_sigma / total_samples
    mean_error_rho = total_error_rho / total_samples
    mean_error_beta = total_error_beta / total_samples
    mean_error_allpara = total_error_allpara / total_samples
    error_all.append(mean_error_allpara)
    error_sigma.append(mean_error_sigma)
    error_rho.append(mean_error_rho)
    error_beta.append(mean_error_beta)
    #visualize the result relative error:
    mean_rela_error_sigma = total_rela_error_sigma / total_samples
    mean_rela_error_rho = total_rela_error_rho / total_samples
    mean_rela_error_beta = total_rela_error_beta / total_samples
    mean_rela_error_allpara = total_rela_error_allpara / total_samples
    rela_error_all.append(mean_rela_error_allpara)
    rela_error_sigma.append(mean_rela_error_sigma)
    rela_error_rho.append(mean_rela_error_rho)
    rela_error_beta.append(mean_rela_error_beta)

"""
print(error_all)
print(error_sigma)
print(error_rho)
print(error_beta)
print(rela_error_all)
print(rela_error_sigma)
print(rela_error_rho)
print(rela_error_beta)
"""

def f(state,t):
    x, y, z = state
    sigma, rho, beta = para
    return sigma * (y-x), x*(rho-z)-y, x*y-beta*z
predic_solu_lastep = []
for i in range(len(predictions_lastep)):
    para = predictions_lastep[i]
    t_seq = np.arange(0, 40, 0.01)
    state0 = [1.0, 1.0, 1.0]
    states = odeint(f, state0, t_seq)
    predic_solu_lastep.append(states)
with open('pickle/pred_solu_mlp.pickle', 'wb') as f:
    pickle.dump(predic_solu_lastep, f)
#Plot the results of all epochs
fig1, (ax1, ax2) = plt.subplots(1,2, sharex=True, figsize=(15, 8))
fig1.suptitle(' Error of Each Epoch')
epochs = np.arange(1, EPOCH+1)

ax1.plot(epochs, error_all, 'g^', label = 'all')
ax1.plot(epochs, error_sigma, 'bs', label = 'sigma')
ax1.plot(epochs, error_rho, 'r--', label = 'rho')
ax1.plot(epochs, error_beta, 'y*', label = 'beta')

ax1.legend()
ax1.set_title('error of each epoch')
ax1.set_xlabel('epoch')
ax1.set_ylabel('absolute error')

ax2.plot(epochs, rela_error_all, 'g^', label = 'all')
ax2.plot(epochs, rela_error_sigma, 'bs', label = 'sigma')
ax2.plot(epochs, rela_error_rho, 'r--', label = 'rho')
ax2.plot(epochs, rela_error_beta, 'y*', label = 'beta')

ax2.legend()
ax2.set_title('error of each epoch')
ax2.set_xlabel('epoch')
ax2.set_ylabel('relative error')

#Plot the absolute errors of last epoch
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 8))
fig2.suptitle('Absolute Error of Each Sample in The Lase Epoch')

samples = np.arange(1, total_samples+1)

ax1.plot(samples, error_all_lastep, 'g^', label = 'all')
ax2.plot(samples, error_sigma_lastep, 'bs', label = 'sigma')
ax3.plot(samples, error_rho_lastep, 'r--', label = 'rho')
ax4.plot(samples, error_beta_lastep, 'y*', label = 'beta')

ax1.legend()
ax1.set_title('absolute error of all')
ax1.set_xlabel('samples')
ax1.set_ylabel('absolute error')

ax2.legend()
ax2.set_title('absolute error of sigma')
ax2.set_xlabel('samples')
ax2.set_ylabel('absolute error')

ax3.legend()
ax3.set_title('absolute error of rho')
ax3.set_xlabel('samples')
ax3.set_ylabel('absolute error')

ax4.legend()
ax4.set_title('absolute error of beta')
ax4.set_xlabel('samples')
ax4.set_ylabel('absolute error')


#Plot the relative errors of last epoch
fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 8))
fig3.suptitle('Relative Error of Each Sample in The Lase Epoch')
samples = np.arange(1, total_samples+1)

ax1.plot(samples, rela_error_all_lastep, 'g^', label = 'all')
ax2.plot(samples, rela_error_sigma_lastep, 'bs', label = 'sigma')
ax3.plot(samples, rela_error_rho_lastep, 'r--', label = 'rho')
ax4.plot(samples, rela_error_beta_lastep, 'y*', label = 'beta')

ax1.legend()
ax1.set_title('relative error of all')
ax1.set_xlabel('samples')
ax1.set_ylabel('relative error')

ax2.legend()
ax2.set_title('relative error of sigma')
ax2.set_xlabel('samples')
ax2.set_ylabel('relative error')

ax3.legend()
ax3.set_title('relative error of rho')
ax3.set_xlabel('samples')
ax3.set_ylabel('relative error')

ax4.legend()
ax4.set_title('relative error of beta')
ax4.set_xlabel('samples')
ax4.set_ylabel('relative error')


fig1.savefig(os.path.join(my_results_path_epoch100, 'all_epochs_mse'))
fig2.savefig(os.path.join(my_results_path_epoch100, 'last_epochs_abo_mse'))
fig3.savefig(os.path.join(my_results_path_epoch100, 'last_epochs_rela_mse'))

"""
plt.figure(1)
x_axis1 = np.arange(1,EPOCH+1)
print(x_axis1)
plt.plot(x_axis1, error_all, 'r--', x_axis1, error_sigma, 'bs', x_axis1, error_rho, 'g^', x_axis1, error_beta, 'y*')
label = ['all', 'sigma', 'rho', 'beta']
plt.legend(label, loc='upper right')
my_results_file1 = 'all_epoch_par02'
plt.savefig(os.path.join(my_results_path_abso, my_results_file1))
plt.figure(2)
x_axis3 = np.arange(1,EPOCH+1)
plt.plot(x_axis3, rela_error_all, 'r--', x_axis3, rela_error_sigma, 'bs', x_axis3, rela_error_rho, 'g^', x_axis3, rela_error_beta, 'y*')
label = ['all', 'sigma', 'rho', 'beta']
plt.legend(label, loc='upper right')
my_results_file3 = 'rela_all_epoch_par02'
plt.savefig(os.path.join(my_results_path_rela, my_results_file3))

plt.figure(3)
x_axis0 = np.arange(1,total_samples + 1)
plt.plot(x_axis0, target_rho, 'bs', x_axis0, pred_rho, 'g^')
label = [ 'target', 'predict']
plt.legend(label, loc='upper right')
my_results_file0 = 'rho_last_epoch_par02'
plt.savefig(os.path.join(my_results_path_abso, my_results_file0))

plt.figure(4)
x_axis2 = np.arange(1,total_samples + 1)
plt.plot(x_axis2, error_all_lastep, 'r--', x_axis2, error_sigma_lastep, 'bs', x_axis2, error_rho_lastep, 'g^', x_axis2, error_beta_lastep, 'y*')
label = ['all', 'sigma', 'rho', 'beta']
plt.legend(label, loc='upper right')
my_results_file2 = 'last_epoch_par02'
plt.savefig(os.path.join(my_results_path_abso, my_results_file2))

plt.figure(5)
x_axis4 = np.arange(1,total_samples + 1)
plt.plot(x_axis4, rela_error_all_lastep, 'r--', x_axis4, rela_error_sigma_lastep, 'bs', x_axis4, rela_error_rho_lastep, 'g^', x_axis4, rela_error_beta_lastep, 'y*')
label = ['all', 'sigma', 'rho', 'beta']
plt.legend(label, loc='upper right')
my_results_file4 = 'rela_last_epoch_par02'
plt.savefig(os.path.join(my_results_path_rela, my_results_file4))
"""
# Plot the relative error

