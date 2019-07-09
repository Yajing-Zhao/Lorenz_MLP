import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
import os
import torch 
import torch.nn as nn
from torch.autograd import Variable
my_path = '/home/peiguo/Lorenz_MLP/training_fig'
my_results_path = '/home/peiguo/Lorenz_MLP/results' 
total_samples = 5000

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
        """
        fig1 = plt.figure()
        ax =fig1.gca(projection = '3d')
        ax.plot(states[:,0], states[:,1], states[:,2])
        my_file = 'training'+str(i)
        plt.savefig(os.path.join(my_path, my_file))
        """
     #   print(states.shape())
        data.append(states)
        targets.append(para)

    data = Variable(torch.from_numpy(np.array(data)).float()).view(total_samples, -1)
    targets = Variable(torch.from_numpy(np.array(targets)).float()).view(total_samples, -1)

    return data.cuda(), targets.cuda()

train_inputs, train_targets = dataset(total_samples)
test_inputs, test_targets = dataset(total_samples)
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
        self.weight = torch.Tensor([.1,1,.1]).cuda()
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

for t in range(EPOCH):
    #training
    batch_num = int(np.ceil(total_samples // batch_size))
    shuffled = np.arange(total_samples)
    np.random.shuffle(shuffled)
    total_loss = 0
    all_data = 0
    for i in range(batch_num):
        start = i*batch_size
        end = min(total_samples, (i+1)*batch_size)
        inputs = train_inputs[shuffled[start:end]]
        targets = train_targets[shuffled[start:end]]
        preds = model(inputs)
        loss = loss_fn(preds, targets)
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
        print(error)
        print(targets)
        print(rela_error)



        if t == EPOCH - 1:
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
x_axis = np.arange(1,EPOCH+1)
plt.plot(x_axis, error_all, 'r--', x_axis, error_sigma, 'bs', x_axis, error_rho, 'g^', x_axis, error_beta, 'y*')
label = ['all', 'sigma', 'rho', 'beta']
plt.legend(label, loc='upper right')
my_results_file1 = 'all_epoch_mse_par4'
plt.savefig(os.path.join(my_results_path, my_results_file1))

x_axis2 = np.arange(1,total_samples + 1)
plt.plot(x_axis2, error_all_lastep, 'r--', x_axis2, error_sigma_lastep, 'bs', x_axis2, error_rho_lastep, 'g^', x_axis2, error_beta_lastep, 'y*')
label = ['all', 'sigma', 'rho', 'beta']
plt.legend(label, loc='upper right')
my_results_file2 = 'last_epoch_mse_par4'
plt.savefig(os.path.join(my_results_path, my_results_file2))

x_axis3 = np.arange(1,EPOCH+1)
plt.plot(x_axis3, rela_error_all, 'r--', x_axis3, rela_error_sigma, 'bs', x_axis3, rela_error_rho, 'g^', x_axis3, rela_error_beta, 'y*')
label = ['all', 'sigma', 'rho', 'beta']
plt.legend(label, loc='upper right')
my_results_file2 = 'rela_all_epoch_mse_par4'
plt.savefig(os.path.join(my_results_path, my_results_file2))

x_axis4 = np.arange(1,total_samples + 1)
plt.plot(x_axis4, rela_error_all_lastep, 'r--', x_axis4, rela_error_sigma_lastep, 'bs', x_axis4, rela_error_rho_lastep, 'g^', x_axis4, rela_error_beta_lastep, 'y*')
label = ['all', 'sigma', 'rho', 'beta']
plt.legend(label, loc='upper right')
my_results_file3 = 'rela_last_epoch_mse_par4'
plt.savefig(os.path.join(my_results_path, my_results_file3))

# Plot the relative error

