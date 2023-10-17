
"""
@author: Ameya Jagtap
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):
        
        self.lb = lb
        self.ub = ub
    
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.u = u
        
        self.layers = layers
        self.nu = nu
        
        # Initialize NNs
        #self.weights, self.biases, self.a = self.initialize_NN(layers)
        self.weights, self.biases, self.a = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])        
                
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf) 
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)  
        
        ############################################################################           

        self.loss = tf.reduce_mean(tf.square(self.f_pred)) + tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) 
                   

        
        self.optimizer_Adam = tf.train.AdamOptimizer(0.0008)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)    
    
        init = tf.global_variables_initializer()
        self.sess.run(init)

                
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            
            weights.append(W)
            biases.append(b)
        
        #a = tf.Variable(0.1, dtype=tf.float32)
        return weights, biases#, a
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):#, a):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))# 10*a*
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)#, self.a)
        return u
    
    def net_f(self, x,t):
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u*u_x - self.nu*u_xx
        
        return f

        
    def train(self,nIter):

        MSE_history=[]
        #a_history=[]

        for it in range(nIter):
            
            tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
            self.sess.run(self.train_op_Adam,tf_dict)
            
            
            if it %1 == 0:
                #elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                #a_value    = self.sess.run(self.a, tf_dict)
                #print('It: %d, Loss: %.3e, a_value: %.3e' % 
                #      (it, loss_value, a_value))
                print('It: %d, Loss: %.3e' % 
                      (it, loss_value))
                #start_time = time.time()
                MSE_history.append(loss_value)
                #a_history.append(a_value)

      
        return MSE_history
                                                                                                                                                
    
    def predict(self, X_star):
                
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})  
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
               
        return u_star, f_star
    
if __name__ == "__main__": 
    nu = 0.01/np.pi
    noise = 0.0        

    N_u = 100
    N_f = 8000
    layers = [2, 20, 20, 20, 20, 20, 20, 1]
    
    data = scipy.io.loadmat('DATA/burgers_shock.mat')
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
        
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]
    
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]
        
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)
    
    start_time = time.time()                
    MSE_hist = model.train(200)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    u_pred, f_pred = model.predict(X_star)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))                     

    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)
    
    
#     ######################################################################
#     ############################# Plotting ###############################
#     ###################################################################### 
#%%   
    fig = plt.figure(1)
    plt.plot(MSE_hist,  'r-', linewidth = 1) 
    plt.xlabel('$\#$ iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    plt.savefig('H2D_MSEhistory.pdf') 

    fig, ax = plt.subplots()
    CS = ax.contourf(X, T,Exact, levels =100, cmap='seismic')
    ax.set_title('Exact')
    plt.xlabel('X', fontsize=14)
    plt.ylabel('T', fontsize=14)
    cbar = fig.colorbar(CS)
    fig.tight_layout()
    
    
    fig, ax = plt.subplots()
    CS = ax.contourf(X, T, U_pred, levels =100, cmap='seismic')
    ax.set_title('PINN Prediction')
    plt.xlabel('X', fontsize=14)
    plt.ylabel('T', fontsize=14)
    cbar = fig.colorbar(CS)
    fig.tight_layout()
  
        
    fig, ax = plt.subplots()
    CS = ax.contourf(X, T,abs(Exact-U_pred), levels =100, cmap='seismic')
    #ax.clabel(CS, inline=True, fontsize=14)
    ax.set_title('Point-wise Error')
    plt.xlabel('X', fontsize=14)
    plt.ylabel('T', fontsize=14)
    cbar = fig.colorbar(CS)
    fig.tight_layout()

