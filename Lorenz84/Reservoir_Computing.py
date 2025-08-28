from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt


class reservoir():
  def __init__(self, trainLen=int, validLen=int, initLen=int, inSize=int, outSize = int,
                 resSize = int, a = float, seed = int, 
                 reg=float):
    self.trainLen  = trainLen    # length for training
    self.validLen  = validLen    # length for validation
    self.initLen   = initLen      # length for initialization (spinup is necessaary to form the reservoir state)
    self.inSize    = inSize       # input variable dimension
    self.outSize   = outSize      # output variable dimension
    self.resSize   = resSize      # reservoir size (how many past states are used)
    self.a         = a            # leaking rate
    self.seed      = seed         # experiment number (any number bigger than 0, if seed=0, randomly initialized)
    self.reg       = reg          # regularization intensity (regid by default)


  def training(self, data): # data should have a dimension of [time X variable]
    if self.seed==0:
      np.random.seed()
    else:
      np.random.seed(self.seed)

    # preparing regression coefficient
    Win   = (np.random.rand(self.resSize,1+self.inSize) - 0.5) * 1
    W     = np.random.rand(self.resSize,self.resSize) - 0.5
    # normalizing and setting spectral radius (correct, slow):
    #print('Computing spectral radius...')
    rhoW  = max(abs(linalg.eig(W)[0]))
    #print('done.')
    W    *= 1.25 / rhoW

    # allocated memory for the design matrix (collected states including the input from current step and past step)
    X     = np.zeros((1+self.inSize+self.resSize,self.trainLen-self.initLen))
    # set the corresponding target matrix directly
    Yt    = data[None,self.initLen+1:self.trainLen+1]

    # rewrite this a batch 
    # run the reservoir with the data and collect X
    x = np.zeros((self.resSize,1))  # input from current time step
    for t in range(self.trainLen):
        u    = data[t,:]
        u    = np.reshape(u,[self.inSize,1])
        unit = np.ones((1,1))
        x    = (1-self.a)*x + self.a*np.tanh( np.dot( Win, np.vstack((1,u))) + np.dot( W, x ) )

        # after initialization step, stacking the input from current step and latent state together
        if t >= self.initLen:
            X[:,t-self.initLen] = np.vstack((1,u,x))[:,0]

    # train the output layer with Normal Equation method (+ rigid regularization)
    Wout = linalg.solve( np.dot(X,X.T) + self.reg*np.eye(1+self.inSize+self.resSize), np.dot(X,Yt.T) ).T
    #return all variables at t=trainLen
    return x, Win, W, Wout

  def training_validation(self,data):
    # training the model
    x, Win, W, Wout = self.training(data)

    # run the trained ESN in a generative mode. no need to initialize here,
    # because x is initialized with training data and we continue from there. (all parameters are frozen)
    Y        = np.zeros((self.outSize,self.validLen)) # the forecast value
    Y[:,0]   = data[self.trainLen,:]  
    u        = data[self.trainLen,:]                 # the input for steps at trainLen
    u        = np.reshape(u,[self.inSize,1])
    unit     = np.ones((1,1))
    x_record = x.copy()
    for t in range(self.validLen-1):
        x        = (1-self.a)*x + self.a*np.tanh( np.dot( Win, np.vstack((unit,u)) ) + np.dot( W, x ) )
        y        = np.dot( Wout, np.vstack((1,u,x)) )
        Y[:,t+1] = np.squeeze(y)
        # generative mode:
        u        = y                                   # using the forecast value as the next time step input
        u        = np.reshape(u,[self.inSize,1])
        ## this would be a predictive mode:
        #u = data[trainLen+t+1]


    # targeting specific lead 
    errorLen = self.validLen
    #print(data[self.trainLen+errorLen].shape)
    mse      = sum( np.square( data[self.trainLen:self.trainLen+errorLen] - np.transpose(Y[:,0:errorLen]) ) ) / errorLen
    
    return Y, x_record, Win, W, Wout, mse

  
  def prediction(self,data,Win,W,Wout,ensemble_size=1000,forecast_time=10):
    dim_t, dim_var = data.shape
    x              = np.zeros((self.resSize,1))
    Y              = np.zeros((self.resSize,forecast_time))
    unit           = np.ones((1,1))
    # initialization to generate x 
    for t in range(dim_t-1):
        u    = data[t,:]
        u    = np.reshape(u,[self.inSize,1])
        unit = np.ones((1,1))
        x    = (1-self.a)*x + self.a*np.tanh( np.dot( Win, np.vstack((1,u))) + np.dot( W, x ))

    data_ensemble      = np.zeros((self.inSize,ensemble_size))
    data_ensemble[:,0] = data[-1,:]       # the first member has no perturbation   
    for i in range(self.inSize):       # other members have perturbation
        data_ensemble[i,1:] = data[-1,i]+np.random.normal(loc=0.0, scale=0.05, size=ensemble_size-1)
        
    u                  = data_ensemble
    u                  = np.reshape(u,[self.inSize,ensemble_size])
    x                  = np.repeat(x, ensemble_size, axis=1)      # repeat the hidden state (identical for all ensemble)
    Y                  = np.zeros((self.inSize,ensemble_size,forecast_time))  # array for saving forecast results
    Y[:,:,0]           = u
    unit               = np.ones((1,ensemble_size))
    for t in range(forecast_time-1): 
        x = (1-self.a)*x + self.a*np.tanh( np.dot( Win, np.vstack((unit,u)) ) + np.dot( W, x ) )
        y = np.dot( Wout, np.vstack((unit,u,x)))
        Y[:,:,t+1] = y
        # generative mode:
        u = y
        u = np.reshape(u,[self.inSize,ensemble_size])
        
    return Y