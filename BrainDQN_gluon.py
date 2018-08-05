# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------
import os

import mxnet as mx 
import numpy as np 
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import random
from collections import deque 
import mxnet.autograd as autograd
from time import time

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0#0.001 # final value of epsilon
INITIAL_EPSILON = 0#0.01 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 100
ctx=mx.gpu()


class QNET(gluon.Block):
    def __init__(self,classNum,ctx,verbose=False, **kwargs):
        super(QNET,self).__init__(**kwargs)
        with self.name_scope():
            self.layers = nn.Sequential()
            self.layers.add( nn.Conv2D(channels=32,kernel_size=8,strides=4,padding=2,activation='relu') )
            self.layers.add( nn.MaxPool2D(pool_size=(2,2),strides=(2,2) ) )
            self.layers.add( nn.Conv2D(channels=64,kernel_size=4,strides=2,padding=1,activation='relu'))
            self.layers.add( nn.Conv2D(channels=64,kernel_size=3,strides=1,padding=1,activation='relu'))
            self.layers.add( nn.Flatten())
            self.layers.add( nn.Dense(512,activation='relu') )
            self.layers.add( nn.Dense(classNum))
        self.ctx = ctx
        self.initialize(ctx=ctx)
        #self.hybridize()
        self.trainer = gluon.Trainer(self.collect_params(),"adam",{"learning_rate":0.0002,"wd":0,"beta1":0.5})
        self.loss_l2 = gluon.loss.L2Loss()
        self.output = None
        return

    def forward(self,X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def get_outputs(self):
        return self.output

    def predict(self,X_):
        X = X_.as_in_context(self.ctx)
        self.output = self.forward(X)
        return self.output

    def fit(self, X_,Y_):
        data = X_[0].as_in_context(self.ctx)
        action = X_[1].as_in_context(self.ctx)
        Y = Y_[0].as_in_context(self.ctx)
        with autograd.record():
            Y1 = self.forward(data)
            #print Y1.asnumpy().shape, action.asnumpy().shape
            #print (action * Y1)
            Y1 = (action * Y1).sum(axis=1) #action: (0,1) or (1,0), Y1: reward
            #print Y1.asnumpy().shape
            loss = self.loss_l2(Y1,Y)
        loss.backward()
        self.trainer.step(data.shape[0])
        return


def dataPrep(data):
    if data.shape[2]>1:
        mean = np.array([128, 128, 128,128])
        reshaped_mean = mean.reshape(1, 1, 4)
    else:
        mean=np.array([128])
        reshaped_mean = mean.reshape(1, 1, 1)
    img = np.array(data, dtype=np.float32)
    data = data - reshaped_mean
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 1, 2)
    data = np.expand_dims(data, axis=0)
    return data

class BrainDQN:
    def __init__(self,actions,param_file=None):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions

        self.trainCost = []

        self.target = QNET(actions,ctx)

        self.Qnet = QNET(actions,ctx)
        if param_file!=None:
            self.Qnet.load_params(param_file)
            self.target.load_params(param_file)
            #self.Qnet.hybridize()
            #self.target.hybridize()
            print 'load pretrained model from {}'.format(param_file)
            #self.Qnet.hybridize()
        #self.copyTargetQNetwork() 
        # saving and loading networks
            
    def copyTargetQNetwork(self):
        root = os.path.split(os.path.realpath(__file__))[0] 
        tmppath = os.path.join(root,'copy.params')
        self.Qnet.save_params(tmppath)
        self.target.load_params(tmppath)
        print 'time to copy'

    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory,BATCH_SIZE)
        state_batch = np.squeeze([data[0] for data in minibatch])
        action_batch =  np.squeeze([data[1] for data in minibatch])
        reward_batch =  np.squeeze([data[2] for data in minibatch])
        nextState_batch =  [data[3] for data in minibatch]

        # Step 2: calculate y 
        y_batch = np.zeros((BATCH_SIZE,))
        Qvalue=[]
        for i in range(BATCH_SIZE):
            #self.target.forward(mx.io.DataBatch([mx.nd.array(nextState_batch[i],ctx)],[]))
            self.target.predict(mx.nd.array(nextState_batch[i],ctx))
            Qvalue.append(self.target.get_outputs()[0].asnumpy())
        Qvalue_batch=np.squeeze(Qvalue)
        terminal=np.squeeze([data[4] for data in minibatch])
        y_batch[:]=reward_batch
        if (terminal==False).shape[0]>0:
            y_batch[terminal==False]+= (GAMMA * np.max(Qvalue_batch,axis=1))[terminal==False]

        #print state_batch.shape, action_batch.shape,  y_batch.shape
        t0 = time()
        self.Qnet.fit([mx.nd.array(state_batch,ctx),mx.nd.array(action_batch,ctx)],[mx.nd.array(y_batch,ctx)])
        self.trainCost.append(time() - t0)
        #self.Qnet.forward(mx.io.DataBatch([mx.nd.array(state_batch,ctx),mx.nd.array(action_batch,ctx)],[mx.nd.array(y_batch,ctx)]),is_train=True)
        #self.Qnet.backward()
        #self.Qnet.update()

        # save network every 1000 iteration
        if self.timeStep % 100 == 0:
            print "train cost {:.4}s".format( np.asarray(self.trainCost).mean() )
            self.Qnet.save_params('saved_networks/network-dqn_gluon%04d.params'%(self.timeStep))

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()


    def setInitState(self,observation):
        temp=dataPrep(np.stack((observation, observation, observation, observation), axis = 2))
        self.currentState = temp
    
    def setPerception(self,nextObservation,action,reward,terminal):
        #newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)

        newState = np.append(self.currentState[:,1:,:,:],dataPrep(nextObservation),axis = 1)
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network

            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if self.timeStep % 500 == 0:
            print "TIMESTEP", self.timeStep, "/ STATE", state, \
            "/ EPSILON", self.epsilon

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        #self.target.forward(mx.io.DataBatch([mx.nd.array(self.currentState,ctx)],[]))
        self.target.predict( mx.nd.array(self.currentState,ctx)  )

        QValue=np.squeeze(self.target.get_outputs()[0].asnumpy())
        #print 'current QValue {}'.format(QValue)
        action = np.zeros(self.actions)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1 # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        return action
