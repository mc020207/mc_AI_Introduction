import numpy as np
import os
import random
from matplotlib import pyplot as plt


class BatchGD():
    def __init__(self, init_lr, model):
        self.init_lr=init_lr
        self.model=model

    def step(self):
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class DisLoss():
    def __init__(self, model):
        self.predicts = None
        self.labels = None
        self.num = None
        self.model = model

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]
        loss = (predicts-labels)*(predicts-labels)
        return np.sum(loss) / (self.num*2)

    def backward(self):
        loss_grad_predicts = -(self.predicts-self.labels)/self.num
        self.model.backward(loss_grad_predicts)

class Linear():
    def __init__(self, input_size, output_size, name, weight_init=np.random.normal, bias_init=np.zeros):
        self.params = {}
        self.params['W'] = weight_init(size=(input_size, output_size))
        self.params['b'] = bias_init((1, output_size))
        self.inputs = None
        self.grads = {}
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        outputs = np.matmul(self.inputs, self.params['W']) + self.params['b']
        return outputs

    def backward(self, grads):
        self.grads['W'] = np.matmul(self.inputs.T, grads)
        self.grads['b'] = np.sum(grads, axis=0)
        return np.matmul(grads, self.params['W'].T)
    
class Logistic():
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def forward(self, inputs):
        outputs = 1.0 / (1.0 + np.exp(-inputs))
        self.outputs = outputs
        return outputs

    def backward(self, grads):
        outputs_grad_inputs = np.multiply(self.outputs, (1.0 - self.outputs))
        return np.multiply(grads,outputs_grad_inputs)

class Model_MLP_L2():
    def __init__(self, input_size, hidden_size1,hidden_size2,hidden_size3,output_size):
        self.fc1 = Linear(input_size, hidden_size1, name="fc1")
        self.act_fn1 = Logistic()
        self.fc2 = Linear(hidden_size1, hidden_size2, name="fc2")
        self.act_fn2 = Logistic()
        self.fc3=Linear(hidden_size2,hidden_size3,name="fc3")
        self.act_fn3=Logistic()
        self.fc4=Linear(hidden_size3,output_size,name="fc4")
        self.layers = [self.fc1, self.act_fn1,self.fc2,self.act_fn2,self.fc3,self.act_fn3,self.fc4]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        for layer in self.layers:
            X=layer.forward(X)
        return X
        
    def backward(self, preds,labels):
        self.layers.reverse()
        loss=(preds-labels)/preds.shape[0]
        for layer in self.layers:
            loss=layer.backward(loss)
        self.layers.reverse()
        
class RunnerV2_1(object):
    def __init__(self, model, optimizer, metric, loss_fn, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.baches=0

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_epochs = kwargs.get("log_epochs", 100)
        save_dir = kwargs.get("save_dir", None)
        best_score = self.evaluate(dev_set)
        X_train,y_train=train_set
        X_num=X_train.shape[0]
        for epoch in range(num_epochs):
            X_train,y_train=train_set
            c = list(zip(X_train,y_train))
            random.shuffle(c)
            X_train,y_train = zip(*c)
            batches=[(X_train[i:min(X_num,i+64)],y_train[i:min(X_num,i+64)]) for i in range(0,X_num-64,64)]
            for (X,y) in batches:
                X=np.array(X)
                y=np.array(y)
                self.baches+=1
                logits = self.model(X)
                trn_loss = self.loss_fn(logits, y)
                self.model.backward(logits,y)
                self.optimizer.step()
            dev_score = self.evaluate(dev_set)
            if dev_score > best_score:
                print(f"[Evaluate] best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
                if save_dir:
                    self.save_model(save_dir)

            if log_epochs and epoch % log_epochs == 0:
                print(f"[Train] epoch: {epoch}/{num_epochs}, loss: {trn_loss.item(0)}")
                
    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        score = self.metric(logits, y).item(0)
        return score
    
    def predict(self, X):
        return self.model(X)

    def save_model(self, save_dir):
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                np.save(os.path.join(save_dir, layer.name+".pdparams"),layer.params)

    def load_model(self, model_dir):
        model_file_names = os.listdir(model_dir)
        name_file_dict = {}
        for file_name in model_file_names:
            name = file_name.replace(".pdparams.npy","")
            name_file_dict[name] = os.path.join(model_dir, file_name)
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                name = layer.name
                file_path = name_file_dict[name]
                temp = np.load(file_path,allow_pickle=True)
                layer.params = temp.item(0)

def accuracy(preds, labels):
    return -(preds-labels)*(preds-labels)

class MLPClassifier(object):
    def __init__(self,input_dim,output_dim):
        self.model=Model_MLP_L2(input_size=input_dim,output_size=output_dim,hidden_size1=32,hidden_size2=16,hidden_size3=8)
        self.runner=None

    def fit(self,X_train,y_train,X_validate,y_validate,lr,num_epochs,log_epochs,save_path="mpl_best_model.pdparams"):
        optimizer=BatchGD(lr,self.model)
        loss_fn = DisLoss(self.model)
        metric = accuracy
        self.save_path=save_path
        self.runner=RunnerV2_1(self.model,optimizer,metric,loss_fn)
        if os.path.exists(save_path):
            self.runner.load_model(model_dir=save_path)
        else:
            os.mkdir(save_path)
        self.runner.train([X_train, y_train], [X_validate, y_validate], num_epochs=num_epochs, log_epochs=log_epochs, eval_epochs=1, save_dir=save_path)
        pass

    def predict(self,X,y,save_path="mpl_best_model.pdparams"):
        self.runner=RunnerV2_1(self.model,None,None,None)
        self.runner.load_model(model_dir=save_path)
        y_predict=self.runner.predict(X)
        return y_predict
    