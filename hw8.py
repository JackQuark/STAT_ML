import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython import display

#active funcion
def Sigmoid(z):
    return 1 / ( 1 + np.exp(-z) )

def relu(p):
    return np.maximum(0,p)

def drelu(x):
    x[x<=0] = 0
    x[x>0]  = 1
    return x

#----------------------------

#deal with data
#because the data is only one feature(wind speed)
#I firstly didn't deal the data with 'feature scaling'
#but the outputs are seen 

raw_x = np.loadtxt('hw8_u_x.csv',skiprows = 0 ,usecols = np.arange(1,145),delimiter = ',')
raw_y = np.loadtxt('hw8_u_y.csv',skiprows = 0 ,usecols = 1               ,delimiter = ',')
train_x , valid_x , train_y , valid_y = train_test_split(raw_x,raw_y,test_size = 0.3   , random_state = 42) \
                                                                                        # 42 make not random
def standardize(data):
    return (data - data.mean()) / data.std()
train_x = standardize(train_x)
valid_x = standardize(valid_x)

train_y = train_y.reshape((-1,1))
valid_y = valid_y.reshape((-1,1))

#----------------------------

#model node & setting
input_nodes  = 144
hidden_nodes = 10
output_nodes = 1
ETA          = 0.0001
EPOCHS       = 60
Loss_slide   = np.zeros((EPOCHS, 2))

#random weight & bias
hidden_w = np.random.randn(input_nodes ,hidden_nodes) * (np.sqrt(2/input_nodes ))
hidden_b = np.zeros((1,hidden_nodes))
output_w = np.random.randn(hidden_nodes,output_nodes) * (np.sqrt(2/hidden_nodes))
output_b = np.zeros((1,output_nodes))

#define the network calculate 
#with input_node & hidden_node & output_node

def forward(x):

    hidden_sum    = np.dot(x,hidden_w)             + hidden_b
    hidden_output = relu(hidden_sum)
    output_sum    = np.dot(hidden_output,output_w) + output_b
    output_output = Sigmoid(output_sum)

    return (hidden_sum, hidden_output , output_sum ,output_output)

def backward():

    #gradient descent to modify w & b
    dz2 = output_output - train_y
    dw2 = np.dot(  hidden_output.T , dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    dz1 = np.dot(dz2,output_w.T) * drelu(hidden_sum)
    dw1 = np.dot(  train_x.T       , dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True) 

    return (dw2,db2,dw1,db1)

#training
for epochs in range(EPOCHS):
    hidden_sum, hidden_output, output_sum, output_output = forward(train_x)
    dw2,db2,dw1,db1 = backward()
    #gradient descent to modify w & b
    hidden_w    = hidden_w - ETA*dw1
    hidden_b    = hidden_b - ETA*db1
    output_w    = output_w - ETA*dw2
    output_b    = output_b - ETA*db2
    train_loss  = - np.sum(train_y * np.log(output_output))
    
    _,__,___,test_output_output = forward(valid_x)
    test_loss   = -np.sum(valid_y * np.log(test_output_output))
    
    Loss_slide[epochs,0], Loss_slide[epochs,1] = train_loss,test_loss

#plot loss value
plt.figure(figsize=(8,5))
plt.title("Loss change") 
plt.xlabel("training times") 
plt.ylabel("Loss value")
plt.plot(Loss_slide[:,0],label='train_loss')
plt.plot(Loss_slide[:,1],label='test_loss')

#accuracy
def classify(x):
    _,__,___, output_output = forward(x)
    return np.around(output_output)
mtx_valid_y = classify(valid_x)

train_right = 0
test_right = 0

for train_Y,train_y in zip(np.around(output_output),train_y):
    if train_y == train_Y:
        train_right +=1
for valid_Y,valid_y in zip(mtx_valid_y,valid_y):
    if valid_y == valid_Y:
        test_right +=1

print("train_accuracy:",( train_right / len(output_output) ) * 100,'%' )
print("test_accuracy:" ,( test_right  / len(mtx_valid_y)   ) * 100,'%' )

plt.legend()
plt.show()

