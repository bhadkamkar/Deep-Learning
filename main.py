import theano
import keras
from hw_utils import *
import time
import sys


sys.stdout = open('reportC', 'w')

X_tr,y_tr,X_te,y_te = loaddata("data.txt")

X_tr, X_te = normalize(X_tr, X_te)

epoch30 = 30
epoch50 = 50
epoch100 = 100

#linear activation 1
print "----------------- Linear Activation 1 -----------------"
start_time = time.time()
a = testmodels(X_tr, y_tr, X_te, y_te, 
                [[50,2],[50,50,2],[50,50,50,2],[50,50,50,50,2]],
                actfn='linear',last_act='softmax', reg_coeffs=[0.0], 
                num_epoch=epoch30, batch_size=1000, sgd_lr=1e-3, 
                sgd_decays=[0.0], sgd_moms=[0.0], 
                sgd_Nesterov=False, EStop=False, verbose=0)
end_time = time.time()
print "Linear activation 1 took: ",end_time-start_time," sec"



#linear activation 2
print "----------------- Linear Activation 2 -----------------"
start_time = time.time()
b = testmodels(X_tr, y_tr, X_te, y_te, 
                [[50,50,2],[50,500,2],[50,500,300,2],[50,800, 500, 300,2],[50, 800, 800, 500, 300,2]],
                actfn='linear',last_act='softmax', reg_coeffs=[0.0], 
                num_epoch=epoch30, batch_size=1000, sgd_lr=1e-3, 
                sgd_decays=[0.0], sgd_moms=[0.0], 
                sgd_Nesterov=False, EStop=False, verbose=0)
end_time = time.time()
print ""Linear activation 2 took: ",end_time-start_time," sec"


#sigmoid activation
print "----------------- Sigmoid Activation -----------------"
start_time = time.time()
c = testmodels(X_tr, y_tr, X_te, y_te, 
                [[50,50,2],[50,500,2],[50,500,300,2],[50,800, 500, 300,2],[50, 800, 800, 500, 300,2]],
                actfn='sigmoid',last_act='softmax', reg_coeffs=[0.0], 
                num_epoch=epoch30, batch_size=1000, sgd_lr=1e-3, 
                sgd_decays=[0.0], sgd_moms=[0.0], 
                sgd_Nesterov=False, EStop=False, verbose=0)

end_time = time.time()
print "Sigmoid activation took: ",end_time-start_time," sec"

#relu activation
print "----------------- Relu Activation -----------------"
start_time = time.time()
d = testmodels(X_tr, y_tr, X_te, y_te, 
                [[50,50,2],[50,500,2],[50,500,300,2],[50,800, 500, 300,2],[50, 800, 800, 500, 300,2]],
                actfn='relu',last_act='softmax', reg_coeffs=[0.0], 
                num_epoch=epoch30, batch_size=1000, sgd_lr=5e-4, 
                sgd_decays=[0.0], sgd_moms=[0.0], 
                sgd_Nesterov=False, EStop=False, verbose=0)

end_time = time.time()
print "Relu activation took: ",end_time-start_time," sec"


#L2-Regularization
print "----------------- L2-Regularization -----------------"
start_time = time.time()
e = testmodels(X_tr, y_tr, X_te, y_te, 
                [[50,800, 500, 300,2]],
                actfn='relu',last_act='softmax', reg_coeffs=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5], 
                num_epoch=epoch30, batch_size=1000, sgd_lr=5e-4, 
                sgd_decays=[0.0], sgd_moms=[0.0], 
                sgd_Nesterov=False, EStop=False, verbose=0)

end_time = time.time()
print "L2 Regularization took: ",end_time-start_time," sec"


#Early Stopping and L2-regularization
print "------------ Early Stopping and L2-regularization ------------"
start_time = time.time()
f = testmodels(X_tr, y_tr, X_te, y_te, 
                [[50,800, 500, 300,2]],
                actfn='relu',last_act='softmax', reg_coeffs=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5], 
                num_epoch=epoch30, batch_size=1000, sgd_lr=5e-4, 
                sgd_decays=[0.0], sgd_moms=[0.0], 
                sgd_Nesterov=False, EStop=True, verbose=0)

end_time = time.time()
print "Early stopping and L2 regularization took: ",end_time-start_time," sec"


#SGD with weight decay
print "----------------- SGD with weight decay -----------------"
start_time = time.time()
g = testmodels(X_tr, y_tr, X_te, y_te, 
                [[50,800, 500, 300,2]],
                actfn='relu',last_act='softmax', reg_coeffs=[5e-7], 
                num_epoch=epoch100, batch_size=1000, sgd_lr=1e-5, 
                sgd_decays=[1e-5, 5e-5, 1e-4, 3e-4, 7e-4, 1e-3], sgd_moms=[0.0], 
                sgd_Nesterov=False, EStop=False, verbose=0)

end_time = time.time()
print "SGD with weight decay took: ",end_time-start_time," sec"



#momentum
print "----------------- momentum -----------------"
start_time = time.time()
h = testmodels(X_tr, y_tr, X_te, y_te, 
                [[50,800, 500, 300,2]],
                actfn='relu',last_act='softmax', reg_coeffs=[0.0], 
                num_epoch=epoch50, batch_size=1000, sgd_lr=1e-5, 
                sgd_decays=[g[2]], sgd_moms=[0.99,0.98,0.95,0.9,0.85], 
                sgd_Nesterov=True, EStop=False, verbose=0)

end_time = time.time()
print "Momentum took: ",end_time-start_time," sec"



#combine
print "----------------- combine -----------------"
start_time = time.time()
i = testmodels(X_tr, y_tr, X_te, y_te, 
                [[50,800, 500, 300,2]],
                actfn='relu',last_act='softmax', reg_coeffs=[e[1]], 
                num_epoch=epoch100, batch_size=1000, sgd_lr=1e-5, 
                sgd_decays=[g[2]], sgd_moms=[h[3]], 
                sgd_Nesterov=True, EStop=True, verbose=0)

end_time = time.time()
print "Combine took: ",end_time-start_time," sec"


#grid search
print "----------------- grid search -----------------"
start_time = time.time()
j = testmodels(X_tr, y_tr, X_te, y_te, 
                [[50,50,2],[50,500,2],[50,500,300,2],[50,800, 500, 300,2],[50,800,800,500,300,2]],
                actfn='relu',last_act='softmax', reg_coeffs=[1e-7,5e-7,1e-6,5e-6,1e-5], 
                num_epoch=epoch100, batch_size=1000, sgd_lr=1e-5, 
                sgd_decays=[1e-5,5e-5,1e-4], sgd_moms=[0.99], 
                sgd_Nesterov=True, EStop=True, verbose=0)

end_time = time.time()
print "Grid search took: ",end_time-start_time," sec"
