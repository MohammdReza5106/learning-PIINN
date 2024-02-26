"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

miu=1
E=1
def pde(x, y):
    # Most backends
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)

    # Backend jax
    # dy_xx, _ = dde.grad.hessian(y, x, i=0, j=0)
    # dy_yy, _ = dde.grad.hessian(y, x, i=1, j=1)

    return -dy_xx - dy_yy - E/(1+miu)

    
def boundary(_, on_boundary):
    return on_boundary





geom = dde.geometry.Rectangle([-1,-1],[1,1])
bc = dde.icbc.DirichletBC(geom, lambda x: np.sin(x[:,-1]) , boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=1200, num_boundary=120, num_test=1500)
net = dde.nn.FNN([2] + [120] * 2 + [1], "tanh", "Glorot uniform")#for having time it should be 3 input probably
model = dde.Model(data, net)

model.compile("adam", lr=0.001)
model.train(iterations=5000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
