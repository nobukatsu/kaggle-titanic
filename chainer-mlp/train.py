import numpy as np
import pandas as pd
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import mlp

data_f = pd.read_csv('../train.csv', header=0)

# Use PClass, Sex, Age only.
data_f = data_f[["Pclass", "Sex", "Age", "Survived"]]

# Fill empty Age with Median.
data_f["Age"] = data_f["Age"].fillna(data_f["Age"].median())
# male -> 1, female -> 0
data_f["Sex"] = data_f["Sex"].replace("male", 1)
data_f["Sex"] = data_f["Sex"].replace("female", 0)

data_array = data_f.as_matrix()

X = []
Y = []
for x in data_array:
    x_split = np.hsplit(x, [3,4])
    X.append(x_split[0].astype(np.float32))
    Y.append(x_split[1].astype(np.int32))

X = np.array(X)
Y = np.ndarray.flatten(np.array(Y))

train = datasets.TupleDataset(X, Y)
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)

model = L.Classifier(mlp.MLP())
optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (50, 'epoch'), out='result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()

# save model
serializers.save_npz('titanic-prediction.model', model)
