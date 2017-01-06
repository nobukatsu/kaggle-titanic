import numpy as np
import pandas as pd
import chainer
from chainer import serializers
from chainer import Variable
import chainer.links as L
import mlp

org_data_f = pd.read_csv('../test.csv', header=0)
data_f = org_data_f.copy()[["Pclass", "Sex", "Age"]]

# Fill empty Age with Median.
data_f["Age"] = data_f["Age"].fillna(data_f["Age"].median())
# male -> 1, female -> 0
data_f["Sex"] = data_f["Sex"].replace("male", 1)
data_f["Sex"] = data_f["Sex"].replace("female", 0)

X = data_f.as_matrix().astype(np.float32)

# load model
model = L.Classifier(mlp.MLP())
serializers.load_npz('titanic-prediction.model', model)

# prediction
y = model.predictor(Variable(X))
result = np.argmax(y.data, axis=1)

result_df = pd.DataFrame(columns=["PassengerId", "Survived"])

for idx, row in org_data_f.iterrows():
    result_df.loc[idx] = [org_data_f["PassengerId"][idx], result[idx]]

result_df = result_df.astype(int)

# output result
result_df.to_csv("result.csv", index=False)

print("### done ###")
