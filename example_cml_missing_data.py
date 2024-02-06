import numpy as np
from scipy.special import comb

from cml.train_utils import train
from sklearn.metrics import hamming_loss
from cml.pred_utils import pred

data = np.load("data/propensity_score_models.npz")
x_train = data["imputed_train"]
x_val = data["imputed_val"]
s_train = data["train_mask"]
s_val = data["val_mask"]

d = np.shape(x_train)[-1]
q = np.shape(s_train)[-1]
K = d * q + 4 * comb(q, 2)  # each individual label with each feature and label pair-wise relationship ->
# there are 4 possible values for pair-wise label relationship, therefore 4.
# co-occurrences
thegma = 2 ** (1)  # 参数寻优，-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6
# 训练数据集
lam = train(K, thegma, x_train, s_train)

predictions = []
predictions_probs = []
for j in range(len(x_val)):
    prediction, pred_prob = pred(x_val[j], lam, d, q)
    predictions.append(prediction)
    predictions_probs.append(pred_prob)
predictions = np.array(predictions)
predictions_probs = np.asarray(predictions_probs)
acc = hamming_loss(s_val, predictions)  # 汉明损失，越低越好
print('acc=', acc)
