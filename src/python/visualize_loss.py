import numpy as np
import matplotlib.pyplot as plt


val_losses = np.load('val_losses.npy')
train_losses = np.load('train_losses.npy')
mAPs = np.load('mAPs.npy')
train_losses_ = train_losses[::9]
mAPs = mAPs*100


plt.xlim(0, 800)
plt.ylim(0, 40)
trn, = plt.plot(train_losses_, label="Training loss")
mAP, = plt.plot(mAPs, label="mAP (%)")
val, = plt.plot(val_losses, label="Validation loss")
plt.legend(handles=[trn, val, mAP])
plt.show()