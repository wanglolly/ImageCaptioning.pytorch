import csv
import matplotlib.pyplot as plt
import numpy as np

captionModel = 'show_attend_tell'
filename = 'LossFile/' + captionModel + '.csv'

def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta)

def get_train_loss(dta):
    loss = []
    for l in dta[:, 2]:
        loss.append(float(l))
    return loss

trainloss = get_train_loss(read_table(open(filename, 'r')))
print('Minimum value: ' + str(min(trainloss)))
#Plot the training loss
plt.subplots()
plt.plot(range(56644), trainloss, label= 'Training Loss')
plt.legend()
plt.ylim([0., 10.0])
plt.xlabel("Iteration")
plt.ylabel('Loss')
plt.savefig("LossFile/" + captionModel + '.png', dpi=300, bbox_inches='tight')
plt.close()
