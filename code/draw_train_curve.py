import matplotlib.pyplot as plt
import cPickle as pkl

FILES = ['plot/logs/RRN_0_0.001_0', 'plot/logs/RRN_WS_0_0.001_0', 'plot/logs/RRN_2HOP_0_0.001_0']
MODELS = ['RRN', 'RRN_WS', 'RRN_2HOP']
LINE_STYLES = ['r-', 'b-', 'g-', 'c-', 'y-']

aucs_train = []
aucs_test = []
losses_train = []
losses_test = []

for i in range(len(FILES)):
    with open(FILES[i]) as f:
        t = pkl.load(f)
        losses_train.append(t[0])
        losses_test.append(t[1])
        aucs_train.append(t[2])
        aucs_test.append(t[3])

# plot
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
for i in range(len(MODELS)):
    x = range(len(aucs_train[i]))
    plt.plot(x, aucs_train[i], LINE_STYLES[i])
plt.legend(MODELS)
plt.title('AUC_TRAIN')

plt.subplot(2, 2, 2)
for i in range(len(MODELS)):
    x = range(len(aucs_test[i]))
    plt.plot(x, aucs_test[i], LINE_STYLES[i])
plt.legend(MODELS)
plt.title('AUC_TEST')

plt.subplot(2, 2, 3)
for i in range(len(MODELS)):
    x = range(len(losses_train[i]))
    plt.plot(x, losses_train[i], LINE_STYLES[i])
plt.legend(MODELS)
plt.title('LOSS_TRAIN')

plt.subplot(2, 2, 4)
for i in range(len(MODELS)):
    x = range(len(losses_test[i]))
    plt.plot(x, losses_test[i], LINE_STYLES[i])
plt.legend(MODELS)
plt.title('LOSS_TEST')


plt.savefig('train_curve.pdf')
