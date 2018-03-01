from tf_lstm import *
from lstm_torch import *
from data_handler import *


if __name__ == '__main__':
    dh = DataHandler('EURUSD.csv')
    dh.timeSeriesToSupervised()
    dh.splitData(1000, 1000, len(dh.tsdata) - 2000)

    train, test, out = dh.getDataSets()

    trainx, trainy = train[:, 1], train[:, 3]
    testx, testy = test[:, 1], test[:, 3]
    trainy = trainy.reshape(trainy.shape[0], 1)
    trainx = trainx.reshape(trainx.shape[0], 1)
    testx = testx.reshape(testx.shape[0], 1)
    trainx = trainx.reshape((trainx.shape[0], 1, trainx.shape[1]))
    testx = testx.reshape((testx.shape[0], 1, testx.shape[1]))

    print(trainx.shape)

    lstm = TensorflowLSTM(trainx.shape[1], 10, [10 for _ in range(0, 10)],
           trainy.shape[1], epochs=1000)
    # print(lstm.predict(testx))
    lstm.train(trainx, trainy)
    print(lstm.predict(testx))
