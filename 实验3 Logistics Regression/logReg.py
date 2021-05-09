import time
import numpy as np
import matplotlib.pyplot as plt
import gzip as gz


def load_data(filename, kind):
    with gz.open(filename, 'rb') as fo:
        buf = fo.read()
        index = 0
        if kind == 'data':
            header = np.frombuffer(buf, '>i', 4, index)
            index += header.size * header.itemsize
            data = np.frombuffer(buf, '>B', header[1] * header[2] * header[3], index).reshape(header[1], -1)
        elif kind == 'lable':
            header = np.frombuffer(buf, '>i', 2, 0)
            index += header.size * header.itemsize
            data = np.frombuffer(buf, '>B', header[1], index)
    return data


X_train = load_data('train-images-idx3-ubyte.gz', 'data')
y_train = load_data('train-labels-idx1-ubyte.gz', 'lable')
X_test = load_data('t10k-images-idx3-ubyte.gz', 'data')
y_test = load_data('t10k-labels-idx1-ubyte.gz', 'lable')
X_train = np.array(X_train[y_train <= 1, :])
y_train = np.array(y_train[y_train <= 1])
X_test = np.array(X_test[y_test <= 1, :])
y_test = np.array(y_test[y_test <= 1])
print('Train data shape:')
print(X_train.shape, y_train.shape)
print('Test data shape:')
print(X_test.shape, y_test.shape)
x_dim = 28 * 28
y_dim = 2
W_dim = (y_dim, x_dim)
b_dim = y_dim


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def loss(W, b, x, y):
    return -np.log(softmax(np.dot(W, x) + b)[y])  # 预测值与标签相同的概率


def L_Gra(W, b, x, y):
    W_G = np.zeros(W.shape)
    b_G = np.zeros(b.shape)
    S = softmax(np.dot(W, x) + b)
    W_row = W.shape[0]
    W_column = W.shape[1]
    b_column = b.shape[0]
    for i in range(W_row):
        for j in range(W_column):
            W_G[i][j] = (S[i] - 1) * x[j] if y == i else S[i] * x[j]
    for i in range(b_column):
        b_G[i] = S[i] - 1 if y == i else S[i]
    return W_G, b_G


def test_accurate(W, b, X_test, y_test):
    num = len(X_test)
    results = []
    for i in range(num):
        y_i = np.dot(W, X_test[i]) + b
        res = 1 if softmax(y_i).argmax() == y_test[i] else 0
        results.append(res)
    accurate_rate = np.mean(results)
    return accurate_rate


def mini_batch(batch_size, alpha, epoches):
    accurate_rates = []
    iters_W = []
    iters_b = []
    W = np.zeros(W_dim)
    b = np.zeros(b_dim)
    x_batches = np.zeros(((int(X_train.shape[0] / batch_size), batch_size, 784)))
    y_batches = np.zeros(((int(X_train.shape[0] / batch_size), batch_size)))
    batches_num = int(X_train.shape[0] / batch_size)
    for i in range(0, X_train.shape[0], batch_size):
        x_batches[int(i / batch_size)] = X_train[i:i + batch_size]
        y_batches[int(i / batch_size)] = y_train[i:i + batch_size]
    print('Start training...')
    start = time.time()
    for epoch in range(epoches):
        for i in range(batches_num):
            W_gradients = np.zeros(W_dim)
            b_gradients = np.zeros(b_dim)
            x_batch, y_batch = x_batches[i], y_batches[i]
            for j in range(batch_size):
                W_g, b_g = L_Gra(W, b, x_batch[j], y_batch[j])
                W_gradients += W_g
                b_gradients += b_g
            W_gradients /= batch_size
            b_gradients /= batch_size
            W -= alpha * W_gradients
            b -= alpha * b_gradients
            accurate_rates.append(test_accurate(W, b, X_test, y_test))
            iters_W.append(W.copy())
            iters_b.append(b.copy())
    end = time.time()
    time_cost = (end - start)
    return W, b, time_cost, accurate_rates, iters_W, iters_b


def run(alpha, batch_size, epochs_num):
    W, b, time_cost, accuracys, W_s, b_s = mini_batch(batch_size, alpha, epochs_num)
    iterations = len(W_s)
    dis_W = []
    dis_b = []
    for i in range(iterations):
        dis_W.append(np.linalg.norm(W_s[i] - W))
        dis_b.append(np.linalg.norm(b_s[i] - b))
    print("the parameters is: step length alpah:{}; batch size:{}; Epoches:{}".format(alpha, batch_size, epochs_num))
    print("Result: accuracy:{:.2%},time cost:{:.2f}".format(accuracys[-1], time_cost))
    plt.title('The Model accuracy variation chart ')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.plot(accuracys, 'm')
    plt.grid()
    plt.show()
    plt.title('The distance from the optimal solution')
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.plot(dis_W, 'r', label='distance between W and W*')
    plt.plot(dis_b, 'g', label='distance between b and b*')
    plt.legend()
    plt.grid()
    plt.show()


alpha = 1e-6
batch_size = 12665  # 5,17,85,745
epochs_num = 1

run(alpha, batch_size, epochs_num)