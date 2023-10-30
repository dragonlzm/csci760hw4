from myfunction import *
from mylayers import *
import numpy as np
import tensorflow as tf
from tensorflow import keras


if __name__ == "__main__":
    # set the hyper-params
    random_seed = 42
    #file_name = "mnist_subset.json"
    num_epoch = 10
    batch_size = 64
    learning_rate = 0.01
    act_fun = my_sigmoid
    L1_dim = 300
    #L1_dim = 200
    output_dim = 10

    # set the random seed to reproduce
    np.random.seed(random_seed)

    # load the data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    trainset_size, d = x_train.shape
    valset_size, _ = x_test.shape
    # train_dataset.data.shape
    trainloader = MyDataloader(x_train, y_train)
    valloader = MyDataloader(x_test, y_test)

    # build the network
    # The network structure is input --> linear --> sigmoid --> linear --> softmax_cross_entropy loss
    # the hidden_layer size (num_L1) is 300/200
    # the output_layer size (num_L2) is 10
    model = dict()
    model['Linear1'] = my_linear(input_dim = d, output_dim = L1_dim)
    model['activation_fun'] = act_fun()
    model['Linear2'] = my_linear(input_dim = L1_dim, output_dim = output_dim)
    model['loss'] = softmax_and_cross_entropy_loss()
    
    # train the model 
    train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []
    for t in range(num_epoch):
        print('Start the epoch ' + str(t + 1))
        idx_order = np.random.permutation(trainset_size)
        train_acc, train_loss, total_train_num = 0.0, 0.0, 0
        val_acc, val_loss, total_val_num = 0.0, 0.0, 0

        for i in range(int(np.floor(trainset_size / batch_size))):
            # get a mini-batch of data
            x, y = trainloader.get_example(idx_order[i * batch_size : (i + 1) * batch_size])
            # forward 
            a1 = model['Linear1'].forward(x)
            h1 = model['activation_fun'].forward(a1)
            a2 = model['Linear2'].forward(h1)
            loss = model['loss'].forward(a2, y)
            # backward 
            grad_a2 = model['loss'].backward(a2, y)
            grad_d1 = model['Linear2'].backward(h1, grad_a2)
            grad_a1 = model['activation_fun'].backward(a1, grad_d1)
            grad_x = model['Linear1'].backward(x, grad_a1)
            # gradient_update 
            model = myminiBatchGradientDescent(model, learning_rate)

        # Computing training acc
        for i in range(int(np.floor(trainset_size / batch_size))):
            x, y = trainloader.get_example(np.arange(i * batch_size, (i + 1) * batch_size))
            # forward 
            a1 = model['Linear1'].forward(x)
            h1 = model['activation_fun'].forward(a1)
            a2 = model['Linear2'].forward(h1)
            loss = model['loss'].forward(a2, y)
            # calculate the loss and acc
            train_loss += loss
            predicted_label = np.argmax(a2, axis=1).astype(float)
            predicted_label = predicted_label.reshape((a2.shape[0], -1))
            train_acc += np.sum(predicted_label == y)
            total_train_num += len(y)
        # aggreagte the acc
        train_acc = train_acc / total_train_num
        train_loss = train_loss / total_train_num
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training acc at epoch ' + str(t + 1) + ' is ' + str(train_acc))

        # Computing validation accuracy
        for i in range(int(np.floor(valset_size / batch_size))):
            x, y = valloader.get_example(np.arange(i * batch_size, (i + 1) * batch_size))
            # forward 
            a1 = model['Linear1'].forward(x)
            h1 = model['activation_fun'].forward(a1)
            a2 = model['Linear2'].forward(h1)
            loss = model['loss'].forward(a2, y)
            # calculate the loss and acc
            val_loss += loss
            predicted_label = np.argmax(a2, axis=1).astype(float)
            predicted_label = predicted_label.reshape((a2.shape[0], -1))
            val_acc += np.sum(predicted_label == y)
            total_val_num += len(y)
        # aggreagte the acc
        val_acc = val_acc / total_val_num
        val_loss = val_loss / total_train_num
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print('Validation acc at epoch ' + str(t + 1) + ' is ' + str(val_acc))

    # save log
    json.dump({'train_acc_list': train_acc_list, 'train_loss_list': train_loss_list, 
               'val_acc_list': val_acc_list, 'val_loss_list': val_loss_list},
              open('MLP_lr' + str(learning_rate) + '_layerdim_' + str(L1_dim) + '_bs_' + str(batch_size) + '.json', 'w'))

    print('Finish training.')


