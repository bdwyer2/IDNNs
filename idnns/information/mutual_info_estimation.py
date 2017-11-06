import numpy as np
import sys
import tensorflow as tf
from idnns.networks import model as mo
import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def estimate_IY_by_network(data, labels, from_layer=0):
    if len(data.shape) > 2:
        input_size = data.shape[1:]
    else:
        input_size = data.shape[1]
    p_y_given_t_i = data
    acc_all = [0]
    if from_layer < 5:

        acc_all = []
        g1 = tf.Graph()  # This is one graph
        with g1.as_default():
            # For each epoch and for each layer we calculate the best decoder - we train a 2 layer network
            cov_net = 4
            model = mo.Model(input_size, [400, 100, 50], labels.shape[1], 0.0001, '', cov_net=cov_net,
                             from_layer=from_layer)
            if from_layer < 5:
                optimizer = model.optimize
            init = tf.global_variables_initializer()
            num_of_ephocs = 50
            batch_size = 51
            batch_points = np.rint(np.arange(0, data.shape[0] + 1, batch_size)).astype(dtype=np.int32)
            if data.shape[0] not in batch_points:
                batch_points = np.append(batch_points, [data.shape[0]])
        with tf.Session(graph=g1) as sess:
            sess.run(init)
            if from_layer < 5:
                for j in range(0, num_of_ephocs):
                    for i in range(0, len(batch_points) - 1):
                        batch_xs = data[batch_points[i]:batch_points[i + 1], :]
                        batch_ys = labels[batch_points[i]:batch_points[i + 1], :]
                        feed_dict = {model.x: batch_xs, model.labels: batch_ys}
                        if cov_net == 1:
                            feed_dict[model.drouput] = 0.5
                        optimizer.run(feed_dict)
            p_y_given_t_i = []
            batch_size = 256
            batch_points = np.rint(np.arange(0, data.shape[0] + 1, batch_size)).astype(dtype=np.int32)
            if data.shape[0] not in batch_points:
                batch_points = np.append(batch_points, [data.shape[0]])
            for i in range(0, len(batch_points) - 1):
                batch_xs = data[batch_points[i]:batch_points[i + 1], :]
                batch_ys = labels[batch_points[i]:batch_points[i + 1], :]
                feed_dict = {model.x: batch_xs, model.labels: batch_ys}
                if cov_net == 1:
                    feed_dict[model.drouput] = 1
                p_y_given_t_i_local, acc = sess.run([model.prediction, model.accuracy],
                                                    feed_dict=feed_dict)
                acc_all.append(acc)
                if i == 0:
                    p_y_given_t_i = np.array(p_y_given_t_i_local)
                else:
                    p_y_given_t_i = np.concatenate((p_y_given_t_i, np.array(p_y_given_t_i_local)), axis=0)
                # print ("The accuracy of layer number - {}  - {}".format(from_layer, np.mean(acc_all)))
    max_indx = len(p_y_given_t_i)
    labels_cut = labels[:max_indx, :]
    true_label_index = np.argmax(labels_cut, 1)
    s = np.log2(p_y_given_t_i[np.arange(len(p_y_given_t_i)), true_label_index])
    I_TY = np.mean(s[np.isfinite(s)])
    PYs = np.sum(labels_cut, axis=0) / labels_cut.shape[0]
    Hy = np.nansum(-PYs * np.log2(PYs + np.spacing(1)))
    I_TY = Hy + I_TY
    I_TY = I_TY if I_TY >= 0 else 0
    acc = np.mean(acc_all)
    sys.stdout.flush()
    return I_TY, acc


def calc_varitional_information(data, labels, model_path, layer_numer, num_of_layers, epoch_index, input_size,
                                layerSize, sigma, pys, ks,
                                search_sigma=False, estimate_y_by_network=False):
    """Calculate estimation of the information using vartional IB"""
    # Assumptions
    estimate_y_by_network = True
    # search_sigma = False
    data_x = data.reshape(data.shape[0], -1)

    if search_sigma:
        sigmas = np.linspace(0.2, 10, 20)
        sigmas = [0.2]
    else:
        sigmas = [sigma]

    I_XT = 0

    if estimate_y_by_network:
        I_TY, acc = estimate_IY_by_network(data, labels, from_layer=layer_numer)
    else:
        I_TY = 0
    with printoptions(precision=3, suppress=True, formatter={'float': '{: 0.3f}'.format}):
        print('[{0}:{1}] - I(X;T) - {2}, I(X;Y) - {3}, accuracy - {4}'.format(epoch_index, layer_numer,
                                                                              np.array(I_XT).flatten(), I_TY, acc))
    sys.stdout.flush()

    # I_est = mutual_inform[ation((data, labels[:, 0][:, None]), PYs, k=ks)
    # I_est,I_XT = 0, 0
    params = {}
    # params['DKL_YgX_YgT'] = DKL_YgX_YgT
    # params['pts'] = p_ts
    # params['H_Xgt'] = H_Xgt
    params['local_IXT'] = I_XT
    params['local_ITY'] = I_TY
    return params
