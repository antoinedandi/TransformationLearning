import os
import warnings

import tensorflow as tf
from tensorflow.python.summary import summary as tf_summary
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import warp, AffineTransform

layers = tf.keras.layers
models = tf.keras.models
callbacks = tf.keras.callbacks
K = tf.keras.backend


def generator(images, labels=None, batch_size=32):
    while True:
        if labels is not None:
            indexes = np.random.choice(len(images), batch_size)
            yield images[indexes], labels[indexes]
        else:
            yield np.random.choice(images, batch_size)


def delta_generator(images, batch_size=32):
    while True:
        im_shape = images.shape[1:]
        trans_imgs = np.zeros((batch_size, *im_shape))
        deltas = np.zeros((batch_size, 2))
        indexes = np.random.choice(len(images), batch_size)
        ref_imgs = images[indexes]
        # TODO: find more efficient way to compute transformation (multiplication by a matrix)
        for i, index in enumerate(indexes):
            # transform image
            deltas[i] = 0.6 * (np.random.rand(2) - 0.5) * ref_imgs[i].shape[0]
            trans_imgs[i] = translate(ref_imgs[i], deltas[i])
        yield [ref_imgs, trans_imgs], deltas


def delta_generator_int(images, batch_size=32):
    while True:
        im_shape = images.shape[1:]
        trans_imgs = np.zeros((batch_size, *im_shape))
        deltas = np.zeros((batch_size, 2), dtype=np.int64)
        indexes = np.random.choice(len(images), batch_size)
        ref_imgs = images[indexes]
        for i, index in enumerate(indexes):
            # transform image
            deltas[i] = - 28 + np.random.randint(57, size=2)
            trans_imgs[i] = translate_int(ref_imgs[i], deltas[i])
        yield [ref_imgs, trans_imgs], deltas


def delta_generator_from_labels(images, labels, batch_size=32):
    n_samples = len(images)
    while True:
        idx = np.random.choice(n_samples, 2 * batch_size)
        ref_idx = idx[:batch_size]
        trans_idx = idx[batch_size:]
        ref_imgs = images[ref_idx]
        trans_imgs = images[trans_idx]
        deltas = labels[trans_idx] - labels[ref_idx]
        yield [ref_imgs, trans_imgs], deltas


def build_delta_model(loc_net_func, in_shape):
    # ref_image tensor
    ref_img_tensor = layers.Input(in_shape)
    trans_img_tensor = layers.Input(in_shape)
    loc_net = loc_net_func(in_shape)
    # localizations
    ref_loc = loc_net(ref_img_tensor)
    trans_loc = loc_net(trans_img_tensor)

    # deltas
    deltas_loc = layers.Subtract()([trans_loc, ref_loc])

    # global inputs
    delta_inputs = [ref_img_tensor, trans_img_tensor]
    # build model
    delta_model = models.Model(inputs=delta_inputs, outputs=deltas_loc)

    return loc_net, delta_model


def translate(img, delta):
    tform = AffineTransform(translation=(- delta[1], - delta[0]))
    return warp(img, tform)


def translate_int(img, delta):
    shape = img.shape
    trans_img = np.zeros(shape)
    height, width = shape[:-1]
    if delta[0] >= 0 and delta[1] >= 0:
        trans_img[delta[0]:, delta[1]:] = img[:height - delta[0], :width - delta[1]]
    if delta[0] < 0 and delta[1] >= 0:
        trans_img[:delta[0], delta[1]:] = img[-delta[0]:, :width - delta[1]]
    if delta[0] >= 0 and delta[1] < 0:
        trans_img[delta[0]:, :delta[1]] = img[:height - delta[0], -delta[1]:]
    if delta[0] < 0 and delta[1] < 0:
        trans_img[:delta[0], :delta[1]] = img[-delta[0]:, width - delta[1]:]
    return trans_img


class CustomCallback(callbacks.Callback):

    def __init__(self,
                 loc_model,
                 model_name,
                 validation_data,
                 validation_labels,
                 weights_dirpath,
                 histo_dirpath,
                 log_dir,
                 histo_freq=1,
                 is_delta=True,
                 batchsize=512,
                 offset_sample=None,
                 lr_factor=0.1,
                 min_lr=1e-6,
                 patience=5):
        super(CustomCallback, self).__init__()
        self.loc_model = loc_model
        self.model_name = model_name
        self.data = validation_data
        self.labels = validation_labels
        self.weights_dirpath = weights_dirpath
        self.histo_dirpath = histo_dirpath
        self.writer = tf.summary.FileWriter(log_dir)
        self.histo_freq = histo_freq
        self.error_history = []
        self.batchsize = batchsize
        self.is_delta = is_delta
        self.previous_val_error = np.inf
        if self.is_delta:
            self.offset_sample, self.offset_label = offset_sample
        # reduce lr attributes
        self.best_val_loss = np.inf
        self.wait = 0
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # validation step
        errors = self.compute_val_errors()
        val_error = errors.mean()
        logs['val_loss'] = errors.mean()
        self.error_history.append(errors)
        # reducing lr
        logs = self.reduce_lr_on_plateau(logs)
        # logging step
        self.write_custom_summaries(epoch, logs)
        if epoch % self.histo_freq == 0:
            self.save_error_histo_and_summary(errors, epoch)
        # checkpoint step
        if self.previous_val_error < val_error:
            self.loc_model.save(self.weights_dirpath + '/' + str(epoch) + '.h5', overwrite=True)
        self.previous_val_error = val_error

    def on_train_end(self, logs=None):
        np.save(os.path.join(self.histo_dirpath, 'histo'), np.asarray(self.error_history))

    def compute_val_errors(self):
        choice = np.random.choice(len(self.labels), self.batchsize)
        batch_data = self.data[choice]
        batch_labels = self.labels[choice]
        batch_labels_pred = self.loc_model.predict(batch_data)
        if self.is_delta:
            offset = self.offset_label - np.squeeze(self.loc_model.predict(self.offset_sample[None, :, :, :]))
        else:
            offset = 0
        errors = ((batch_labels - (offset + batch_labels_pred)) ** 2).sum(axis=1)
        return errors

    def write_custom_summaries(self, step, logs):
        # use FileWriter from v1 summary
        for name, value in logs.items():
            summary = tf_summary.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, step)
        self.writer.flush()

    def save_error_histo_and_summary(self, errors, epoch):
        # write tf.summary.image for tensorboard
        # create plot and save
        plt.figure()
        plt.hist(errors, 50)
        plt.savefig(os.path.join(self.histo_dirpath, 'hist_{}'.format(epoch)))
        plt.close()
        # Convert error array to tensor
        errors_tensor = tf.convert_to_tensor(errors)
        # Add image summary
        summary_op = tf.summary.histogram("histogram", errors_tensor, family="val_loss_histogram_" + self.model_name)
        with tf.Session() as sess:
            # Run
            summary = sess.run(summary_op)
            # Write summary
            self.writer.add_summary(summary)
        # Write summary
        self.writer.add_summary(summary, global_step=epoch)
        self.writer.flush()

    def reduce_lr_on_plateau(self, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        monitor = 'val_loss'
        min_delta = 1e-4

        def monitor_op(a, b):
            return np.less(a, b - min_delta)

        current = logs.get(monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        else:
            if monitor_op(current, self.best_val_loss):
                self.best_val_loss = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.lr_factor
                        new_lr = max(new_lr, self.min_lr)

                        K.set_value(self.model.optimizer.lr, new_lr)
                        self.wait = 0
        return logs


def prepare_callbacks(model_dir, model_name, weights_dir, log_dir, loc_model, images_test, labels_test, histo_freq=10,
                      batchsize=2048, is_delta=False, offset_sample=None, lr_factor=0.1):
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    hist_dirpath = os.path.join(model_dir, 'hist/')
    if not os.path.exists(hist_dirpath):
        os.makedirs(hist_dirpath)
    val = CustomCallback(
        loc_model=loc_model,
        model_name=model_name,
        validation_data=images_test,
        validation_labels=labels_test,
        weights_dirpath=weights_dir,
        histo_dirpath=hist_dirpath,
        log_dir=log_dir,
        histo_freq=histo_freq,
        batchsize=batchsize,
        is_delta=is_delta,
        offset_sample=offset_sample,
        lr_factor=lr_factor)
    return val
