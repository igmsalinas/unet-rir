import math
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from datageneratorv2 import DataGenerator
from dataset import Dataset
from dl_models.u_net import UNet
from dl_models.autoencoder import Autoencoder
from dl_models.res_ae import ResAE
from dl_models.vae import VAE
import time


def sigmoid(beta, dimensions):
    x = np.linspace(-10, 10, dimensions[1])
    z = 1 / (1 + np.exp(-(x + 5) * beta))
    z = np.flip(z)
    sig = np.tile(z, (dimensions[0], 1))
    return sig


if __name__ == '__main__':
    ########################################################
    # Inputs and model selection
    ########################################################
    target_size = (144, 160, 2)
    rooms = ['LargeMeetingRoom']
    arrays = ['PlanarMicrophoneArray', 'CircularMicrophoneArray']
    name = 'unet'

    ########################################################
    # Hyperparams - TEACHER MODEL TRAINING
    ########################################################
    debug = True

    alpha = 0.9
    sigmoid_loss = False
    diff_loss = False
    beta = 0.5

    n_epochs = 500
    lr = 5e-7
    batch_size_per_replica = 16

    optimizer_sel = "adam"
    lr_exp_decay = [True, 80]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    sig_size = (target_size[0], target_size[1])
    sig = sigmoid(beta, sig_size)
    sig = np.repeat(np.expand_dims(sig, 0), batch_size_per_replica, axis=0)

    strategy = tf.distribute.MirroredStrategy()

    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    ########################################################
    # Data directories and folders
    ########################################################

    file_name = '../results/' + name
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    ########################################################
    # Main training
    ########################################################

    # Prepare data generators
    dataset = Dataset('../../../datasets', 'room_impulse', normalization=True, debugging=debug, extract=False,
                      room=rooms, array=arrays)

    train_generator = DataGenerator(dataset, batch_size=global_batch_size, partition='train', shuffle=True)
    val_generator = DataGenerator(dataset, batch_size=global_batch_size, partition='val', shuffle=True)


    def generator_t(stop):
        i = 0
        while i < stop:
            spec_in, emb, spec_out = train_generator.__getitem__(i)
            yield spec_in, emb, spec_out
            i += 1


    def generator_v(stop):
        i = 0
        while i < stop:
            spec_in, emb, spec_out = val_generator.__getitem__(i)
            yield spec_in, emb, spec_out
            i += 1


    train_dataset = tf.data.Dataset.from_generator(generator_t, args=[train_generator.__len__()],
                                                   output_types=(tf.float32, tf.int32, tf.float32),
                                                   output_shapes=(tf.TensorShape(
                                                       [None, target_size[0], target_size[1], target_size[2]]),
                                                                  tf.TensorShape([None, 2, 16]),
                                                                  tf.TensorShape([None, target_size[0], target_size[1],
                                                                                  target_size[2]])))

    val_dataset = tf.data.Dataset.from_generator(generator_v, args=[val_generator.__len__()],
                                                 output_types=(tf.float32, tf.int32, tf.float32),
                                                 output_shapes=(tf.TensorShape(
                                                     [None, target_size[0], target_size[1], target_size[2]]),
                                                                tf.TensorShape([None, 2, 16]),
                                                                tf.TensorShape([None, target_size[0], target_size[1],
                                                                                target_size[2]])))

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    with strategy.scope():

        if name == "ae":
            # Create model
            model = Autoencoder(
                input_shape=target_size,
                inf_vector_shape=(2, 16),
                conv_filters=(64, 128, 256, 512),
                conv_kernels=(3, 3, 3, 3),
                conv_strides=(2, 2, 2, 2),
                latent_space_dim=64,
                n_neurons=32 * 64,
                name=name
            )
        elif name == "resae":
            model = ResAE(
                input_shape=target_size,
                inf_vector_shape=(2, 16),
                conv_filters=(32, 64, 128, 256),
                conv_kernels=(3, 3, 3, 3),
                conv_strides=(2, 2, 2, 2),
                latent_space_dim=32,
                n_neurons=16 * 64,
                name=name
            )
        elif name == "vae":
            model = VAE(
                input_shape=target_size,
                inf_vector_shape=(2, 16),
                conv_filters=(64, 128, 256, 512),
                conv_kernels=(3, 3, 3, 3),
                conv_strides=(2, 2, 2, 2),
                latent_space_dim=64,
                n_neurons=32 * 64,
                name=name
            )

        elif name == "unet":
            model = UNet(input_shape=target_size,
                         inf_vector_shape=(2, 16),
                         mode=0,
                         number_filters_0=32,
                         kernels=3,
                         name="U-Net"
                         )

        # Set optimizer and checkpoint
        if 'nadam' in optimizer_sel:
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        elif 'sgd' in optimizer_sel:
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif 'adam' in optimizer_sel:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model.model)
        manager = tf.train.CheckpointManager(checkpoint, directory=file_name, max_to_keep=2)

        model.summary()

        # Define the loss function

        loss_object_amplitude = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE)

        loss_object_phase = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE)

        def phase_loss(y_true, y_pred):
            y_true = y_true * 2 * math.pi - math.pi
            y_pred = y_pred * 2 * math.pi - math.pi
            y_diff = y_true - y_pred
            phase = (y_diff + math.pi) % (2 * math.pi) - math.pi
            loss = 1 - tf.math.cos(phase)
            return loss

        def kl_loss_object(mean, log_var):
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            return kl_loss

        def compute_kl_loss(mean, log_var):
            kl_loss = kl_loss_object(mean, log_var)
            per_example_loss = tf.reduce_sum(kl_loss, axis=1)
            kl_loss = tf.nn.compute_average_loss(per_example_loss,
                                                 global_batch_size=global_batch_size)
            return kl_loss

        def compute_loss(x, y_true, y_pred, model_losses):
            stft_true = y_true[:, :, :, 0]
            phase_true = y_true[:, :, :, 1]
            stft_pred = y_pred[:, :, :, 0]
            phase_pred = y_pred[:, :, :, 1]

            phase_x = x[:, :, :, 1]

            per_example_loss_amplitude = loss_object_amplitude(tf.expand_dims(stft_true, -1),
                                                               tf.expand_dims(stft_pred, -1))

            if diff_loss:

                per_example_loss_phase = phase_loss(phase_true - phase_x, phase_pred)
            else:
                per_example_loss_phase = phase_loss(phase_true, phase_pred)

            # per_example_loss_phase = loss_object_phase(tf.expand_dims(phase_true, -1),
            #                                                    tf.expand_dims(phase_pred, -1))

            if sigmoid_loss:
                per_example_loss_phase = per_example_loss_phase * tf.convert_to_tensor(sig, np.float32)

            per_example_loss = alpha * per_example_loss_amplitude + (1 - alpha) * per_example_loss_phase

            per_example_loss /= tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)

            loss = tf.nn.compute_average_loss(per_example_loss,
                                              global_batch_size=global_batch_size)
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))

            return loss


        # Define metrics
        train_loss_amplitude = tf.keras.metrics.Mean(
            name='train_loss_amplitude')
        train_loss_phase = tf.keras.metrics.Mean(name='train_loss_phase')

        val_loss_amplitude = tf.keras.metrics.Mean(name='val_loss_amplitude')
        val_loss_phase = tf.keras.metrics.Mean(name='val_loss_phase')

        # VAE metrics

        if name == "vae":
            train_loss_kl = tf.keras.metrics.Mean(name='train_loss_kl')
            val_loss_kl = tf.keras.metrics.Mean(name='val_loss_kl')


    def train_step(inputs):
        spec_in, emb, spec_out = inputs

        with tf.GradientTape() as tape:
            if name == "vae":
                z, mean, log_var = model.encoder([spec_in, emb], training=True)
                spec_pred = model.decoder(z, training=True)
            else:
                spec_pred = model.model([spec_in, emb], training=True)

            loss = compute_loss(spec_in, spec_out, spec_pred, model.model.losses)
            if name == "vae":
                loss += compute_kl_loss(mean, log_var)

        gradients = tape.gradient(loss, model.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.model.trainable_variables))

        stft_true = spec_out[:, :, :, 0]
        phase_true = spec_out[:, :, :, 1]
        stft_pred = spec_pred[:, :, :, 0]
        phase_pred = spec_pred[:, :, :, 1]

        loss_amplitude = loss_object_amplitude(tf.expand_dims(stft_true, -1),
                                               tf.expand_dims(stft_pred, -1))

        if diff_loss:
            loss_phase = phase_loss(phase_true - spec_in[:, :, :, 1], phase_pred)
        else:
            loss_phase = phase_loss(phase_true, phase_pred)

        train_loss_amplitude.update_state(loss_amplitude)
        train_loss_phase.update_state(loss_phase)

        if name == "vae":
            loss_kl = kl_loss_object(mean, log_var)
            train_loss_kl.update_state(loss_kl)

        return loss


    def test_step(inputs):
        spec_in, emb, spec_out = inputs

        if name == "vae":
            z, mean, log_var = model.encoder([spec_in, emb], training=True)
            spec_pred = model.decoder(z, training=True)
        else:
            spec_pred = model.model([spec_in, emb], training=True)

        stft_true = spec_out[:, :, :, 0]
        phase_true = spec_out[:, :, :, 1]
        stft_pred = spec_pred[:, :, :, 0]
        phase_pred = spec_pred[:, :, :, 1]

        loss_amplitude = loss_object_amplitude(tf.expand_dims(stft_true, -1),
                                               tf.expand_dims(stft_pred, -1))

        if diff_loss:
            loss_phase = phase_loss(phase_true - spec_in[:, :, :, 1], phase_pred)
        else:
            loss_phase = phase_loss(phase_true, phase_pred)

        val_loss_amplitude.update_state(loss_amplitude)
        val_loss_phase.update_state(loss_phase)

        if name == "vae":
            loss_kl = kl_loss_object(mean, log_var)
            val_loss_kl.update_state(loss_kl)


    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.run(test_step, args=(dataset_inputs,))


    start = time.time()

    for epoch in range(n_epochs):
        # TRAIN LOOP
        epoch_start = time.time()

        # exponential lr in last epochs
        if lr_exp_decay[0]:
            if epoch >= lr_exp_decay[1]:
                K.set_value(optimizer.learning_rate, lr * 0.9 ** (epoch / lr_exp_decay[1]))

        total_loss = 0.0
        num_batches = 0

        for x in train_dist_dataset:
            print("Training step: " + str(num_batches) + "/" + str(train_generator.__len__()), end='\r')
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        # TEST LOOP

        val_batches = 0
        for x in val_dist_dataset:
            print("Val step: " + str(val_batches) + "/" + str(val_generator.__len__()), end='\r')
            distributed_test_step(x)
            val_batches += 1

        if epoch % 2 == 0:
            save_path = manager.save()

        epoch_end = time.time()

        if name == "vae":
            template = ("Epoch {}, Loss: {}, Epoch time: {}\n"
                        "Train | MSE Loss: {}, Phase Loss: {}, KL Loss: {}\n"
                        "Val   | MSE Loss: {}, Phase Loss: {}, KL Loss: {}\n"
                        "lr    | {}")
            print(template.format(epoch + 1, train_loss, (epoch_end - epoch_start),
                                  train_loss_amplitude.result(), train_loss_phase.result(), train_loss_kl.result(),
                                  val_loss_amplitude.result(), val_loss_phase.result(), val_loss_kl.result(),
                                  optimizer.lr.numpy()))
        else:
            template = ("Epoch {}, Loss: {}, Epoch time: {}\n"
                        "Train | MSE Loss: {}, Phase Loss: {}\n"
                        "Val   | MSE Loss: {}, Phase Loss: {}\n"
                        "lr    | {}")
            print(template.format(epoch + 1, train_loss, (epoch_end - epoch_start),
                                  train_loss_amplitude.result(), train_loss_phase.result(),
                                  val_loss_amplitude.result(), val_loss_phase.result(),
                                  optimizer.lr.numpy()))

        train_loss_amplitude.reset_states()
        train_loss_phase.reset_states()
        val_loss_amplitude.reset_states()
        val_loss_phase.reset_states()

    end = time.time()
    print("Training complete, took " + str(end - start))
