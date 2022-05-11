import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as hp
import Models
import datetime
import time


def train_gan(encoder: Models.Encoder, decoder: Models.Decoder, dataset, epochs):
    @tf.function
    def _gan_train_step(encoder: kr.Model, decoder: kr.Model, latent_var_trace, data, encoder_optimizer,
                        decoder_optimizer):
        with tf.GradientTape(persistent=True) as tape:
            real_images = data['image']
            batch_size = real_images.shape[0]
            latent_vectors = hp.latent_dist_func(batch_size)

            if hp.is_dls:
                latent_scale_vector = tf.sqrt(
                    tf.cast(hp.latent_vector_dim, dtype='float32') * latent_var_trace / tf.reduce_sum(latent_var_trace))
            else:
                latent_scale_vector = tf.ones([hp.latent_vector_dim])

            with tf.GradientTape() as r1_tape:
                r1_tape.watch(real_images)
                real_adv_values, _ = encoder(real_images)
            reg_losses = tf.reduce_sum(tf.square(r1_tape.gradient(real_adv_values, real_images)), axis=[1, 2, 3])

            fake_images = decoder(latent_vectors * latent_scale_vector[tf.newaxis])
            fake_adv_values, rec_latent_vectors = encoder(fake_images)

            encoder_losses = tf.reduce_mean(
                tf.square((rec_latent_vectors - latent_vectors) * latent_scale_vector[tf.newaxis]), axis=-1)

            discriminator_adv_losses = tf.squeeze(tf.nn.softplus(-real_adv_values) + tf.nn.softplus(fake_adv_values))
            generator_adv_losses = tf.squeeze(tf.nn.softplus(-fake_adv_values))

            discriminator_loss = tf.reduce_mean(
                discriminator_adv_losses + hp.enc_weight * encoder_losses + hp.reg_weight * reg_losses)
            generator_loss = tf.reduce_mean(
                generator_adv_losses + hp.enc_weight * encoder_losses + hp.reg_weight * reg_losses)

        latent_var_trace = latent_var_trace * hp.latent_var_decay_rate + \
                           tf.reduce_mean(tf.square(rec_latent_vectors), axis=0) * (1.0 - hp.latent_var_decay_rate)

        decoder_optimizer.apply_gradients(
            zip(tape.gradient(generator_loss, decoder.trainable_variables),
                decoder.trainable_variables)
        )
        encoder_optimizer.apply_gradients(
            zip(tape.gradient(discriminator_loss, encoder.trainable_variables),
                encoder.trainable_variables)
        )

        del tape

        return latent_var_trace, tf.reduce_mean(encoder_losses), tf.reduce_mean(reg_losses)

    for epoch in range(epochs):
        print(datetime.datetime.now())
        print('epoch', epoch)
        start = time.time()
        enc_losses = []
        reg_losses = []

        for data in dataset:
            decoder.latent_var_trace, enc_loss, reg_loss = _gan_train_step(encoder.model, decoder.model,
                                                                           decoder.latent_var_trace,
                                                                           data, encoder.optimizer, decoder.optimizer)
            enc_losses.append(enc_loss)
            reg_losses.append(reg_loss)

        print('enc loss:', tf.reduce_mean(tf.convert_to_tensor(enc_losses)).numpy())
        print('reg loss:', tf.reduce_mean(tf.convert_to_tensor(reg_losses)).numpy())

        encoder.optimizer.lr = encoder.optimizer.lr * hp.lr_decay_rate
        decoder.optimizer.lr = decoder.optimizer.lr * hp.lr_decay_rate

        print('saving...')
        encoder.save()
        decoder.save()
        print('saved')
        print('time: ', time.time() - start, '\n')


def train_ae(encoder: Models.Encoder, decoder: Models.Decoder, dataset, epochs):
    @tf.function
    def _ae_train_step(encoder: kr.Model, decoder: kr.Model, data, encoder_optimizer, decoder_optimizer):
        with tf.GradientTape(persistent=True) as tape:
            real_images = data['image']
            rec_images = decoder(encoder(real_images)[1])

            rec_losses = tf.sqrt(tf.reduce_sum(tf.square(rec_images - real_images), axis=[1, 2, 3]))
            rec_loss = tf.reduce_mean(rec_losses)
        decoder_optimizer.apply_gradients(
            zip(tape.gradient(rec_loss, decoder.trainable_variables),
                decoder.trainable_variables)
        )
        encoder_optimizer.apply_gradients(
            zip(tape.gradient(rec_loss, encoder.trainable_variables),
                encoder.trainable_variables)
        )
        del tape
        return rec_loss

    for epoch in range(epochs):
        print(datetime.datetime.now())
        print('epoch', epoch)
        start = time.time()
        rec_losses = []

        for data in dataset:
            rec_losses.append(_ae_train_step(encoder.model, decoder.model, data, encoder.optimizer, decoder.optimizer))

        print('rec loss:', tf.reduce_mean(tf.convert_to_tensor(rec_losses)).numpy())

        encoder.optimizer.lr = encoder.optimizer.lr * hp.lr_decay_rate
        decoder.optimizer.lr = decoder.optimizer.lr * hp.lr_decay_rate

        print('saving...')
        encoder.save()
        decoder.save()
        print('saved')
        print('time: ', time.time() - start, '\n')


def train_classifier(encoder: Models.Encoder, svm: Models.Svm, dataset, epochs):
    @tf.function
    def _classifier_train_step(encoder: kr.Model, svm: kr.Model, data, encoder_optimizer, svm_optimizer):
        with tf.GradientTape(persistent=True) as tape:
            real_images = data['image']
            real_labels = data['label']
            predict_logits = svm(encoder(real_images)[1])
            ce_losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(real_labels, predict_logits))

        encoder_optimizer.apply_gradients(
            zip(tape.gradient(ce_losses, encoder.trainable_variables),
                encoder.trainable_variables)
        )
        svm_optimizer.apply_gradients(
            zip(tape.gradient(ce_losses, svm.trainable_variables),
                svm.trainable_variables)
        )
        del tape
        return ce_losses

    for epoch in range(epochs):
        print(datetime.datetime.now())
        print('epoch', epoch)
        start = time.time()
        ce_losses = []

        for data in dataset:
            ce_losses.append(_classifier_train_step(encoder.model, svm.model, data, encoder.optimizer, svm.optimizer))

        print('ce loss:', tf.reduce_mean(tf.convert_to_tensor(ce_losses)).numpy())

        encoder.optimizer.lr = encoder.optimizer.lr * hp.lr_decay_rate
        svm.optimizer.lr = svm.optimizer.lr * hp.lr_decay_rate

        print('saving...')
        encoder.save()
        svm.save()
        print('saved')
        print('time: ', time.time() - start, '\n')
