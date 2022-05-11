import tensorflow.keras as kr
import tensorflow as tf
import tensorflow_probability as tfp
import HyperParameters as hp
from scipy.linalg import sqrtm
import numpy as np
import scipy.stats as ss
from sklearn.metrics import auc

inception_model = tf.keras.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)


@tf.function
def _get_feature_samples(decoder: kr.Model, real_images, latent_scale_vector):
    batch_size = real_images.shape[0]
    latent_vectors = hp.latent_dist_func(batch_size)

    fake_images = tf.clip_by_value(
        decoder(latent_vectors * latent_scale_vector[tf.newaxis]),
        clip_value_min=-1, clip_value_max=1)
    real_images = tf.tile(real_images, [1, 1, 1, 3])
    fake_images = tf.tile(fake_images, [1, 1, 1, 3])
    real_images = tf.image.resize(real_images, [299, 299])
    fake_images = tf.image.resize(fake_images, [299, 299])

    real_features = inception_model(real_images)
    fake_features = inception_model(fake_images)

    return real_features, fake_features


def _get_features(decoder: kr.Model, test_dataset: tf.data.Dataset, latent_scale_vector):
    real_features = []
    fake_features = []

    for data in test_dataset:
        real_features_batch, fake_features_batch = _get_feature_samples(decoder, data['image'], latent_scale_vector)
        real_features.append(real_features_batch)
        fake_features.append(fake_features_batch)

    real_features = tf.concat(real_features, axis=0)
    fake_features = tf.concat(fake_features, axis=0)

    return real_features, fake_features


def get_fid(decoder: kr.Model, test_dataset: tf.data.Dataset, latent_scale_vector):
    real_features, fake_features = _get_features(decoder, test_dataset, latent_scale_vector)
    real_features_mean = tf.reduce_mean(real_features, axis=0)
    fake_features_mean = tf.reduce_mean(fake_features, axis=0)

    mean_difference = tf.reduce_sum((real_features_mean - fake_features_mean) ** 2)
    real_cov, fake_cov = tfp.stats.covariance(real_features), tfp.stats.covariance(fake_features)
    cov_mean = sqrtm(tf.matmul(real_cov, fake_cov))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    cov_difference = tf.linalg.trace(real_cov + fake_cov - 2.0 * cov_mean)

    fid = mean_difference + cov_difference

    return fid


def evaluate_fake_rec(encoder: kr.Model, decoder: kr.Model, test_dataset: tf.data.Dataset, latent_scale_vector):
    average_psnr = []
    average_ssim = []
    for _ in test_dataset:
        latent_vectors = hp.latent_dist_func(hp.batch_size)
        fake_images = tf.clip_by_value(
            decoder(latent_vectors * latent_scale_vector[tf.newaxis]),
            clip_value_min=-1, clip_value_max=1)

        _, rec_latent_vectors = encoder(fake_images)
        rec_images = tf.clip_by_value(
            decoder(rec_latent_vectors * latent_scale_vector[tf.newaxis]),
            clip_value_min=-1, clip_value_max=1)

        average_psnr.append(tf.reduce_mean(tf.image.psnr(fake_images, rec_images, max_val=2.0)))
        average_ssim.append(tf.reduce_mean(tf.image.ssim(fake_images, rec_images, max_val=2.0)))

    return tf.reduce_mean(average_psnr), tf.reduce_mean(average_ssim)


def evaluate_real_rec(encoder: kr.Model, decoder: kr.Model, test_dataset: tf.data.Dataset, latent_scale_vector):
    average_psnr = []
    average_ssim = []
    for data in test_dataset:
        real_images = data['image']
        _, rec_latent_vectors = encoder(real_images)
        rec_images = tf.clip_by_value(
            decoder(rec_latent_vectors * latent_scale_vector[tf.newaxis]),
            clip_value_min=-1, clip_value_max=1)

        average_psnr.append(tf.reduce_mean(tf.image.psnr(real_images, rec_images, max_val=2.0)))
        average_ssim.append(tf.reduce_mean(tf.image.ssim(real_images, rec_images, max_val=2.0)))

    return tf.reduce_mean(average_psnr), tf.reduce_mean(average_ssim)


def get_accuracy(encoder: kr.Model, svm: kr.Model, test_dataset: tf.data.Dataset):
    total_count = 0
    wrong_count = 0
    for data in test_dataset:
        real_labels = tf.argmax(data['label'], axis=-1)
        predict_labels = tf.argmax(svm(encoder(data['image'])[1]), axis=-1)

        batch_size = real_labels.shape[0]
        wrong = tf.math.count_nonzero(real_labels - predict_labels)

        total_count += batch_size
        wrong_count += wrong

    return (total_count - wrong_count) / total_count


def get_anodlsgan_ood_scores(encoder: kr.Model, test_dataset: tf.data.Dataset, var_trace):
    scores = []
    for data in test_dataset:
        real_images = data['image']
        batch_size = real_images.shape[0]
        _, rec_latent_vectors = encoder(real_images)

        for i in range(batch_size):
            log_probs = -tf.math.log(tf.cast(ss.norm.pdf(rec_latent_vectors[i], scale=tf.sqrt(var_trace)), 'float32'))
            scores.append(tf.reduce_sum(log_probs))

    return tf.convert_to_tensor(scores)


def get_inv_cov_feature_mean(encoder: kr.Model, train_dataset: tf.data.Dataset):
    feature_vectors = []
    for data in train_dataset:
        feature_vectors.append(encoder(data['image'])[1])

    feature_vectors = tf.concat(feature_vectors, axis=0)
    inv_cov = tf.linalg.inv(tfp.stats.covariance(feature_vectors))
    feature_mean = tf.reduce_mean(feature_vectors, axis=0, keepdims=True)

    return inv_cov, feature_mean


def get_mahalanobis_ood_scores(encoder: kr.Model, inv_cov, feature_mean, test_dataset: tf.data.Dataset):
    scores_sets = []
    for data in test_dataset:
        _, feature_vectors = encoder(data['image'])

        scores = tf.matmul(tf.matmul((feature_vectors - feature_mean), inv_cov), tf.transpose(feature_vectors - feature_mean))
        scores = tf.linalg.diag_part(scores)
        scores_sets.append(scores)

    return tf.concat(scores_sets, axis=0)


def get_energy_ood_scores(encoder: kr.Model, svm: kr.Model, test_dataset):
    latent_vectors_set = [encoder(data['image'])[1] for data in test_dataset]
    latent_values = tf.reshape(tf.concat(latent_vectors_set, axis=0), [-1])
    latent_values = tf.sort(latent_values)
    activation_threshold = latent_values[round((latent_values.shape[0] - 1) * hp.react_p)]

    scores_sets = []
    for latent_vectors in latent_vectors_set:
        logits = svm(tf.math.minimum(latent_vectors, activation_threshold))
        scores = -hp.temperature * tf.math.log(tf.reduce_sum(tf.exp(logits / hp.temperature), axis=-1))
        scores_sets.append(scores)

    return tf.concat(scores_sets, axis=0)


def get_rec_ood_scores(encoder: kr.Model, decoder: kr.Model, dataset: tf.data.Dataset):
    scores_sets = []
    for data in dataset:
        real_images = data['image']
        rec_images = decoder(encoder(real_images)[1])
        scores = tf.sqrt(tf.reduce_sum(tf.square(real_images - rec_images), axis=[1, 2, 3]))
        scores_sets.append(scores)

    return tf.concat(scores_sets, axis=0)


def get_aurocs(in_ood_scores, out_ood_scores_sets):
    aurocs = []
    for out_ood_scores in out_ood_scores_sets:
        fprs = [0.0]
        tprs = [0.0]

        ood_thresholds = np.linspace(np.max(in_ood_scores) + 1e-6, np.min(in_ood_scores) - 1e-6, 1000)

        for ood_threshold in ood_thresholds:
            in_ood_num = tf.math.count_nonzero(tf.where(in_ood_scores > ood_threshold, 1.0, 0.0))
            fpr = in_ood_num / in_ood_scores.shape[0]
            fprs.append(fpr)

            out_ood_num = tf.math.count_nonzero(tf.where(out_ood_scores > ood_threshold, 1.0, 0.0))
            tpr = out_ood_num / out_ood_scores.shape[0]
            tprs.append(tpr)
        fprs.append(1.0)
        tprs.append(1.0)

        aurocs.append(auc(fprs, tprs))

    return aurocs
