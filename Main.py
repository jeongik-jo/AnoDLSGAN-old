import HyperParameters as hp
import Train
import time
import Models
import Dataset
import Evaluate
import tensorflow as tf


def train_gan():
    results = {'aurocs': [], 'fid': [], 'fake_psnr': [], 'fake_ssim': [], 'real_psnr': [], 'real_ssim': []}
    for i in range(hp.fold_size):
        i_start = int(70000 / hp.fold_size * i)
        i_end = int(70000 / hp.fold_size * (i + 1))

        train_dataset, test_dataset = Dataset.load_dataset(i_start, i_end)
        ood_datasets = Dataset.load_ood_datasets(i_start, i_end)

        encoder = Models.Encoder()
        decoder = Models.Decoder()

        if hp.load_model:
            encoder.load(), decoder.load()

        Train.train_gan(encoder, decoder, train_dataset, hp.epochs)

        print('evaluating...', '\n')
        start = time.time()
        in_ood_scores = Evaluate.get_anodlsgan_ood_scores(encoder.model, test_dataset, decoder.latent_var_trace)
        out_ood_scores_sets = [Evaluate.get_anodlsgan_ood_scores(encoder.model, ood_dataset, decoder.latent_var_trace)
                               for ood_dataset in ood_datasets]
        results['aurocs'].append(Evaluate.get_aurocs(in_ood_scores, out_ood_scores_sets))
        if hp.is_dls:
            latent_scale_vector = tf.sqrt(tf.cast(hp.latent_vector_dim, dtype='float32') * decoder.latent_var_trace / tf.reduce_sum(decoder.latent_var_trace))
        else:
            latent_scale_vector = tf.ones([hp.latent_vector_dim])

        results['fid'].append(Evaluate.get_fid(decoder.model, test_dataset, latent_scale_vector))
        fake_psnr, fake_ssim = Evaluate.evaluate_fake_rec(encoder.model, decoder.model, test_dataset, latent_scale_vector)
        real_psnr, real_ssim = Evaluate.evaluate_real_rec(encoder.model, decoder.model, test_dataset, latent_scale_vector)
        results['fake_psnr'].append(fake_psnr)
        results['fake_ssim'].append(fake_ssim)
        results['real_psnr'].append(real_psnr)
        results['real_ssim'].append(real_ssim)
        print('time: ', time.time() - start, '\n')

    file_name = ''
    if hp.is_dls:
        file_name += 'DLS_'
    else:
        file_name += 'NoDLS_'
    file_name += 'ood_' + str(hp.ood_intensity)

    with open(file_name + '.txt', 'w') as file:
        for key in results:
            if key == 'aurocs':
                aurocs = tf.reduce_mean(tf.convert_to_tensor(results[key]), axis=0).numpy()

                for ood_dataset_name, auroc in zip(hp.ood_datasets, aurocs):
                    print(ood_dataset_name, 'auroc:', auroc)
                    file.write(ood_dataset_name + ' auroc: ' + str(auroc) + '\n')
            else:
                average = tf.reduce_mean(tf.convert_to_tensor(results[key])).numpy()
                print(key, ':', average)
                file.write(key + ':' + str(average))


def train_ae():
    results = {'mah_aurocs': [], 'rec_aurocs': [], 'real_psnr': [], 'real_ssim': []}
    for i in range(hp.fold_size):
        i_start = int(70000 / hp.fold_size * i)
        i_end = int(70000 / hp.fold_size * (i + 1))

        train_dataset, test_dataset = Dataset.load_dataset(i_start, i_end)
        ood_datasets = Dataset.load_ood_datasets(i_start, i_end)

        encoder = Models.Encoder()
        decoder = Models.Decoder()

        if hp.load_model:
            encoder.load(), decoder.load()

        Train.train_ae(encoder, decoder, train_dataset, hp.epochs)

        print('evaluating...', '\n')
        start = time.time()

        inv_cov, feature_mean = Evaluate.get_inv_cov_feature_mean(encoder.model, train_dataset)
        in_ood_scores = Evaluate.get_mahalanobis_ood_scores(encoder.model, inv_cov, feature_mean, test_dataset)
        out_ood_scores_sets = [Evaluate.get_mahalanobis_ood_scores(encoder.model, inv_cov, feature_mean, ood_dataset)
                               for ood_dataset in ood_datasets]
        results['mah_aurocs'].append(Evaluate.get_aurocs(in_ood_scores, out_ood_scores_sets))

        in_ood_scores = Evaluate.get_rec_ood_scores(encoder.model, decoder.model, test_dataset)
        out_ood_scores_sets = [Evaluate.get_rec_ood_scores(encoder.model, decoder.model, ood_dataset)
                               for ood_dataset in ood_datasets]
        results['rec_aurocs'].append(Evaluate.get_aurocs(in_ood_scores, out_ood_scores_sets))

        real_psnr, real_ssim = Evaluate.evaluate_real_rec(encoder.model, decoder.model, test_dataset, tf.ones([hp.latent_vector_dim]))
        results['real_psnr'].append(real_psnr)
        results['real_ssim'].append(real_ssim)
        print('time: ', time.time() - start, '\n')

    file_name = 'AE_ood_' + str(hp.ood_intensity)
    with open(file_name + '.txt', 'w') as file:
        for key in results:
            if key == 'mah_aurocs' or key == 'rec_aurocs':
                aurocs = tf.reduce_mean(tf.convert_to_tensor(results[key]), axis=0).numpy()

                for ood_dataset_name, auroc in zip(hp.ood_datasets, aurocs):
                    print(ood_dataset_name, key, ':', auroc)
                    file.write(ood_dataset_name + '_' + key + ': ' + str(auroc) + '\n')
            else:
                average = tf.reduce_mean(tf.convert_to_tensor(results[key])).numpy()
                print(key, ':', average)
                file.write(key + ':' + str(average))


def train_classifier():
    results = {'mah_aurocs': [], 'energy_aurocs': [], 'accuracy': []}
    for i in range(hp.fold_size):
        i_start = int(70000 / hp.fold_size * i)
        i_end = int(70000 / hp.fold_size * (i + 1))

        train_dataset, test_dataset = Dataset.load_dataset(i_start, i_end)
        ood_datasets = Dataset.load_ood_datasets(i_start, i_end)

        encoder = Models.Encoder()
        svm = Models.Svm()

        if hp.load_model:
            encoder.load(), svm.load()

        Train.train_classifier(encoder, svm, train_dataset, hp.epochs)

        print('evaluating...', '\n')
        start = time.time()

        inv_cov, feature_mean = Evaluate.get_inv_cov_feature_mean(encoder.model, train_dataset)
        in_ood_scores = Evaluate.get_mahalanobis_ood_scores(encoder.model, inv_cov, feature_mean, test_dataset)
        out_ood_scores_sets = [Evaluate.get_mahalanobis_ood_scores(encoder.model, inv_cov, feature_mean, ood_dataset)
                               for ood_dataset in ood_datasets]
        results['mah_aurocs'].append(Evaluate.get_aurocs(in_ood_scores, out_ood_scores_sets))

        in_ood_scores = Evaluate.get_energy_ood_scores(encoder.model, svm.model, test_dataset)
        out_ood_scores_sets = [Evaluate.get_energy_ood_scores(encoder.model, svm.model, ood_dataset)
                               for ood_dataset in ood_datasets]
        results['energy_aurocs'].append(Evaluate.get_aurocs(in_ood_scores, out_ood_scores_sets))

        results['accuracy'].append(Evaluate.get_accuracy(encoder.model, svm.model, test_dataset))
        print('time: ', time.time() - start, '\n')

    file_name = 'classifier_ood_' + str(hp.ood_intensity) + '_t_' + str(hp.temperature) + '_p_' + str(hp.react_p)

    with open(file_name + '.txt', 'w') as file:
        for key in results:
            if key == 'mah_aurocs' or key == 'energy_aurocs':
                aurocs = tf.reduce_mean(tf.convert_to_tensor(results[key]), axis=0).numpy()

                for ood_dataset_name, auroc in zip(hp.ood_datasets, aurocs):
                    print(ood_dataset_name, key, ':', auroc)
                    file.write(ood_dataset_name + '_' + key + ': ' + str(auroc) + '\n')
            else:
                average = tf.reduce_mean(tf.convert_to_tensor(results[key])).numpy()
                print(key, ':', average)
                file.write(key + ':' + str(average))


train_gan()
