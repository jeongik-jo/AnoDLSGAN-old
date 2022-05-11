import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

learning_rate = 0.001
lr_decay_rate = 0.95

image_resolution = 32
class_size = 10
train_dataset = 'mnist'
ood_datasets = ['mnist_corrupted/shot_noise', 'mnist_corrupted/impulse_noise', 'mnist_corrupted/glass_blur',
                'mnist_corrupted/motion_blur', 'mnist_corrupted/stripe', 'mnist_corrupted/fog',
                'mnist_corrupted/spatter', 'mnist_corrupted/dotted_line', 'mnist_corrupted/zigzag',
                'fashion_mnist', 'kmnist']

#ood_datasets = ['mnist_corrupted/shot_noise']
ood_intensity = 0.1
latent_vector_dim = 256

#GAN
is_dls = True
reg_weight = 0.1
enc_weight = 1.0
latent_var_decay_rate = 0.999
latent_dist_func = lambda batch_size: tf.random.normal([batch_size, latent_vector_dim])

#Energy
temperature = 1.0
react_p = 1.0


load_model = False
epochs = 30

batch_size = 32
fold_size = 7
save_image_size = 8





