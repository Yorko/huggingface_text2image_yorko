import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, one_hot_from_names


def print_image(numpy_array):
    """

    Utility function to print a numpy uint8 array as an image

    :param numpy_array: a NumPy array shaped [width x height x 3]
    :return: None
    """
    img = Image.fromarray(numpy_array)
    plt.imshow(img)
    plt.show()


def generate_images(dense_class_vector=None, name=None, noise_seed_vector=None,
                    truncation=0.4, gan_model=None,
                    pretrained_gan_model_name='biggan-deep-128',
                    batch_size=15):
    """

    :param dense_class_vector: used as a replacement of BigGAN internal
            ImageNet class embeddings (torch.Tensor with 128 elements)
    :param name: converted in an associated ImageNet class and then
            a dense class embedding using BigGAN's internal ImageNet
            class embeddings (string)
    :param noise_seed_vector: a vector used to control the seed
            (seed set to the sum of the vector elements)
    :param truncation: a float between 0 and 1 to control image quality/diversity
            tradeoff (see BigGAN paper)
    :param gan_model: a BigGAN model from pytorch_pretrained_biggan library.
                If None a model is instanciated from a pretrained model name given
                by `pretrained_gan_model_name`. List of possible names:
                https://github.com/huggingface/pytorch-pretrained-BigGAN#models
    :param pretrained_gan_model_name: shortcut name of the GAN model to instantiate
            if no gan_model is provided. Default to 'biggan-deep-128'
    :param batch_size: number of images to generate
    :return: a tuple with two numpy arrays, the first one is shaped
            [batch_size, 128, 128, 3], the second one is a raw output
            from BigGAM generator, used to save images
    """

    seed = int(noise_seed_vector.sum().item()) if noise_seed_vector is not None else None
    noise_vector = truncated_noise_sample(truncation=truncation,
                                          batch_size=batch_size, seed=seed)
    noise_vector = torch.from_numpy(noise_vector)
    if gan_model is None:
        gan_model = BigGAN.from_pretrained(pretrained_gan_model_name)
    if name is not None:
        class_vector = one_hot_from_names([name], batch_size=batch_size)
        class_vector = torch.from_numpy(class_vector)
        dense_class_vector = gan_model.embeddings(class_vector)
    else:
        dense_class_vector = dense_class_vector.view(-1, 128)
    input_vector = torch.cat([noise_vector, dense_class_vector], dim=1)

    # Generate an image
    with torch.no_grad():
        output = gan_model.generator(input_vector, truncation)
        raw_biggan_output = output.clone()

    output = output.cpu().numpy()
    output = output.transpose((0, 2, 3, 1))

    if batch_size == 1:
        output = output[0]

    output = ((output + 1.0) / 2.0) * 256
    output.clip(0, 255, out=output)
    output = np.asarray(np.uint8(output), dtype=np.uint8)
    return output, raw_biggan_output
