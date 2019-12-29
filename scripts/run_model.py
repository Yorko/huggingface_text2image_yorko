import numpy as np
from pathlib import Path
import yaml
import logging
import torch
from matplotlib import pyplot as plt
from IPython import display
from transformers import AutoTokenizer, AutoModel
from pytorch_pretrained_biggan.utils import one_hot_from_names
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, display_in_terminal


class TextToImageModel:
    def __init__(self):
        """
        Initializes a tokenizer and Transformer as well as BigGAN model.

        """

        self.config = self._load_config()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        
        self.torch_device = torch.device(self.config['torch_device'])

        self.tokenizer, self.language_model = \
            self._initialize_tokenizer_n_transformer()

        self.gan_model = self._initialize_gan_model()

        self.mapping_model = self._load_mapping_model()

        self.max_seq_length = self.config['max_seq_length']



    @staticmethod
    def _get_project_dir():
        return Path(__file__).resolve().parent.parent

    def _load_config(self):

        with open(self._get_project_dir() / 'config.yml', 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        return config

    def _initialize_tokenizer_n_transformer(self):
        """
        Initializes tokenizer and language model from the
        Transformers library. Transformer pretrained model
        name is specified in the config.yaml file.

        :return: a tuple with tokenizer and model
        """

        self.logger.info('Initializing transformer...')
        lm_tokenizer = AutoTokenizer.from_pretrained(
            self.config['transformer_pretrained_model_name'])

        lm_model = AutoModel.from_pretrained(
            self.config['transformer_pretrained_model_name']
        ).to(self.torch_device)

        return lm_tokenizer, lm_model

    def _initialize_gan_model(self):
        """
        Initializes PyTorch BigGAN model. Pretrained model
        name is specified in the config.yaml file

        :return: instance of pytorch_pretrained_biggan.BigGAN class
        """

        self.logger.info('Initializing BigGAN model...')
        gan_model = BigGAN.from_pretrained(
            self.config['biggan_pretrained_model_name']
        ).to(self.torch_device)

        return gan_model

    def _load_mapping_model(self):

        self.logger.info('Loading mapping model...')

        checkpoint = torch.load(
            f=self._get_project_dir() / self.config['path_to_mapping_model_ckpt'],
            map_location=self.torch_device)

        mapping_model = torch.nn.Sequential(
            torch.nn.Linear(
                self.config['bert_hidden_size'],
                self.config['biggan_emb_size']
            )
        )

        mapping_model.load_state_dict(
            checkpoint['model_state_dict'])
        mapping_model.to(self.torch_device)
        mapping_model.eval()

        return mapping_model

    def encode_text_input(self, text):
        """

        Encodes the passed text with a Transformer model.

        :param text: text to encode (str)

        :return: Transformer hidden states (torch.Tensor),
                 size [max_seq_len x lm_hidden_size] (15 x 768)
        """

        pad_vid = self.tokenizer.vocab["[PAD]"]

        # encoding the text
        text_encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # padding short texts
        true_seq_length = text_encoded.size(1)
        pad_size = self.max_seq_length - true_seq_length

        # either padding or trimming
        if pad_size > 0:
            pad_ids = torch.Tensor([pad_vid] * pad_size).long().view([-1, pad_size])
            text_encoded_padded_or_trimmed = torch.cat((text_encoded, pad_ids), dim=1)
        else:
            text_encoded_padded_or_trimmed = text_encoded[:self.max_seq_length]
        
        text_encoded_padded_or_trimmed = text_encoded_padded_or_trimmed.to(self.torch_device)

        with torch.no_grad():
            # Last layer hidden-states are the first output of
            # Transformers' library models output tuple
            lm_hidden_states = self.language_model(text_encoded_padded_or_trimmed)[0]

            # Keep first example in the batch - output shape (max_seq_length, hidden size)
            lm_hidden_states_first_example = lm_hidden_states[0]

        return lm_hidden_states_first_example

    def create_biggan_embeggings(self,
                                 text,
                                 label):
        """

        :param text: input text (str)
        :param label: ImageNet class name (str)

        :return: BigGAN class embeddings (torch.Tensor),
                 size [max_seq_length x biggan_class_emb_dim] (15 x 128)
        """
        tokens = text.split()

        if len(tokens) < self.max_seq_length:
            tokens += ['[PAD]' for _ in range(self.max_seq_length - len(tokens))]
        else:
            tokens = tokens[:self.max_seq_length]

        try:
            # create a 1000-long OHE-vector indicating the ImageNet class for `label`
            # easier to use try/except logic here though if-else blocks are preferred
            class_vector = torch.tensor(one_hot_from_names(label))
        except AssertionError:
            # zeros if `label` is not found
            class_vector = torch.zeros([1, self.config['num_imagenet_classes']])
        
        class_vector = class_vector.to(self.torch_device)

        # BigGAN embeddings are identical for all words in `text`
        embs = [self.gan_model.embeddings(class_vector) for _ in tokens]

        with torch.no_grad():
            embs = torch.cat(embs)

        return embs

    def generate_images(self, dense_class_vector=None, name=None, noise_seed_vector=None,
                        truncation=0.4, batch_size=15):
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
        :param batch_size: number of images to generate
        :return: a tuple with two numpy arrays, the first one is shaped
                [batch_size, 128, 128, 3], the second one is a raw output
                from BigGAM generator, used to save images
        """

        self.logger.info('Generating images...')

        seed = int(noise_seed_vector.sum().item()) if noise_seed_vector is not None else None
        noise_vector = truncated_noise_sample(truncation=truncation,
                                              batch_size=batch_size, seed=seed)
        noise_vector = torch.from_numpy(noise_vector).to(self.torch_device)

        if name is not None:
            class_vector = one_hot_from_names([name], batch_size=batch_size)
            class_vector = torch.from_numpy(class_vector).to(self.torch_device)
            dense_class_vector = self.gan_model.embeddings(class_vector)
        else:
            dense_class_vector = dense_class_vector.view(-1, 128)
  
        input_vector = torch.cat([noise_vector, dense_class_vector], dim=1)

        # Generate an image
        with torch.no_grad():
            output = self.gan_model.generator(input_vector, truncation)

        output = output.cpu().numpy()
        output = output.transpose((0, 2, 3, 1))
        output = ((output + 1.0) / 2.0) * 256
        output.clip(0, 255, out=output)
        output = np.asarray(np.uint8(output), dtype=np.uint8)
        return output

    def play(self, text):

        words = text.split()

        hidden_states = self.encode_text_input(text)
        mapping_model_output = self.mapping_model(hidden_states)
        generated_images = self.generate_images(
            dense_class_vector=mapping_model_output)

        for i, img in enumerate(generated_images[1:len(words) + 1]):
            plt.figure(1);
            plt.clf()
            title = '{} {} {}'.format(" ".join(words[:i]),
                                      words[i].upper(),
                                      " ".join(words[i + 1:]))
            plt.title(title)
            plt.imshow(img)
            plt.pause(self.config['display_pause_time'])
            display.clear_output(wait=True)
        plt.close()


if __name__ == '__main__':
    model = TextToImageModel()

    while True:
        text = input("Insert text:\t")
        model.play(text)