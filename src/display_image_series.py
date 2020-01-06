import argparse
from pathlib import Path
from matplotlib import pyplot as plt
from IPython import display


def slideshow(text, path_to_gen_images, file_mask, pause=0.5):
    """

    Runs slideshow for a given text phrase using saved earlier images
    corresponding to each word in a phrase.

    :param text: text to be displayed word by word, str
    :param path_to_gen_images: ath to images generated with the model, str
    :param file_mask: file mask for a series of images to be displayed
                    eg. "ex1_dog" or "ex2_bus", look at the img folder
    :param pause: time interval between to consequent slides, float
    :return: None
    """

    words = text.split()

    images = [plt.imread(str(f)) for f in
              sorted(Path(path_to_gen_images).glob(file_mask + '*.png'))]

    for i, img in enumerate(images):
        plt.figure(1, frameon=False)
        plt.clf()

        title = '{} {} {}'.format(" ".join(words[:i]),
                                  words[i].upper(),
                                  " ".join(words[i + 1:]))
        plt.title(title)
        plt.imshow(img)
        plt.pause(pause)
        display.clear_output(wait=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--text", type=str,
                        help="Text to be displayed word by word")
    parser.add_argument("--path_to_gen_images", default='../img', type=str,
                        help="Path to images generated with the model")
    parser.add_argument("--file_mask",  type=str,
                        help="File mask for a series of images to be displayed")

    args = parser.parse_args()

    slideshow(text=args.text,
              path_to_gen_images=args.path_to_gen_images,
              file_mask=args.file_mask)
