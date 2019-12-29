# 🤗Hugging Face ML researcher/engineer code exercise - Funky mutli-modal version

By [Yury Kashnitsky](https://yorko.github.io/)

There are 4 parts that I present here:
1. A Jupyter notebook which expressess the way I tackled the problem
1. Readme here is mostly populated with the content of that Jupyter notebook 
1. Console utility to play around with the model
1. Streamlit application (WIP)

# Part 1. Jupyter
[Jupyter Notebook](notebooks/simple_mapping_model_coco_captions.ipynb), stand-alone [HTML version](notebooks/simple_mapping_model_coco_captions.html). This is a "researchy" Jupyter notebook, it gives a good idea of how I approached the problem, it's also good to analyze model behavior and, in general, to present stuff. But the code quality shall not be judged here :) For this purpose, better refer to parts 3 and 4. 

# Part 2. README
## Intro
Here I train a mapping model with COCO captions. I realized that COCO captions is almost an ideal source to train the mapping model, although it covers a bit more than 50% of ImageNet classes and only 155 are left when we select those classes known to BERT (possible [improvements](#Possible-improvements) are discussed in the end).

I select only those captions containing exactly **one** word from ImageNet classes.

a [UNK] stop sign and a red [UNK] on the road ⟶ sign <br>
the vanity contains two sinks with a towel for each ⟶ towel<br>
 a person [UNK] from the rain by their umbrella walks down the road ⟶ umbrella<br>
 two bicycles and a woman walking in front of a shop ⟶ shop <br>
 a brown horse is grazing grass near a red house ⟶ horse  <br>


The model itself is just a dense layer, so it performs a linear mapping of BERT hidden states to BigGAN class embeddings.

Mapping model input is sized `[max_seq_len x lm_hidden_size]` [15 x 768] <br>
Mapping model output is sized `[max_seq_len x biggan_class_emb_dim]` [15 x 128]

The model behavior is [analyzed](#Playing-around-with-the-model) in the end of this notebook. Also, [possible improvements](#Possible-improvements) are listed. 

<img src="https://habrastorage.org/webt/ar/ed/sc/aredscxfrkrucqkmz3qfkslenfa.png" />

## COCO captions
I utilize the `load_coco_data` script from Stanford's cs231n course and end up with ~400k training captions and 200k validation captions. Then I select only captions containing exactly one ImageNet word, thus left with 162k train captions and 79k validation captions.


## Mapping model
Mapping model is just a Linear layer. It's output is compared to true BigGAN embeddings via MSE loss.

<img src="https://habrastorage.org/webt/8w/0p/gi/8w0pgizh3_vhgaadly3tmsg3vnu.png" width=50% />

- `X` - BERT hidden states for the text input, `[max_seq_len x bert_hidden_size]`
- `Z` - BigGAN embeddings for each input token, `[max_seq_len x gan_emb_size]`
- `W` - mapping matrix, learned below, `[bert_hidden_size x gan_emb_size]`
- `max_seq_len` = 15 - maximal sequence length in words, shorter texts are padded, longer ones are trimmed
- `bert_hidden_size` = 768 - dimensionality of BERT (or DistilBERT) LM output hidden states
- `gan_emb_size` = 128 - dimensionality of BigGAN class embeddings

## Training the mapping model
Described in the [notebook](notebooks/simple_mapping_model_coco_captions.ipynb), the actual training is done with a script based on that notebook (not tidied up and thus not attached here).

## Playing around with the model
### Good cases
This approach works fairly well with classes well presented in COCO captions, eg. dog, bus, shop, coffee, sign. 

<img src="https://habrastorage.org/webt/e2/td/ns/e2tdnsciohg3dhcphp02fywsc7y.gif" width=30%/>

Creating GIFs is a bit tedious, so please look for more examples in the notebook. Also, these animations can be played with saved images:

> python scripts/display_image_series.py --path_to_gen_images img --file_mask ex1_dog --text "a dog is faster than a cat"

<img src="https://habrastorage.org/webt/mx/a1/g9/mxa1g9efast3iwzlcunqpptai-8.gif" width=30%/>

> python scripts/display_image_series.py --path_to_gen_images img --file_mask ex2_bus --text "i went there by bus"

<img src="https://habrastorage.org/webt/2i/mm/6m/2imm6mhqueplxlkvagidnfgikdq.gif" width=30%/>

> python scripts/display_image_series.py --path_to_gen_images gen_images --file_mask ex5_sign_building  --text "there is a sign in front of a building"


### Bad cases
Some of classes are present in COCO captions but still are dominated by other objects. Other classes are just not present in COCO captions, eg. cobra or volcano. 

<img src="https://habrastorage.org/webt/ce/yl/af/ceylaf77oqokvrxbxmiezya-qho.gif" width=30% />

> python scripts/display_image_series.py --path_to_gen_images img --file_mask ex7_cobra --text "cobra is a dangerous animal"

The model doesn't seem to cope with >1 class in a caption. In this example, we have both "elephant" and "bus", and some funny mixtures of animals and means of transport are produced. 

<img src="https://habrastorage.org/webt/sv/_j/vx/sv_jvxa2xtnpu2g4xhx4q-ng7bs.gif" width=30% />

> python scripts/display_image_series.py --path_to_gen_images img --file_mask ex9_elephant  --text "an elephant is slower than a bus"

## Retrospective

### Good traits of the built solution

- The mapping is done with a simple model (a linear mapping) and it's proved to work for some ImageNet classes
- The model copes fairly enough with images corresponding to ImageNet classes that are well presented in COCO captions. eg. bus, sign, coffee or, of course, dog. 

### Bad traits of the built solution

- Only 155 ImageNet classes are there in training COCO captions, so we fail to visualize some concepts like cobra or volcano
- The model doesn't seem to cope with >1 class in a caption, eg. "a dog is faster that a cat" - only dogs are drawn. Or "an elephant is slower than a bus" - some funny mixtures of animals and means of transport are produced.
- Text is split by spaces for visualization word by word, thus plural forms or punctuation can spoil the result (easy to fix though)

### What didn't work

- Keeping original captions with up to 8 ImageNet words in each one. Thus the target BigGAN embedding matrix contains embeddings for more than one class. In my case the model failed to learn anything
- Replacing simple linear mapping with a more complex MLP didn't improve results
- Tried adding simple patterns to COCO captions, actually had a template for several patterns, each one produced 1k examples like "I noticed a truck", "she saw a sign" etc. The model didn't train well for me, probably just a matter of learning rate


### Possible improvements

> List some improvements you would propose to improve this data generation process 

Here I address both "Building a dataset of text sequence associated to ImageNet classes" and "Building a dataset for training a mapping function" parts of the provided assignment. 

- Here I used COCO captions. This covered 155 ImageNet classes out of 598 that we used here. But actually, any text will do as long as it contains ImageNet classes. So the training dataset can be extended with eg. extracts from Wikipedia pages with corresponding words ("Lion", "Cheetah", "Volcano" etc.)
- Adding synthetic captions. With a template like "<SUBJ> <VERB> <WORD> <PLACE>" we can generate as many captions as we wish if we vary subject (I, he, she, we, they etc.), verb (saw, noticed, spotted etc.), place (here, there, in front of the building etc) and word (1k ImageNet classes). As I mentioned, I didn't train a better model with these synthetic captions, but still it's a good way to extend the training set for the mapping model
- Augmentations, eg. synonym replacement. We didn't find the word "beast" in COCO captions, but we can take existing ones with words "lion" or "tiger" and replace them with the word "beast" to get new captions. 
    
More remarks:
 - The training dataset can be further enlarged with other captions, eg. [Conceptual Captions](https://ai.googleblog.com/2018/09/conceptual-captions-new-dataset-and.html)
 - Technically, training could have been done better, eg. with LR Finder
 - A more complicated model can be trained, eg. a MLP
 
# Part 3. Console utility

# Part 4. Streamlit application 
Work in progress. This will be a fully dockerized application, ready to be deployed to a AWS instance or similar cloud service. 