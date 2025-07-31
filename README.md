# RQ-VAE Recommender
This is a PyTorch implementation of a generative retrieval model using semantic IDs based on RQ-VAE from "Recommender Systems with Generative Retrieval". 
The model has two stages:
1. Items in the corpus are mapped to a tuple of semantic IDs by training an RQ-VAE (figure below).
2. Sequences of semantic IDs are tokenized by using a frozen RQ-VAE and a transformer-based is trained on sequences of semantic IDs to generate the next ids in the sequence.
![image](https://github.com/EdoardoBotta/RQ-VAE/assets/64335373/199b38ac-a282-4ba1-bd89-3291617e6aa5)

### Currently supports
* **Datasets:** Amazon Reviews (Beauty, Sports, Toys), MovieLens 1M, MovieLens 32M
* RQ-VAE Pytorch model implementation + KMeans initialization + RQ-VAE Training script.
* Decoder-only retrieval model + Training code with semantic id user sequences from randomly initialized or pretrained RQ-VAE.

### 🤗 Usage on Hugging Face 
RQ-VAE trained model checkpoints are available on Hugging Face 🤗: 
* [**RQ-VAE Amazon Beauty**](https://huggingface.co/edobotta/rqvae-amazon-beauty) checkpoint.

### Installing
Clone the repository and run `pip install -r requirements.txt`. 

No manual dataset download is required.

### Executing
RQ_VAE tokenizer model and the retrieval model are trained separately, using two separate training scripts. 
#### Custom configs
Configs are handled using `gin-config`. 

The `train` functions defined under `train_rqvae.py` and `train_decoder.py` are decorated with `@gin.configurable`, which allows all their arguments to be specified with `.gin` files. These include most parameters one may want to experiment with (e.g. dataset, model sizes, output paths, training length). 

Sample configs for the `train.py` functions are provided under `configs/`. Configs are applied by passing the path to the desired config file as argument to the training command. 
#### Sample usage
To train both models on the **Amazon Reviews** dataset, run the following commands:
* **RQ-VAE tokenizer model training:** Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py configs/rqvae_amazon.gin`
* **Retrieval model training:** Trains retrieval model using a frozen RQ-VAE: `python train_decoder.py configs/decoder_amazon.gin`

To train both models on the **MovieLens 32M** dataset, run the following commands:
* **RQ-VAE tokenizer model training:** Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py configs/rqvae_ml32m.gin`
* **Retrieval model training:** Trains retrieval model using a frozen RQ-VAE: `python train_decoder.py configs/decoder_ml32m.gin`

### Next steps
* Comparison encoder-decoder model vs. decoder-only model.

### References
* [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) by Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy
* [Categorical Reparametrization with Gumbel-Softmax](https://openreview.net/pdf?id=rkE3y85ee) by Eric Jang, Shixiang Gu, Ben Poole
* [Restructuring Vector Quantization with the Rotation Trick](https://arxiv.org/abs/2410.06424) by Christopher Fifty, Ronald G. Junkins, Dennis Duan, Aniketh Iger, Jerry W. Liu, Ehsan Amid, Sebastian Thrun, Christopher Ré
* [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) by lucidrains
* [deep-vector-quantization](https://github.com/karpathy/deep-vector-quantization) by karpathy
* 







          
# Generative Recommendation based on HiD-VAE
This is a PyTorch implementation of a generative retrieval model using semantic IDs based on HiD-VAE, addressing limitations in existing methods by learning hierarchically disentangled item representations. The model overcomes semantic flatness and representation entanglement through hierarchically-supervised quantization and a uniqueness loss, enabling interpretable and diverse recommendations.

The framework operates in two stages:
1. Items are mapped to hierarchically disentangled semantic IDs using HiD-VAE, supervised by multi-level tags.
2. A transformer-based model is trained on sequences of these IDs to generate recommendations.

![image](https://github.com/user/repo/blob/main/fig/intro.png)  
*(Figure illustrating HiD-VAE's hierarchical disentanglement compared to traditional methods, showing reduced ID collisions and interpretable semantic paths.)*

### Currently supports
* **Datasets:** Amazon Reviews (Beauty, Sports), KuaiRand
* HiD-VAE PyTorch model with hierarchical supervision + Training script.
* Transformer-based retrieval model + Training code using pretrained HiD-VAE.

### Installing
Clone the repository and run `pip install -r requirements.txt`.

No manual dataset download is required.

### Dataset Processing
Datasets are automatically processed when training. For example:
- **Amazon Beauty:** Set `dataset_folder="dataset/amazon"` and `dataset_split="beauty"` in the config file. The script will handle embedding generation and hierarchical tag creation using LLM if needed.
- **KuaiRand:** Set `dataset_folder="dataset/kuairand"`. For datasets lacking explicit tags like KuaiRand, the framework uses an LLM-based approach to generate high-quality hierarchical tags automatically during processing.

### Training
HiD-VAE tokenizer model and the retrieval model are trained separately using two scripts.

#### Custom configs
Configs are handled using `gin-config`. The `train` functions in `train_h_rqvae.py` and `train_decoder.py` are configurable via `.gin` files under `configs/`. Apply configs by passing the path as an argument.

#### Sample usage
To train on **Amazon Beauty** dataset:
* **HiD-VAE tokenizer training:** Trains HiD-VAE with hierarchical supervision. Executed via `python train_h_rqvae.py configs/h_rqvae_amazon.gin` (set `dataset_split="beauty"`).
* **Transformer training:** Trains the retrieval model using frozen HiD-VAE: `python train_decoder.py configs/decoder_amazon.gin` (set `dataset_split="beauty"` and point `pretrained_rqvae_path` to the trained HiD-VAE checkpoint).

For **KuaiRand**:
* Use `configs/h_rqvae_kuairand.gin` for HiD-VAE and `configs/decoder_kuairand.gin` for the transformer, following similar commands.

### Next steps
*  Incorporate multi-modal data for richer representations and integrate LLMs for better sequential modeling

        
  
