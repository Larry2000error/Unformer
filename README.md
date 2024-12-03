# Unformer
The source code for paper "Uncertainty-Driven Multi-View Contrastive Learning for Multimodal Relation Extraction under Unpaired Data"
# Abstract
Despite advances in multimodal relation extraction (MRE) by deep learning, existing methods often overlook the challenge posed by unpaired multimodal data, which is a common issue in its open-world applications. These approaches typically assume all visual-text pairs are perfectly matched with strong semantic relations, failing to account for the variability and complexity inherent in data from social media. 
 To this end, this paper is the first attempt to explore the problem by introducing a novel benchmark termed Multimodal Relation Extraction under Unpaired Data (MREUD).
 To improve the mismatched robustness of MRE models, we propose an innovative yet practical Uncertainty-driven Multi-view Contrastive Learning strategy (UMCL), which contains two cascading crucial concepts: Equitable and Robust Multimodal Feature Extractor (ERMFE) and Heteroscedastic Uncertainty-Driven Gaussian Modeling (HUDGM).Specifically, the former ensures the balanced feature representations across visual-textual modalities, while the latter leverages data uncertainty to improve model adaptability in unpaired multimodal data. Extensive experiments validate the effectiveness of our method, which achieves state-of-the-art performance in contrast to competitive MRE approaches. This work lays a practical foundation for improving MRE systems, making them more applicable to the diverse and dynamic conditions found in open-world environments.
 
## Data preprocessing

### MNRE dataset

Due to the large size of MNRE dataset, please download the dataset from the [original repository](https://github.com/thecharm/MNRE). 

Unzip the data and rename the directory as `mnre`, which should be placed in the directory `data`:

```bash
mkdir data logs ckpt
```

We also use the detected visual objects provided in [previous work](https://github.com/zjunlp/MKGformer), which can be downloaded using the commend:

```bash
cd data/
wget 120.27.214.45/Data/re/multimodal/data.tar.gz
tar -xzvf data.tar.gz
```

## Dependencies

Install all necessary dependencies:

```bash
pip install -r requirements.txt
```

## Training the model

The best hyperparameters we found have been witten in `run_mre.sh` file.

You can simply run the bash script for multimodal relation extraction:

```bash
bash run_mre.sh
```
