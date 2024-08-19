# Cortex-SSL

This repository is the official implementation of our conference paper titled [*Self-Supervised Pretraining for Cortical Surface Analysis*](https://link.springer.com/chapter/10.1007/978-3-031-66955-2_7) published in Medical Image Understanding and Analysis.

<img src="images/ssl_full.drawio.svg"/>

*In Phase I, the GNN autoencoder is trained to reconstruct the original cortical surfaces from their masked versions. In Phase II, the pretrained encoder weights are leveraged for supervised fine-tuning, focusing on target tasks such as segmentation and age prediction.*

If you use our code or results, please cite our paper and consider giving this repo a :star: :
```
@inproceedings{unyi2024cortexssl,
  title={Self-Supervised Pretraining for Cortical Surface Analysis},
  author={Unyi, D{\'a}niel and Gyires-T{\'o}th, B{\'a}lint},
  booktitle={Annual Conference on Medical Image Understanding and Analysis},
  pages={96--108},
  year={2024},
  organization={Springer}
}
```

## How to run

The scripts are designed to be executed on a computing cluster managed by SLURM. SLURM is required to distribute the workload across multiple nodes and manage the job resources.

### Pretraining

To run the pretraining phase, use the provided `pretrain.sh` script with the following command:
```bash
sbatch pretrain.sh
```

### Segmentation

To run the segmentation task, use the provided `segmentation.sh` script with the following command:
 ```bash
 sbatch segmentation.sh
 ```
Make sure the pretrained model is available.

The key parameters `pretrained`, `frozen`, and `num_labeled`, can be adjusted within the script:
- pretrained: Whether to use the pretrained model weights (True or False).
- frozen: Whether to freeze the pretrained model weights (True or False).
- num_labeled: Number of labeled samples to use for fine-tuning (e.g., 7).

We conducted two types of experiments for segmentation:

1. **Few-Shot Learning:**
   - **Configuration:** We used `pretrained=True`, `frozen=False`, and `num_labeled=1/7/14/etc`.
   - **Baseline:** The results were compared with a randomly initialized model (`pretrained=False`, `frozen=False`) using the same number of labeled samples (`num_labeled=1/7/14/etc`).

2. **Linear Probing:**
   - **Configuration:** We used `pretrained=True`, `frozen=True`, and `num_labeled=70`.
   - **Baseline:** The results were compared with a randomly initialized model (`pretrained=False`, `frozen=False`) with `num_labeled=70`.

In both cases, we used 10 samples for validation and 21 samples for testing.

### Age Prediction

To run the age prediction task, use the provided `age_prediction.sh` script with the following command:
 ```bash
 sbatch age_prediction.sh
 ```
Everything is set up similarly to segmentation, but since this is a regression task, we only experimented with **Linear Probing:**
   - **Configuration:** We used `pretrained=True`, `frozen=True`, and `num_labeled=70`.
   - **Baseline:** The results were compared with a randomly initialized model (`pretrained=False`, `frozen=False`) with `num_labeled=70`.

## Accessing the Datasets

The HCP 1200 subjects data release can be easily downloaded following [this repo](https://github.com/jokedurnez/HCP_download) (and don't forget to give it a :star:).

The MindBoggle-101 dataset can be downloaded from [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/mindboggle).

## Acknowledgments

The authors are grateful to Petar Veličković for supporting the project with his valuable insights. The work reported in this paper carried out at BME, has been supported by the European Union project RRF-2.3.1-21-2022-00004 within the framework of the Artificial Intelligence National Laboratory. Project no. TKP2021-NVA-02 has been implemented with the support provided by the Ministry of Culture and Innovation of Hungary from the National Research, Development and Innovation Fund, financed under the TKP2021-NVA funding scheme. The presented work of D. Unyi was also supported by the ÚNKP-23-3-II-BME-399 New National Excellence Program of the Ministry for Culture and Innovation from the source of the National Research, Development and Innovation Fund. We thank the Governmental Agency for IT Development (KIFÜ) for the opportunity provided by the Komondor supercomputer, which they operate, and where the computations were performed.
