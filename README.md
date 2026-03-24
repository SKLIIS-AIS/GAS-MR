# FedGAC: Geometric Alignment Constraints for Mitigating Membership Inference in Federated Learning

This article has been accepted by IJCNN 2026, and the original author's project link is https://github.com/yuy63434-tech/GAS-MR

This repository provides the official implementation of **FedGAC**, a lightweight geometric alignment constraint for mitigating membership inference attacks in federated learning.

FedGAC regularizes the geometry of generated representations using classifier-induced class-wise references, without modifying the federated learning protocol or local training procedure.

---

## Repository Structure

```
.
├── fedgac.py                # Main federated training pipeline
├── model.py                 # Model architectures
├── config.py                # Training configuration
├── cvdataset.py             # Dataset loading and preprocessing
├── data_partition.py        # Data partition strategies (IID / non-IID)

├── gen_utils/               # Data generation utilities
│   ├── fedgen.py
│   ├── generate_image.py
│   └── generate_utils.py

├── mi_loss_attack.py        # Loss-based MIA (LB-MIA)
├── mi_feature_attack.py     # Feature-based MIA (FedMIA)

├── requirements.txt
└── README.md
```

---

## Environment Setup

Tested with:

* Python 3.8+
* PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run federated training:

```bash
python fedgac.py
```

Key configurations (dataset, partition strategy, number of clients, etc.) are defined in:

```
config.py
```

---

## Membership Inference Attacks

We provide two types of server-side MIA:

* **Loss-based attack (LB-MIA)**

  ```bash
  python mi_loss_attack.py
  ```

* **Feature-based attack (FedMIA)**

  ```bash
  python mi_feature_attack.py
  ```

Evaluation metrics include:

* AUC
* TPR@1%
* TPR@0.1%

---

## Data Partition

Data partition strategies are implemented in:

```
data_partition.py
```

Supported settings:

* IID
* Dirichlet-based non-IID
* Class-skew non-IID

---

## Reproducibility

* Random seeds are controlled in the scripts
* Key hyperparameters are defined in `config.py`

---

## Notes

* This code focuses on **server-side membership inference attacks**

* No additional communication overhead is introduced
* Raw client data is not required

---

## Citation

If you find this work useful, please cite:

```
@inproceedings{FedGAC,
  title     = {Geometric Alignment Constraints for Mitigating Membership Inference in Federated Learning},
  author    = {Anonymous},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026}
}
```

---

  Contact

For questions, please open an issue in this repository.
