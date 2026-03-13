# SPENav: Dynamic Object Filtering with Spatial Perception Enhancement for Vision-Language Navigation

<div align="center">

![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyTorch 1.7+](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
[![arXiv](https://img.shields.io/badge/arXiv-10.1109%2FTCSVT.2026.3651320-red)](https://doi.org/10.1109/TCSVT.2026.3651320)

</div>

## 📖 Project Introduction

SPENav (Spatial Perception Enhancement for Vision-Language Navigation) is a research project focused on the Visual Language Navigation (VLN) task. We propose a **dynamic object filtering** and **spatial perception enhancement** method to improve navigation performance through aggregating open-vocabulary perception and multi-level information modeling.

This project aims to address two main issues in existing methods:

1. **Task-irrelevant cues** commonly present in the environment can continuously introduce localization errors during navigation.
2. **Limited spatial perception** due to the lack of transferable general knowledge priors in existing agents.

### 🌟 Core Innovations

| Component | Description |
|-----------|-------------|
| **Hierarchical Semantic Prior Extractor** | Constructs task-oriented semantic priors to capture critical objects and suppress irrelevant features |
| **Room-Information-Guided Filtering** | Utilizes room-level information to filter out environmental distractions |
| **Spatial-Instructional Guided Dual Attention Module** | Combines spatial information and instruction guidance for goal-oriented memory |
| **Open-Vocabulary Perception** | Integrates open-vocabulary capabilities for better generalization |
| **Multi-Level Information Modeling** | Integrates information at local and global levels |

### 🚀 Main Features

- **Multiple Navigation Model Implementations**: CMA, DUET, GridMap, and VLNBERT models
- **Grid Map Construction**: Builds navigation grid maps using environmental information
- **Multi-Modal Fusion**: Fuses visual and language information for navigation decisions
- **Waypoint Prediction**: Transformer-based navigation waypoint prediction
- **Multi-Dataset Support**: Compatible with R2R, REVERIE, SOON, and RxR datasets
- **High Performance**: Achieves **76% Success Rate (SR)** and **65% SPL** on R2R unseen test split

---

## 📁 Project Structure

```
SPENav/
├── VLN_CE/                      # Core navigation model for continuous environments
│   ├── habitat/                 # Habitat environment interface
│   ├── habitat_extensions/      # Environment extension functions
│   ├── vlnce_baselines/        # Baseline model implementations
│   │   ├── common/             # Common utilities
│   │   ├── config/             # Model configurations
│   │   └── models/             # Model definitions (CMA, DUET, GridMap, VLNBERT)
│   ├── waypoint_prediction/    # Waypoint prediction module
│   ├── run.py                  # Main running script
│   └── requirements.txt        # Dependencies
├── map_nav_src_fullmodule/      # Fine-tuning module
├── pretrain-gate+gridEnhance2/ # Pretraining module
├── data/                        # Raw data (needs manual setup)
├── datasets/                    # Preprocessed features (needs manual setup)
├── preprocess/                  # Feature extraction scripts
├── xlm-roberta-base/           # Pretrained language model
└── README.md                    # This file
```

---

## 🛠️ Installation & Setup

### Step 1: Environment Configuration

```bash
# Clone the project
git clone https://github.com/yuan-jac/SPENav.git
cd SPENav

# Create and activate conda environment
conda create -n spenav python=3.8
conda activate spenav

# Install dependencies
pip install -r requirements.txt
cd VLN_CE
pip install -r requirements.txt
cd ..
```

### Step 2: Download Dataset and Simulators

**VLN-CE Dataset**: Download from [Google Drive](https://drive.google.com/drive/folders/1544Lb4mySyTsh3aKOIzag4_R7HCeULEV?usp=drive_link)

**Matterport3D Simulator**: Follow the instructions [here](https://github.com/peteanderson80/Matterport3DSimulator)

**VLN-CE Repository**: Reference [here](https://github.com/jacobkrantz/VLN-CE)

**DINOv2**: Follow the instructions [here](https://github.com/facebookresearch/dinov2)

**SigLIP 2**: Follow the instructions [here](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md)

**StartQwen** (for large model service and spatial dataset processing): Follow the instructions [here](https://github.com/yuan-jac/StartQwen/tree/master)

### Step 3: Prepare Dataset and Data Folders

Place the downloaded `datasets/` and `data/` folders directly into the project root directory.

---

## 💻 Usage

### Pretrain
```bash
bash pretrain-gate+gridEnhance2/run_r2r.sh
```

### Fine-tune
```bash
bash map_nav_src_fullmodule/scripts/run_r2r.sh
```

### Continuous Environment (R2R-CE)
```bash
bash VLN_CE/run_GridMap.bash
```

### Evaluation
Set `EVAL: True` in the configuration file, then run:
```bash
bash VLN_CE/run_GridMap.bash
```

### Test Set Inference
```bash
python VLN_CE/run.py --config-path VLN_CE/vlnce_baselines/config/r2r_configs/ --config-name test_set_inference.yaml
```

---

## 🧠 Model Description

### 1. CMA (Cross-Modal Attention)
Navigation model based on cross-modal attention mechanism. Fuses visual and language features for decision making.

### 2. DUET
Dual encoder structure that processes visual and language information separately. Uses graph structure for environmental representation.

### 3. GridMap
Builds grid map representation of the environment. Uses grid map for navigation planning.

### 4. VLNBERT
BERT-based visual language navigation model. Leverages pretrained language models for improved performance.

---

## 📍 Waypoint Prediction

The project includes a Transformer-based waypoint prediction module located in `waypoint_prediction/`, used to predict key waypoints during navigation.

---

## ⚙️ Configuration Files

Model configurations are in `VLN_CE/vlnce_baselines/config/`, containing hyperparameters and training settings for different models.

---

## 📝 Citation

If you use this project, please cite:

```bibtex
@ARTICLE{11333293,
   author={Yuan, Shuai and Zhang, Huaxiang and Liu, Li and Zhu, Lei and Dong, Xinfeng},
   journal={IEEE Transactions on Circuits and Systems for Video Technology},
   title={SPENav: Dynamic Object Filtering with Spatial Perception Enhancement for Vision-Language Navigation},
   year={2026},
   doi={10.1109/TCSVT.2026.3651320}
}
```

### Abstract

The Vision-language navigation task requires agents to efficiently interpret visual cues in the environment and accurately follow long-range instructions, posing significant challenges to their scene memory and spatial reasoning capabilities. Existing methods typically construct memory systems directly from raw visual observations. However, task-irrelevant cues commonly present in the environment can continuously introduce localization errors during navigation, severely limiting the agent's performance in complex scenes. Meanwhile, due to the lack of transferable general knowledge priors, existing agents exhibit notable limitations in spatial perception, which undermines the reliability of their decision-making in unseen environments. To address these issues, this paper proposes the dynamic object filtering with Spatial Perception Enhancement for Vision-Language Navigation (SPENav), which aggregates open-vocabulary perception with multi-level information modeling. At the local level, the Hierarchical Semantic Prior Extractor and Room-Information-Guided Filtering construct task-oriented semantic priors to capture critical objects and suppress irrelevant features. At the global level, the Spatial-Instructional Guided Dual Attention module leverages spatial information and instruction guidance to enable the agent to develop selective memory that is goal- and task-oriented. On the unseen test split of R2R, SPENav achieves a 76% Success Rate (SR) and a 65% Success weighted by Path Length (SPL). These results demonstrate the effectiveness of task-oriented feature selection and multi-level semantic modeling in enhancing cross-modal understanding and adaptive navigation performance.

---

## � Authors

| Name | Email |
|------|-------|
| Shuai Yuan | yuan2645@gmail.com |
| Huaxiang Zhang | - |
| Li Liu | - |
| Lei Zhu | - |
| Xinfeng Dong | - |

---

## 📧 Contact

- **Email**: yuan2645@gmail.com
- **GitHub Issues**: [https://github.com/yuan-jac/SPENav/issues](https://github.com/yuan-jac/SPENav/issues)

---

<div align="center">

**⭐ Star us on GitHub if you find this project useful!**

</div>
