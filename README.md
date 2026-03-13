# SPENav: Dynamic Object Filtering with Spatial Perception Enhancement for Vision-Language Navigation

## Project Introduction

SPENav (Spatial Perception Enhancement for Vision-Language Navigation) is a research project focused on the Visual Language Navigation (VLN) task, proposing a dynamic object filtering and spatial perception enhancement method to improve navigation performance through aggregating open-vocabulary perception and multi-level information modeling.

This project aims to address two main issues in existing methods:
1. Task-irrelevant cues commonly present in the environment can continuously introduce localization errors during navigation
2. Due to the lack of transferable general knowledge priors, existing agents exhibit notable limitations in spatial perception

### Core Innovations

- **Hierarchical Semantic Prior Extractor**: Constructs task-oriented semantic priors to capture critical objects and suppress irrelevant features
- **Room-Information-Guided Filtering**: Utilizes room-level information to filter out environmental distractions
- **Spatial-Instructional Guided Dual Attention Module**: Combines spatial information and instruction guidance to enable the agent to develop goal- and task-oriented selective memory
- **Open-Vocabulary Perception**: Integrates open-vocabulary capabilities to improve generalization performance in unseen environments
- **Multi-Level Information Modeling**: Integrates information at both local and global levels to enhance cross-modal understanding

### Main Features

- **Multiple Navigation Model Implementations**: Including CMA, DUET, GridMap, and VLNBERT models
- **Grid Map Construction**: Builds navigation grid maps using environmental information
- **Multi-Modal Fusion**: Fuses visual and language information for navigation decisions
- **Waypoint Prediction**: Predicts navigation waypoints through Transformer models
- **Multi-Dataset Support**: Compatible with mainstream VLN datasets like R2R
- **High Performance**: Achieves 76% Success Rate (SR) and 65% Success weighted by Path Length (SPL) on the unseen test split of R2R

## Project Structure

```
SPENav/
├── VLN_CE/               # Core navigation model implementation
│   ├── habitat/          # Habitat environment interface
│   ├── habitat_extensions/ # Environment extension functions
│   ├── vlnce_baselines/  # Baseline model implementations
│   │   ├── common/       # Common tools and components
│   │   ├── config/       # Model configuration files
│   │   └── models/       # Model definitions
│   ├── waypoint_prediction/ # Waypoint prediction module
│   ├── run.py            # Main running script
│   └── requirements.txt  # Dependencies
├── xlm-roberta-base/     # Pretrained language model
├── README.md             # Project description
└── requirements.txt      # Project dependencies
```

## Environment Requirements

- Python 3.7+
- PyTorch 1.7+
- Habitat-Sim
- Transformers
- NumPy
- SciPy

## Installation Steps

1. **Clone the project**
   ```bash
   git clone <repository-url>
   cd SPENav
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   cd VLN_CE
   pip install -r requirements.txt
   ```

3. **Download pretrained models**
   - Ensure the `xlm-roberta-base` directory contains the pretrained language model
   
4. **Configure dataset paths**
   - All dataset and model paths in configuration files need to be modified to the actual paths on your local hard drive
   - Especially the `ddppo_checkpoint` path in `run_GridMap.yaml`
   - And the model paths in `vlnce_baselines/models/gridmap/vlnbert_init.py`

## Usage

### Pretrain Model

Run the pretraining script:

```bash
bash pretrain-gate+gridEnhance2/run_r2r.sh
```

### Fine-tune Model

Run the fine-tuning script:

```bash
bash map_nav_src_fullmodule/scripts/run_r2r.sh
```

### Continuous Environment Run

Run the GridMap model in continuous environment:

```bash
bash VLN_CE/run_GridMap.bash
```

### Evaluate Model

Modify the `EVAL` option in the configuration file to `True`, then run the corresponding script:

```bash
bash VLN_CE/run_GridMap.bash
```

### Test Set Inference

Use the `test_set_inference.yaml` configuration file for test set inference:

```bash
python VLN_CE/run.py --config-path VLN_CE/vlnce_baselines/config/r2r_configs/ --config-name test_set_inference.yaml
```

## Model Description

### 1. CMA (Cross-Modal Attention)
- Navigation model based on cross-modal attention mechanism
- Fuses visual and language features for decision making

### 2. DUET
- Dual encoder structure that processes visual and language information separately
- Uses graph structure to represent environmental information

### 3. GridMap
- Builds grid map representation of the environment
- Uses grid map for navigation planning

### 4. VLNBERT
- BERT-based visual language navigation model
- Uses pretrained language model to improve performance

## Waypoint Prediction

The project includes a Transformer-based waypoint prediction module located in the `waypoint_prediction/` directory, used to predict key waypoints during navigation.

## Configuration Files

Model configuration files are located in the `VLN_CE/vlnce_baselines/config/` directory, containing hyperparameters and training settings for different models.

## License

This project uses the MIT License, see the `VLN_CE/LICENSE` file for details.

## Authors

- **Shuai Yuan** - yuan2645@gmail.com
- **Huaxiang Zhang**
- **Li Liu**
- **Lei Zhu**
- **Xinfeng Dong**

## Citation

If you use this project, please cite the related research paper:

```
@ARTICLE{11333293,
   author={Yuan, Shuai and Zhang, Huaxiang and Liu, Li and Zhu, Lei and Dong, Xinfeng},
   journal={IEEE Transactions on Circuits and Systems for Video Technology},
   title={SPENav: Dynamic Object Filtering with Spatial Perception Enhancement for Vision-Language Navigation},
   year={2026},
   volume={},
   number={},
   pages={1-1},
   keywords={Navigation;Semantics;Visualization;Feature extraction;Vocabulary;Object detection;Knowledge based systems;Three-dimensional displays;Random access memory;Memory management;vision language navigation;open-vocabulary perception;cross-modal understanding},
   doi={10.1109/TCSVT.2026.3651320}
}
```

### Abstract

The Vision-language navigation task requires agents to efficiently interpret visual cues in the environment and accurately follow long-range instructions, posing significant challenges to their scene memory and spatial reasoning capabilities. Existing methods typically construct memory systems directly from raw visual observations. However, task-irrelevant cues commonly present in the environment can continuously introduce localization errors during navigation, severely limiting the agent's performance in complex scenes. Meanwhile, due to the lack of transferable general knowledge priors, existing agents exhibit notable limitations in spatial perception, which undermines the reliability of their decision-making in unseen environments. To address these issues, this paper proposes the dynamic object filtering with Spatial Perception Enhancement for Vision-Language Navigation (SPENav), which aggregates open-vocabulary perception with multi-level information modeling. At the local level, the Hierarchical Semantic Prior Extractor and Room-Information-Guided Filtering construct task-oriented semantic priors to capture critical objects and suppress irrelevant features. At the global level, the Spatial-Instructional Guided Dual Attention module leverages spatial information and instruction guidance to enable the agent to develop selective memory that is goal- and task-oriented. On the unseen test split of R2R, SPENav achieves a 76% Success Rate (SR) and a 65% Success weighted by Path Length (SPL). These results demonstrate the effectiveness of task-oriented feature selection and multi-level semantic modeling in enhancing cross-modal understanding and adaptive navigation performance.

## Contact

For questions or suggestions, please contact us through:
- Email: yuan2645@gmail.com
- GitHub Issues: https://github.com/yuan-jac/SPENav/issues
