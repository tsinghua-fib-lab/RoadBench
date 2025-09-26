# RoadBench

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

RoadBench evaluates MLLMs across six distinct tasks that are fundamental to road network understanding:

### 📋 Task Categories

#### **Bird's-Eye View (BEV) Tasks**
- **Task 1: Lane Counting** - Count lanes from aerial/satellite perspective with reference line guidance
- **Task 2: Lane Designation Recognition** - Identify lane purposes from bird's-eye view with directional annotations
- **Task 3: Road Network Correction** - Detect and correct road network topology errors

#### **First-Person View (FPV) Tasks**
- **Task 4: Lane Counting** - Determine the number of available lanes for vehicle travel
- **Task 5: Lane Designation Recognition** - Identify turning directions and lane purposes (straight, left-turn, right-turn, u-turn)
- **Task 6: Road Type Classification** - Distinguish between main roads and service roads

## 🏗️ Architecture

### Core Components

- **`roadnetbenchmark/`** - Core benchmark library
  - `vlm_client.py` - Universal VLM client supporting OpenAI-compatible APIs
  - `image.py` - Image processing and annotation utilities
  - `metric.py` - Evaluation metrics including F1 scores and geometric distance measures
  - `coord.py` - Coordinate transformation utilities
  - `satellite.py` - Satellite imagery processing
  - `concurrent_jsonl_writer.py` - Efficient concurrent data I/O

### Task Scripts

Each task includes three types of scripts:
- **Evaluation Scripts** (`*_*.py`) - Run VLM inference on datasets
- **Metrics Scripts** (`*_*_metrics.py`) - Calculate performance metrics
- **Shell Scripts** (`*_*.sh`) - Automated execution with various configurations

## 📊 Evaluation Metrics

RoadBench employs multiple sophisticated metrics tailored to different task types:

### **Geometric Metrics**
- **Junction Point Distance** - Assesses accuracy of road segment termination points
- **Fréchet Distance** - Evaluates similarity of road line trajectories
- **Buffer F1 Score** - Measures overlap between predicted and ground truth road geometries

### **Multi-Class Classification Metrics**
- **Weighted F1 Score** - Balanced precision and recall weighted by class frequency
- **Weighted Precision/Recall** - Class-weighted precision and recall metrics
- **RMSE** - Root Mean Square Error for lane counting regression tasks

### **Multi-Label Classification Metrics**
- **Hamming Loss** - Proportion of incorrect label predictions across all lane designations
- **Exact Match Ratio (Accuracy)** - Percentage of samples where all lane designations are correctly predicted

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**

2. **Install dependencies:**
```bash
# Using UV (recommended)
uv install

# Or using pip
pip install -e .
```

3. **Set up environment variables:**
```bash
# Create .env file with your API keys
cp .env.example .env
# Edit .env with your VLM API credentials
```

### Environment Variables

Create a `.env` file with the following variables:

```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# For other VLM providers, add respective API keys
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

## 📖 Usage

### Running Individual Tasks

#### Lane Counting (BEV)
```bash
python bev_lane_counting.py
```

#### Lane Designations (BEV)
```bash
python bev_lane_designations.py
```

#### Road Network Correction (BEV)
```bash
python bev_roadnet_correction.py
```

#### Lane Counting (FPV)
```bash
python fpv_lane_counting.py
```

#### Lane Designations (FPV)
```bash
python fpv_lane_designations.py
```

#### Road Type Classification (FPV)
```bash
python fpv_road_type_classification.py
```

### Computing Metrics

After running evaluations, compute performance metrics:

```bash
# Calculate BEV lane counting metrics
python bev_lane_counting_metrics.py

# Calculate BEV lane designations metrics
python bev_lane_designations_metrics.py

# Calculate BEV road network correction metrics
python bev_roadnet_correction_metrics.py

# Calculate FPV lane counting metrics
python fpv_lane_counting_metrics.py

# Calculate FPV lane designations metrics
python fpv_lane_designations_metrics.py

# Calculate FPV road type classification metrics
python fpv_road_type_classification_metrics.py
```

## 📁 Dataset Structure

```
data/
├── task_1_2/          # BEV Lane Counting & Designations
│   └── dataset/
│       ├── *.png     # BEV road images
│       └── labels.jsonl
├── task_3/            # BEV Road Network Correction
├── task_4_5/          # FPV Lane Tasks
└── task_6/            # FPV Road Type Classification
```
