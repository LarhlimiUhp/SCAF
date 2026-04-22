# SCAF-LS Dependencies Setup

## Overview
SCAF-LS (Scalable Cross-Asset Forecasting - Long Short) requires several Python dependencies for machine learning, data processing, and visualization.

## Files
- `requirements.txt`: Complete list of dependencies with version constraints
- `install_dependencies.py`: Automated installation script

## Quick Setup

### Option 1: Automated Installation (Recommended)
Run the installation script:
```bash
python install_dependencies.py
```

This script will:
- Check Python version (requires 3.11+)
- Upgrade pip
- Install all dependencies
- Verify installation by importing key modules

### Option 2: Manual Installation
Install dependencies manually:
```bash
pip install -r requirements.txt
```

## Dependencies Included

### Core Scientific Computing
- `numpy>=1.21.0`: Numerical computing
- `pandas>=1.3.0`: Data manipulation
- `scipy>=1.7.0`: Scientific computing

### Machine Learning
- `scikit-learn>=1.0.0`: Traditional ML algorithms
- `lightgbm>=3.3.0`: Gradient boosting framework
- `torch>=1.12.0`: PyTorch for deep learning
- `optuna>=2.10.0`: Hyperparameter optimization
- `shap>=0.40.0`: Model interpretability

### Visualization
- `matplotlib>=3.5.0`: Basic plotting
- `seaborn>=0.11.0`: Statistical visualization
- `plotly>=5.0.0`: Interactive plots

### System & Utilities
- `psutil>=5.8.0`: System monitoring
- `streamlit>=1.10.0`: Web dashboard
- `joblib>=1.1.0`: Model serialization
- `tqdm>=4.62.0`: Progress bars

## Compatibility
- **Python**: 3.11 or higher
- **Operating Systems**: Windows, macOS, Linux
- All dependencies are compatible with Python 3.11+

## Troubleshooting

### Import Errors
If you still get import errors after installation:
1. Restart your Python session
2. Check that you're using the correct Python environment
3. Run: `python install_dependencies.py` again

### Version Conflicts
If you encounter version conflicts:
1. Create a new virtual environment
2. Install dependencies in the clean environment
3. Or update conflicting packages manually

### GPU Support
For GPU acceleration with PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verification
After installation, verify SCAF-LS works:
```python
import scaf_ls
print("SCAF-LS imported successfully!")
```