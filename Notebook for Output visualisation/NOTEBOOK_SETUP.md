# Jupyter Notebook Virtual Environment Setup

This guide will help you set up a dedicated virtual environment for the wall normal analysis notebook.

## Quick Setup

### 1. Create Virtual Environment

```bash
cd '/Users/ishan-aiworkspace/Documents/apps/rgb-x/Notebook\ for\ Output\ visualisation'
python3 -m venv venv_notebook
```

### 2. Activate Virtual Environment

```bash
source venv_notebook/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements_notebook.txt
```

### 4. Register Jupyter Kernel

This makes the venv available as a kernel in Jupyter:

```bash
python -m ipykernel install --user --name=rgb-x-analysis --display-name "Python (RGB-X Analysis)"
```

### 5. Launch Jupyter

You can now launch Jupyter from anywhere:

```bash
jupyter notebook
```

Or launch it from this directory:

```bash
jupyter notebook wall_normal_analysis.ipynb
```

### 6. Select the Kernel

When the notebook opens:
1. Click on "Kernel" in the menu bar
2. Select "Change kernel"
3. Choose "Python (RGB-X Analysis)"

Now all packages will be installed in the isolated venv!

## Adding More Dependencies Later

If you need to add more packages (like transformers for SegFormer):

```bash
# Activate the venv
source venv_notebook/bin/activate

# Install new packages
pip install transformers torch torchvision

# Update requirements file (optional)
pip freeze > requirements_notebook.txt
```

The kernel will automatically use the updated packages.

## Deactivate Virtual Environment

When you're done:

```bash
deactivate
```

## Remove Kernel (if needed)

To remove the Jupyter kernel:

```bash
jupyter kernelspec uninstall rgb-x-analysis
```

## VSCode Integration

If you're using VSCode with Jupyter extension:
1. Open the notebook in VSCode
2. Click on the kernel selector in the top-right
3. Choose "Python (RGB-X Analysis)"
4. VSCode will automatically use your venv

## Troubleshooting

**Kernel not showing up?**
- Make sure you activated the venv before installing ipykernel
- Try: `jupyter kernelspec list` to see registered kernels

**Import errors in notebook?**
- Verify the correct kernel is selected
- Check installed packages: `pip list` (with venv activated)
