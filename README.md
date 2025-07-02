
# Cross-Section Image Generation, Simulation, and Machine Learning Pipeline

This repository provides a comprehensive workflow for generating cross-sectional images from CAD models, simulating mechanical properties using ANSYS Workbench, and training machine learning models to predict Young’s modulus from image data. The process involves data generation, simulation, image preprocessing, and multiple training methodologies employing CNN and Transformer architectures.

---

## Workflow and Instructions

### 1. Data Generation

The initial step involves generating cross-sectional images along with their corresponding STEP files. This is performed using the script **`CAD_script.txt`**, which allows customization of parameters such as the number of runs, number of pores, radius values, standard deviation, and more.  
- **Recommended Software:** FreeCAD 0.20  
- **Execution:** The script should be run within FreeCAD’s Python console, which can be accessed via the **View** menu. Under **View > Panels**, enable the **Python Console** to open it.  
- **Note:** Parameters must be manually updated for each distinct combination; this can be automated in future versions.

### 2. Simulation in ANSYS Workbench

Once the STEP files are generated, the next step is to simulate these models in ANSYS Workbench to obtain stress and strain data, from which Young’s modulus is calculated.  
- The simulation automation script is provided in **`ansys_simulation_automation.ipynb`**.  
- To run this script, copy its content into the ACT console in ANSYS Workbench (recommended version: 2024). The ACT console is accessible via the **Extensions** menu in Workbench.

### 3. Post-Simulation Processing

After simulation:  
- Calculate Young’s modulus using **`calculate_ym.py`** and save the results in a CSV file.  
- Cross-sectional images typically contain extra background, which should be removed using **`cropping_images.py`**.  
- For consistency and compatibility with neural networks, images can be resized using **`resizing_images.py`** to any desired dimension.

### 4. Preparing Data for Training

There are five distinct training methodologies implemented, differing in how the image data and auxiliary parameters are fed into the network:

1. Feeding images individually using a CNN architecture.  
2. Stacking multiple images as a single multi-channel input to a CNN.  
3. Feeding images individually with depth as a spatial dimension in CNN.  
4. Feeding images individually with depth, radius, and pore count as spatial dimensions in CNN.  
5. Feeding images individually with depth as a spatial dimension, using a pre-trained Vision Transformer (ViT) architecture.

#### Mapping Images to Ground Truth

- A mapping CSV file associating images with their corresponding Young’s modulus is necessary for all methods.  
- Use **`mapping_file.py`** for methods 1, 3, 4, and 5.  
- Use **`mapping_file_stacked.py`** for method 2 (stacked images).

#### Image Loading

- For method 1, use **`image_loader.py`**.  
- For method 2, use **`image_loader_stacked.py`**.  
- For methods 3, 4, and 5, use **`image_loader_depth.py`**. Note that this file varies across different GitHub commits depending on the method; users are advised to check relevant commit versions.

---

## Dependencies

All required Python packages and dependencies for running the scripts smoothly are listed in the **`requirements.txt`** file. It is recommended creating a virtual environment and installing the dependencies using:

```bash
pip install -r requirements.txt
```

This ensures compatibility and a streamlined setup process.

---

### 5. Model Training and Evaluation

- Network architectures are defined in **`nn_models.py`**.  
- Training loops for CNN architectures are in **`cnn.py`**, while Transformer training is handled in **`transformer.py`**.  
- Training, validation, and testing results can be saved for subsequent analysis and metric computation.  
- Training progress can be visualized using **`plot.py`**, and testing results plotted with **`test_plot.py`**.  
- Performance metrics such as Mean Squared Error (MSE) and R² score are calculated with **`calculate_r2.py`**.  
- For Artificial Neural Network (ANN) training, see **`ann.py`**.  
- Different training methods correspond to different versions of the training scripts; users should review commit histories and messages for details.

---

### 6. Data Analysis and Validation

- To verify data completeness and perform exploratory analysis, use **`data_analysis.py`** and **`random_checks.py`**.  
- Helper functions utilized throughout the codebase are found in **`helper_functions.py`**.

---

## Recommendations

- Follow the outlined order for the pipeline to ensure smooth data flow and consistency.  
- Verify parameter changes carefully when generating new datasets, as this process is currently manual.  
- Explore commit histories for specific file versions that best suit your chosen training methodology.  
- Use the recommended versions of FreeCAD and ANSYS Workbench to avoid compatibility issues.

---

