
# Cross-Section Image Generation, Simulation, and Machine Learning Pipeline

This repository provides a comprehensive workflow for generating cross-sectional images from CAD models, simulating mechanical properties using ANSYS Workbench, and training machine learning models to predict Young’s modulus from image data. The process involves data generation, simulation, image preprocessing, and multiple training methodologies employing CNN and Transformer architectures.

---

## Workflow and Instructions

### 1. Data Generation

The initial step involves generating cross-sectional images along with their corresponding STEP files. This is performed using the script **`cad_script.py`**, which allows customization of parameters such as the number of runs, number of pores, radius values, standard deviation, and more.

- **Recommended Software:** FreeCAD 0.20.2
- **Execution:** The script must be executed within FreeCAD’s Python console. This console can be accessed by navigating to **View > Panels > Python Console** within the FreeCAD interface.  
- **Note:** Parameters must be manually updated for each distinct combination, as the script was not fully automated due to time constraints; this script can be modified in the future to be fully automated.
- **Important:** Avoid using semicolons (`;`) in file or folder names, as they may interfere with downstream processing—particularly because all CSV files in this pipeline use `;` as the delimiter.

#### **Data Generation Protocol**

For each unique set of parameters, 25 independent simulations are executed to ensure a randomized distribution of pores. Each simulation run yields a comprehensive set of outputs, including:

- **22 cross-sectional images**: Eleven generated along the XY plane and eleven along the XZ plane. 
- **Orthogonal views**: Images of the front, right, and top perspectives.  
- **Isometric view**: A single image providing a comprehensive 3D perspective.  
- **Report**: A summary report documenting all relevant simulation parameters.  
- **STEP files**: CAD-compatible STEP files for each individual simulation.

> 📌 All generated images are **grayscale** and contain **a single channel**.

#### **File and Folder Naming Convention**

All files and folders follow a strict naming convention to ensure compatibility across different stages of the pipeline:

- **STEP files and images:**  
  Named as `radX.XXX_stdX.XX_poresXXX_runXX.step`, `radX.XXX_stdX.XX_poresXXX_runXX_(xy or xz)-depthX.XX.png`, where `XXX` are natural numbers (padded if required) and `X.XXX` are fractional values.

- **Simulation result files:**  
  Named as `radX.XXX_stdX.XX_poresXXX_runXX.txt`, saved under `base_analysis_files/user_files/`.

- **Folders:**  
  Use consistent naming for folders (`radX.XXX_stdX.XX_poresXXX_runXX`).

> ⚠️ Consistency in naming is essential for automation scripts to correctly locate and process the data.


---

### 2. Simulation in ANSYS Workbench

Once the STEP files are obtained, the next step is to simulate those models in ANSYS Workbench to obtain stress and strain data, from which Young’s modulus is calculated.  
- **Script:** The simulation automation script is provided in **`ansys_simulation_script.py`**.  
- **Recommended Software:** ANSYS Workbench 2024  
- **Execution:** To run the simulations, copy the whole script and paste it into the ACT console in ANSYS Workbench. The ACT console is accessible via the **Extensions** menu in Workbench.

#### **Simulation Usage Guide**

To run automated simulations correctly, please follow these steps:

##### **1. Directory Setup**

Ensure the following files and folders are present in your working directory:
```
simulation_project/
├── ansys_simulation_script.py
├── base_analysis.wbpj
├── base_analysis_files/
├── model_001.step
├── model_002.step
├── ...
```
- The **`Python script`**, **`base_analysis.wbpj`** project file, and **`.step`** files must all be in the **same root directory**.
- The folder **`base_analysis_files/`** should contain all linked resources and configuration files needed by the `.wbpj` project.

##### **2. Running the Script in ANSYS**

1. Open the provided project file: `base_analysis.wbpj` with the ANSYS Workbench 2024.
2. Navigate to **Extensions > View ACT Console**.
3. Copy the entire content of **`ansys_simulation_script.py`**.
4. Paste it into the **ACT console** and press **Enter** to begin simulations.

##### **3. Output File Structure**

After running the simulations, the output result files (stress–strain data in `.txt` format) are saved automatically within the `base_analysis_files/` directory under a subfolder named `user_files`. The file structure will look like this:

```
simulation_project/
├── base_analysis_files/
│ ├── user_files/
│ │ ├── model_001.txt
│ │ ├── model_002.txt
│ │ ├── ...
```

- Each `.txt` file contains stress–strain data for a corresponding STEP model.
- These files are later used by `calculate_ym.py` to compute Young’s modulus values.

---

### 3. Post-Simulation Processing

After simulation:

- **Calculate Young’s modulus:**  
  Use **`calculate_ym.py`** to compute and store Young’s modulus values from the simulation output.  
  - You must provide:  
    1. The **directory path** where all the result files (stress–strain data) are stored  
    2. The **output CSV file path and name** where the calculated Young's Modulus will be saved  
  - This script **both calculates and saves** Young’s modulus values automatically — no manual export is needed.

- **Crop cross-sectional images:**  
  Use **`cropping_images.py`** to remove the excess background from generated cross-sectional images.  
  - The presence of extra background is an artifact of how images are rendered during data generation in FreeCAD.
  - You must provide:  
    1. The **directory path** where all the images are stored.
    2. The **output directory path** where the directory with all the cropped images will be saved.
  - Cropping ensures only the relevant geometry is retained, improving the quality of input data for training.

- **Resize images for consistency:**  
  Use **`resizing_images.py`** to resize all images to a uniform size, as required by your model architecture.  
  - This script uses **OpenCV**, specifically the `cv2.resize()` function, to perform the resizing operation.
  - You must provide:  
    1. The **directory path** where all the cropped images are stored  
    2. The **output directory path** where the directory with all the resized images will be saved
  - Uniform image dimensions are critical for batching and training in deep learning models.

---

### 4. Preparing Data for Training

The project implements five distinct training methodologies, each differing in how image data and auxiliary parameters are processed and fed into the network.

#### **CNN Architecture**

- **Feeding images individually**  
  *Each image is treated as an independent training sample, regardless of whether it originates from the same or a different simulation run.*  
  ➤ **Training file:** `training_cnn_individual.py`

- **Stacking multiple images as a single multi-channel input**  
  *Combines 11 cross-sectional images (from either XY or XZ plane) generated in a single simulation run into one multi-channel image.*  
  ➤ **Training file:** `training_cnn_stacked.py`

- **Feeding images individually with depth as an additional feature vector**  
  *Extends the first method by including the depth parameter as an additional feature vector.*  
  ➤ **Training file:** `training_cnn_individual_with_depth.py`

- **Feeding images individually with depth, radius, and pore count as additional feature vectors**  
  *Further extends the third method by incorporating additional input parameters (radius and pore count).*  
  ➤ **Training file:** `training_cnn_individual_with_all_vectors.py`

#### **Transformer Architecture (ViT: vit-base-patch16-224-in21k)**

- **Feeding images individually with depth as an additional feature vector**  
  *Same concept of image individuality as above, enhanced with depth information.*  
  ➤ **Training file:** `training_transformer_individual_with_depth.py`


#### **Mapping Images to Ground Truth**

Each method requires a CSV file that maps input images to their corresponding Young’s modulus values. Use the appropriate script based on how your input is structured:

- **Use `mapping_file.py`** for:
  - Individually fed images (single-channel)
  - Individually fed images with depth
  - Individually fed images with depth, radius, and pore count
  - Individually fed images for Transformer input with depth

- **Use `mapping_file_stacked.py`** if:
  - You are stacking multiple cross-sectional images into a single multi-channel image for input

#### **Image Loading**

The image loading process varies depending on how the data is structured for training. Use the appropriate loader script based on your selected methodology:

- **Use `image_loader.py`** if:
  - You are feeding individual grayscale images (as standalone samples) into the network.

- **Use `image_loader_stacked.py`** if:
  - You are stacking 11 cross-sectional images (from the same simulation run and same plane) into a single multi-channel input image.

- **Use `image_loader_depth.py`** if:
  - You are feeding individual images along with additional features such as depth, or
  - You are using a Vision Transformer (ViT) and passing depth as an additional feature.

- **Use `image_loader_additional_features.py`** if:
    - You are including depth, radius, and pore count as additional feature vectors.


#### **Dependencies**

All development and experimentation in this project was conducted in a Python virtual environment that included the packages listed in the **`requirements.txt`** file.

While the complete list is provided for reproducibility, the core functionality of the scripts relies primarily on the following key packages:

- `numpy`  
- `os`
- `re`
- `glob`
- `csv`
- `cv2`
- `pandas`  
- `tqdm`
- `opencv-python` (for image processing)  
- `matplotlib` and `seaborn` (for plotting)  
- `scikit-learn` (for evaluation metrics and utility functions)  
- `torch` and `torchvision` (for training deep learning models)
- `cuda` compatible version with **torch version**

One may optionally create a virtual environment and install all packages used previously using:

```bash
python -m venv /path/to/new/virtual/environment
source myenv\Scripts\activate
pip install -r requirements.txt
```

This ensures compatibility and a streamlined setup process.

#### **Rescaling**

To improve model performance and training stability, the ground truth values (Young’s modulus) and any additional feature vectors (e.g., depth, radius, pore count) are standardized prior to training.

- **Standardization Process:**  
  This is handled using utility functions in **`helper_functions.py`**. The process involves:
  - First computing the **mean** and **standard deviation** of the relevant features.
  - Then applying standardization:  
    standardized_value = (x - mean) / std

- **Note on De-standardization:**  
  The pipeline does **not** automatically reverse the scaling after predictions.  
  If de-standardization is required, **users must manually save the computed mean and standard deviation** for each training session.  
  > ⚠️ Because the dataset is randomly split each time, the mean and standard deviation will vary every independent training session.  

  To recover the original scale:
  original_value = (standardized_value * std) + mean

---

### 5. Model Training and Evaluation

- Network architectures are defined in **`nn_models.py`**.  
- Training, validation, and testing results are saved for subsequent analysis and metric computation.  
- Performance metrics such as Mean Squared Error (MSE) and R² score are calculated with **`calculate_metrics.py`**. 
- Training progress can be visualized using **`train_plot.py`**, and testing results plotted with **`result_plot.py`**.   
- Different training methods correspond to different versions of the training scripts; users should review commit histories and messages for details.

---

### 6. Data Analysis and Validation

- To verify data completeness and perform exploratory analysis, use **`data_analysis.py`** and **`random_checks.py`**.  
- Helper functions utilized throughout the codebase are found in **`helper_functions.py`**.
- The helper functions used initially for naming convention which are no longer required are found in **`old_helper_functions.py`**.

---

## Recommendations

- Follow the outlined order for the pipeline to ensure smooth data flow and consistency.  
- Verify parameter changes carefully when generating new datasets, as this process is currently manual.  
- Use the recommended versions of FreeCAD and ANSYS Workbench to avoid compatibility issues.
- **Maintain the Suggested Folder Structure**

  It is strongly recommended to maintain the following folder structure when working with this project. Almost all Python scripts that require input/output paths use **relative paths**, which means that one do **not** need to manually update any file paths—simply preserving this structure allows all scripts to run seamlessly.

```
├── .venv/
├── pycache/
├── jupyter_notebooks/
├── saved_models/
├── testing_results/
├── training_results/
├── youngs_modulus_prediction/
├── .gitignore
├── all python files
```
---

## Troubleshoot

- Ensure that the versions of **CUDA** and **PyTorch** are compatible for GPU acceleration. Incompatible versions may result in failure to utilize the GPU or runtime errors.

- **ANSYS Simulation Time Increases Over Time**  
  It has been observed that as more STEP files are simulated consecutively in a single ANSYS Workbench session, the total simulation time per file tends to increase significantly. Although the exact cause is unknown, this behavior may be related to internal memory accumulation or resource handling by ANSYS.

  To mitigate this, it is advisable to divide the total STEP files into smaller batches (e.g., 200–300 files) and simulate each batch separately. After completing one batch, **close and reopen ANSYS Workbench** before proceeding with the next to maintain consistent simulation performance.


---

## Description of python files

- write which file is doing what
