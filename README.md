# Pavement Cell Segmentation

This repository contains the implementation of the research paper titled **"A Simple Approach to Pavement Cell Segmentation"** by Rostislav Shepel, Andrew Romanowski, and Mario Valerio Giuffrida.

## Abstract

This study focuses on segmenting pavement cells from microscopy images of Arabidopsis thaliana plants, which is critical for linking cellular traits to overall plant performance. Differently than the current state-of-the-art, we propose a simple, easy-to-train approach using partially annotated datasets to address the challenges of irregular pavement cell shapes. Specifically, we employed U-Net and DeepLabV3 architectures for segmentation, showing that both models can perform well despite the constraints. Post-segmentation, we used PaCeQuant to extract phenotyping data, demonstrating the effectiveness of our method. The results indicate that U-Net provides a slightly closer match to the true mask, though DeepLabV3 also performs robustly. This approach facilitates more accurate and efficient plant phenotyping, contributing to sustainable agricultural practices.

## Repository Contents

- `Unet_pavement__cell.ipynb`: Jupyter notebook implementing the original U-Net and U-Net with double decoder architectures for pavement cell segmentation.
- `DeepLabV3_pavement__cell.ipynb`: Jupyter notebook implementing the DeepLabV3 architecture for pavement cell segmentation.

## Running the Notebooks

To run these notebooks, follow the steps below:

### 1. Google Colab

Both notebooks are compatible with Google Colab. To run the notebooks:

1. Upload the notebook to your Google Drive.
2. Open the notebook in Google Colab.
3. Install necessary dependencies if not already installed (these are typically managed within the notebooks).
4. Run the cells sequentially.

### 2. Local Environment

If you prefer to run the notebooks locally:

1. Clone this repository:
    ```bash
    git clone https://github.com/Rosti35/pavement-cell-segmentation
    ```
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter notebooks using Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter notebook Unet_pavement__cell.ipynb
    jupyter notebook DeepLabV3_pavement__cell.ipynb
    ```

## Dependencies

The notebooks rely on the following Python libraries:

- torch==2.0.1
- torchvision==0.15.2
- numpy==1.24.0
- Pillow==9.5.0
- matplotlib==3.7.0
- scikit-image==0.20.0
- tqdm==4.66.1.

Make sure to install these dependencies before running the notebooks. A `requirements.txt` file is also provided to facilitate the installation of dependencies.

## Citation

If you use this code or find it useful, please cite the original paper.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or issues, please contact:

- Rostislav Shepel: [efyrs5@nottingham.ac.uk](mailto:efyrs5@nottingham.ac.uk)
- Andrew Romanowski: [andres.romanowski@wur.nl](mailto:andres.romanowski@wur.nl)
- Mario Valerio Giuffrida: [valerio.giuffrida@nottingham.ac.uk](mailto:valerio.giuffrida@nottingham.ac.uk)
