# chessCV: FEN String Generation from Chessboard Images

## Overview  
This project focuses on generating **Forsyth–Edwards Notation (FEN)** strings directly from images of real-world chessboards captured at angles. Using the [ChessReD](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f/2) dataset, a complete **image processing pipeline** and a custom **Convolutional Neural Network (CNN)** were developed to achieve state-of-the-art accuracy.  

Previous state-of-the-art methods reached only ~15% full-board FEN accuracy. The final model achieves **63.96% full-board accuracy**, representing a **4× improvement** over prior work.

## Background & Previous Work  
ChessReD was introduced with the paper [*“End-to-End Chess Recognition”* (Masouris & van Gemert, 2023)](https://arxiv.org/pdf/2310.04086). Their approach trained a ResNeXt-101 CNN to directly predict FEN strings from full input images, without using any geometric preprocessing such as board corner detection or perspective warping. While novel, the method achieved only **15.26% full-board FEN reconstruction accuracy** on ChessReD test images.  

This project improves on that baseline by incorporating an **image processing pipeline** (edge detection, contour detection, and perspective warping) before classification. By transforming angled photos into normalized top-down board views, the model can more reliably identify pieces and board layouts. This preprocessing step, combined with a custom residual CNN, enabled reaching **63.96% accuracy**, a significant step forward compared to the previous state of the art.  

### Example Images from ChessReD  
The ChessReD dataset consists of smartphone photos of real chessboards under varied lighting and camera angles, along with annotations for where each piece is location on the board, allowing for ground truth FEN labels to be generated easily. Below are sample images:  

<div align="center">
  <img src="https://github.com/VedPatel10/chessCV/blob/main/images/chessred1.jpg?raw=true" alt="ChessReD Example 1" width="30%"/>
  <img src="https://github.com/VedPatel10/chessCV/blob/main/images/chessred2.JPG?raw=true" alt="ChessReD Example 2" width="30%"/>
</div>

---

## Repository Structure
- **`chess.ipynb`** – The complete original notebook containing the full pipeline: preprocessing, model training, and evaluation. Click [here](https://colab.research.google.com/drive/1X_6f3QCK5oZ73d2Pce3HCc1nok8EPjDH?usp=sharing) to view this file (too large for GitHub)  
- **`preprocessing.ipynb`** – Focused notebook containing only preprocessing steps, including image warping and board standardization.  
- **`train.ipynb`** – Focused notebook for model creation, training, testing, and results visualization.  

---

## Background  
The task of mapping images of chessboards to valid FEN strings involves:  
1. Detecting the board from an angled camera perspective.  
2. Warping the detected board into a **bird’s-eye view**.  
3. Classifying each of the 64 squares into one of 13 classes (6 white pieces, 6 black pieces, or empty).  
4. Reconstructing the FEN string.  

Traditional approaches often fail to generalize under varied lighting, camera angles, or board styles. By combining a tailored preprocessing pipeline with a CNN designed for spatial reasoning, this project demonstrates significant improvements in robustness and accuracy.

<p align="center">
  <img src="https://github.com/VedPatel10/chessCV/blob/main/images/Illustration-Model%20Overview.png?raw=true" alt="Project Illustration" width="600"/>
</p>

---

## Image Processing Pipeline  
The preprocessing pipeline consists of the following steps:  
- **Edge Detection (Canny + Gaussian blur)** to highlight board boundaries.  
- **Hough Line Transform** to detect major grid lines.  
- **Contour Detection** to isolate the chessboard region.  
- **Corner Detection & Ordering** to identify the four board corners.  
- **Perspective Warping** to produce a normalized **256×256 top-down chessboard image**.  

<p align="center">
  <img src="https://github.com/VedPatel10/chessCV/blob/main/images/processing%20(1).png?raw=true" alt="Processing Pipeline" width="650"/>
</p>

---

## Baseline Model  
A classical machine learning baseline was implemented using **Support Vector Machines (SVMs)** trained on **Histogram of Oriented Gradients (HOG)** features extracted from square-level images. The SVM achieved ~68% tile-level accuracy, performing well on distinctive pieces like queens and bishops, but struggling on low-contrast classes such as empty squares and kings. This baseline highlighted the difficulty of the problem and demonstrated the necessity of a deep learning approach.  

<p align="center">
  <img src="https://github.com/VedPatel10/chessCV/blob/main/images/baseline_confusion.png?raw=true" alt="Baseline Confusion Matrix" width="500"/>
</p>

---

## Model Architecture  
The final model (3rd attempt with our preprocessing pipeline) is a **Residual CNN** with:  
- Initial 7×7 convolution + batch normalization + ReLU + max pooling  
- 3 × Pre-Activation Residual Blocks for stable training  
- Adaptive Average Pooling mapped to an **8×8 chessboard grid**  
- Classifier head outputting shape `(B, 64, 13)`  

<p align="center">
  <img src="https://github.com/VedPatel10/chessCV/blob/main/images/model_architecture.png?raw=true" alt="Model Architecture" width="500"/>
</p>

---

## Results  
- **Full-board FEN accuracy:** **63.96%** (1145/1790 boards with full FEN construction, test dataset) 
- **Per-square accuracy:** **98.93%**, significantly higher across most piece classes, with especially strong performance on common pieces like pawns and rooks.  
- **Comparison to state-of-the-art:** Prior best ≈ **15%** full-board FEN accuracy (ChessReD baseline). This approach improves that by over **4×**.  

<!-- Side-by-side: Final Board Error Count + Confusion Matrix -->
<div align="center">
  <img src="https://github.com/VedPatel10/chessCV/blob/main/images/Final%20Model%20Errors%20Bar%20Graph.png?raw=true" alt="Final Board Error Count" width="48%"/>
  <img src="https://github.com/VedPatel10/chessCV/blob/main/images/test_confusion.png?raw=true" alt="Test Confusion Matrix" width="48%"/>
</div>

<!-- Train/Val curves below, matching the combined width -->
<p align="center">
  <img src="https://github.com/VedPatel10/chessCV/blob/main/images/Train-Val%20Accuracy%20and%20Loss%20Graphs.png?raw=true" alt="Train/Val Curves" width="96%"/>
</p>


---

## Future Work  
Several directions exist to extend this project:  
- **Stockfish Integration:** Predicted FEN strings can be passed to Stockfish via its Python API to provide users with the best next move in real time.  
- **Mobile App Deployment:** Embedding the pipeline in a mobile app would allow players to take a quick photo of the board and instantly receive analysis.  
- **Improved Preprocessing:** More reliable corner detection could reduce errors in warped images, especially under glare or partial occlusion.  
- **Alternative Architectures:** Exploring transformer-based vision models or attention mechanisms could further improve per-square classification accuracy.  

---

## Acknowledgements  
This project was made as a part of APS360 (Applied Fundamentals of Deep Learning), a University of Toronto undergraduate course.

## Contributions
This project would not have been possible without the help of [Gawtham Senthilvelan](https://github.com/Gawthaman), [Luthira Abeykoon](https://github.com/Luthiraa), and [Darshan Kasundra](https://github.com/Darshan-Kasundra).

---
