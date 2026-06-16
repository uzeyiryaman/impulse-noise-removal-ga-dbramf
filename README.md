# Impulse Noise Removal with Edge Preservation

This repository contains the implementation files for the term project:

Impulse Noise Removal with Edge Preservation

## Methods

- Standard Median Filter
- Standard DBRAMF
- Proposed GA-DBRAMF

## Noise Models

- Salt-and-Pepper Noise
- Random-Valued Impulse Noise
- Structured Burst Noise

## Evaluation Metrics

- MSE
- PSNR
- SSIM
- Pratt's Figure of Merit

## How to Run

Install the required libraries:

pip install -r requirements.txt

Run the project:

python ga_dbramf_project.py

## Round-2 Update

The Round-2 version extends the original experiment according to the instructor feedback.

Main Round-2 updates:

- The evaluation dataset was expanded from 2 benchmark images to 6 benchmark images:
  Lena, Cameraman, Coins, Moon, Page, and Text.
- A stronger baseline method, Switching Adaptive Median Filter (SAMF), was added.
- The RVIN handling was improved using an adaptive local median deviation-based detector.
- RVIN detector performance was evaluated using precision, recall, F1-score, and accuracy.
- Results were reported using mean ± standard deviation over the expanded experimental matrix.
- Runtime and complexity analysis were added.

## Repository Files

- `ga_dbramf_project.py`: original implementation file.
- `ga_dbramf_round2_project.ipynb`: complete Round-2 Google Colab notebook.
- `ga_dbramf_round2_project.py`: Round-2 Python code version.
- `requirements.txt`: required Python libraries.
- `README.md`: project documentation.

## Round-2 Experimental Matrix

The Round-2 experimental matrix includes:

- 6 benchmark images
- 3 impulse noise types
- 5 noise density levels
- 4 restoration methods

This gives:

```text
6 images × 3 noise types × 5 densities × 4 methods = 360 restored-method results

## Author

Uzeyir Yaman  
Student ID: 2111011074  
Department of Electrical and Electronics Engineering  
Abdullah Gul University
