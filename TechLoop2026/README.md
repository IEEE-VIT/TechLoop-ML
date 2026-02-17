# IEEE TechLoop+ 2026: Introduction to Machine Learning: A Hands-On Workshop
# PyTorch Basics → MNIST Digit Classifier → Photo Inference Pipeline

This repo is a mini learning + build session:
1) **PyTorch basics** (tensors, `nn.Module`, forward pass, loss, optimizer, train/eval)
2) **MNIST digit classifier** (training a simple MLP on MNIST)
3) **ML fundamentals (handwritten notes)**: what a model is, Linear Regression, Neural Networks etc.
4) **Inference pipeline** to test the trained model on **real handwritten digit photos** (jpg/jpeg/png).

---

## Repo structure

- `flow.ipynb` — Notebook with the full training flow (dataset → model → training loop → evaluation → TorchScript export)
- `predict_digit_ts_mlp.py` — Inference script (loads TorchScript model + preprocessing + prediction)
- `requirements.txt` — Python dependencies
- `Handwritten_Notes.pdf` — Notes from the session
- `Machine Learning.pptx` — Slides / material
- `my_digit.jpeg`, `my_digit1.jpeg`, `my_digit2.jpeg` — Sample test images

> **Important:** The inference script expects a TorchScript model file:
- `mnist_digit_model_ts.pt` (generated/exported from the notebook)

If you don’t see `mnist_digit_model_ts.pt` in the repo yet, export it from `flow.ipynb` (see below).

---

## Setup

### 1) Install Dependencies
pip install -r requirements.txt

### 2) Train and Export the model
Open flow.ipynb, run training, then export the TorchScript model.

### 3) Inference on handwritten digit photos
The inference script:

loads mnist_digit_model_ts.pt

preprocesses photos to look MNIST-like (grayscale, autocontrast, invert option, thresholding, cropping, centering)

outputs predicted digit + confidence

Predict a single image using (for dark ink on white paper):-
python predict_digit_ts_mlp.py --model mnist_digit_model_ts.pt --image my_digit.jpeg --invert --debug 

(for white ink on black paper) :-
python predict_digit_ts_mlp.py --model mnist_digit_model_ts.pt --image my_digit.jpeg --debug
