# LipNet – 3D CNN-LSTM Lip Reading Model 👄
### Silent Video to Text using Deep Learning

A TensorFlow implementation of lip reading trained on the GRID corpus. The model watches silent video of a person's lips and transcribes what they're saying — no audio required. Includes a full Streamlit web app for live demo.

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | GRID Corpus (Speaker S1, ~500 videos) |
| Architecture | 3D CNN + Bidirectional LSTM (8.4M parameters) |
| Training | 100 epochs, CTC loss, Adam optimizer |
| Final training loss | 5.46 |
| Final validation loss | 2.94 |
| **Word Error Rate (WER)** | **11.61%** |

**Sample predictions after 100 epochs:**
```
Real:      place green by y five soon
Predicted: place green by y five soon  ✅

Real:      lay blue at y zero please
Predicted: lay blue at y zero please   ✅

Real:      bin green with b five soon
Predicted: bin green with b five soon  ✅

Real:      bin red by m six now
Predicted: bin red by v six now        (1 word off)

Real:      set green in o seven soon
Predicted: set green in s seven soon   (1 word off)
```

---

## Architecture

```
Input: 75-frame grayscale video (75 × 46 × 140 × 1)
    │
    ▼
Conv3D(128, 3×3×3) → ReLU → MaxPool3D(1×2×2)
Conv3D(256, 3×3×3) → ReLU → MaxPool3D(1×2×2)
Conv3D(75,  3×3×3) → ReLU → MaxPool3D(1×2×2)
    │
    ▼
TimeDistributed(Flatten)
    │
    ▼
Bidirectional LSTM(128) → Dropout(0.5)
Bidirectional LSTM(128) → Dropout(0.5)
    │
    ▼
Dense(vocab_size + 1, softmax)
    │
    ▼
CTC Decode → Text output
```

---

## Streamlit App

A full web interface for running predictions on any GRID video.

```bash
cd app
streamlit run streamlitapp.py
```

The app lets you select any video from the dataset and shows:
- The original video (converted to mp4)
- The cropped lip region the model sees (as a GIF)
- The raw token output from the model
- The final decoded text prediction

---

## Setup

### Prerequisites
- Python 3.10+
- TensorFlow 2.x
- Google Colab (recommended for training — requires GPU)

### Dataset
Download the GRID corpus (Speaker S1) from the official source:
- Videos: `http://spandh.dcs.shef.ac.uk/gridcorpus/`
- Place videos in `data/s1/` and alignment files in `data/alignments/s1/`

### Install dependencies
```bash
pip install tensorflow opencv-python matplotlib imageio streamlit gdown numpy
```

### Run the notebook
Open `LipNet.ipynb` in Google Colab with T4 GPU runtime enabled.

To use pre-trained weights, download `checkpoint.weights.h5` and load:
```python
model.load_weights('models/checkpoint.weights.h5')
```

---

## Project Structure

```
LipReading_AI/
├── streamlitapp.py      # Streamlit web interface
├── modelutil.py         # Model architecture + weight loading
├── utils.py             # Video preprocessing + alignment loading
├── LipNet.ipynb             # Training notebook
├── README.md
└── .gitignore
```

---

## Data Pipeline

Each video is processed as follows:
- Frames extracted via OpenCV, converted to grayscale
- Lip region cropped: `frame[190:236, 80:220]` → resized to `46×140`
- Padded or truncated to exactly 75 frames
- Normalized: `(frames - mean) / std`

Alignment files are parsed to extract spoken words, filtered for silence tokens (`sil`), and converted to character-level integer sequences for CTC training.

---

## Training Details

- **Loss:** CTC (Connectionist Temporal Classification) — handles variable-length sequence alignment without requiring frame-level labels
- **Optimizer:** Adam, initial lr=0.0001, exponential decay after epoch 30
- **Batch size:** 2 (constrained by GPU memory)
- **Train/test split:** 450 / 50 batches

---

## Limitations

- Trained on a single speaker (S1) — will not generalize to unseen speakers without retraining
- GRID corpus uses a constrained vocabulary (~50 words) — not suitable for open-vocabulary lip reading
- To improve generalization: add more GRID speakers (`s2`, `s3`, ...) or fine-tune on LRW dataset

---

## Tech Stack

- **TensorFlow / Keras** — model training and CTC loss
- **OpenCV** — video frame extraction and preprocessing
- **NumPy** — array manipulation and WER calculation
- **Streamlit** — web app interface
- **Google Colab** — T4 GPU training environment

---

## License

MIT
