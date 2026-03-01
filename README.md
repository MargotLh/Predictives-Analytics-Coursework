# Hotel Booking Cancellation Prediction

Binary classification model to predict hotel booking cancellations using the Hotel Booking Demand dataset (119,390 bookings, 2015-2017).

**Target variable:** `is_canceled` (0 = kept, 1 = cancelled)
**Evaluation metrics:** ROC-AUC, F1-score, Precision/Recall (evaluated separately per hotel type)

---

## Project Workflow

This project follows an agentic methodology — each stage was planned, 
delegated to an AI agent, verified, and revised before proceeding.

| Stage | Description | Agent Role |
|---|---|---|
| Section 1 — Problem Framing | Define target, metrics, constraints | Validated framing, identified leakage risk |
| Section 2 — EDA | Visual analysis of distributions, missingness, outliers | Generated initial code, iterated on feedback |
| Section 3 — Data Preparation | Preprocessing pipeline, time-based split | Suggested cleaning steps, verified and corrected |
| Section 4 — Modelling | Baseline → Random Forest → XGBoost → MLP | TBD |
| Section 5 — Evaluation | Per hotel type evaluation, threshold tuning | TBD |

Detailed agent interactions and accept/reject decisions are documented 
in the Agent Usage Log (Appendix).

---

## Repository Structure

```
hotel-booking-cancellation/
├── EDA_section2.ipynb            # Section 2 — Exploratory Data Analysis
├── preprocessing_section3.ipynb  # Section 3 — Data Preparation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Data

The dataset is publicly available on Kaggle:
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

1. Download `hotel_bookings.csv` from the link above
2. Place it in the root of this repository (same folder as the notebooks)

The dataset is not included in this repository due to Kaggle's redistribution terms.

---

## How to Run

**Requirements:** Python 3.8+

1. Clone this repository
```bash
git clone <repository-url>
cd hotel-booking-cancellation
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the dataset (see Data section above)

4. Launch Jupyter
```bash
jupyter notebook
```

5. Run the notebook in order:
  

---

## Key Decisions

- **Split strategy:** Time-based (train < 2017 / val Jan-Apr 2017 / test May-Aug 2017) to simulate real deployment conditions
- **Leakage columns excluded:** `reservation_status`, `reservation_status_date`, `assigned_room_type`
- **Duplicates kept:** 26.8% of rows appear duplicated due to anonymisation, not true duplicates
- **Scaling:** StandardScaler applied to MLP inputs only -- tree-based models are scale-invariant
