# Hotel Booking Cancellation Prediction

Binary classification model to predict hotel booking cancellations using the Hotel Booking Demand dataset (119,390 bookings, 2015–2017).

**Target:** `is_canceled` (0 = kept, 1 = cancelled)  
**Metrics:** ROC-AUC, F1-score, Precision/Recall, evaluated separately per hotel type

## Project Workflow

This project follows an agentic methodology: each stage was planned, delegated to an AI agent, verified, and revised before proceeding.

| Stage | Description | Agent Role |
|---|---|---|
| Section 1 : Problem Framing | Define target, metrics, constraints | Validated framing, identified leakage risk |
| Section 2 : EDA | Visual analysis of distributions, missingness, outliers | Generated initial code, iterated on feedback |
| Section 3 : Data Preparation | Preprocessing pipeline, time-based split | Suggested cleaning steps, verified and corrected |
| Section 4 : Modelling | Baseline, Random Forest, XGBoost, MLP | Generated model definitions, corrected imports and architecture |
| Section 5 : Evaluation | Per hotel type evaluation, threshold tuning | Generated evaluation code, corrected shape mismatches |

Detailed agent interactions and accept/reject decisions are documented in the Agent Usage Log (Appendix).

## Repository Structure
```
hotel-booking-cancellation/
├── final_notebook_predictives.ipynb  # Full pipeline, EDA through final evaluation
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Data

Publicly available on Kaggle: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

1. Download `hotel_bookings.csv`
2. Place it in the root of this repository, same folder as the notebook

Not included due to Kaggle's redistribution terms.

## How to Run

Requirements: Python 3.8+
```bash
git clone <repository-url>
cd hotel-booking-cancellation
pip install -r requirements.txt
jupyter notebook
```

Open `final_notebook_predictives.ipynb` and run all cells top to bottom. Section 3 must complete before Sections 4 and 5, as intermediate CSVs are generated there and read by subsequent sections.

## Key Decisions

| Decision | Choice |
|---|---|
| Split strategy | Time-based (train < 2017, val Jan–Apr 2017, test May–Aug 2017) |
| Leakage columns excluded | `reservation_status`, `reservation_status_date`, `assigned_room_type` |
| Duplicates | Kept, 26.8% apparent duplicates due to anonymisation, not true duplicates |
| Scaling | StandardScaler on MLP and Logistic Regression inputs only, tree models are scale-invariant |
