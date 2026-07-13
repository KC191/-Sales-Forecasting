# Sales Forecasting Project - Superstore Dataset

Week 3 & 4 internship project submission.

## What's in here
- `analysis.ipynb` - main notebook, covers Tasks 1-6 (EDA, time series decomposition, the 3 forecasting models + comparison, category/region forecasts, anomaly detection, clustering). I tried to explain my reasoning in markdown cells as I went, including a few spots where I wasn't fully sure I did things the textbook-correct way (see the "Challenges" section at the end of the notebook).
- `train.csv` - the Superstore dataset from Kaggle.
- `app.py` - Streamlit dashboard for Task 7, 4 pages.
- `requirements.txt` - packages you need to run either the notebook or the app.
- `summary.docx` - 2-page business report for Task 8.
- `charts/` - all the chart PNGs exported from the notebook.

## Running the notebook
```
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

## Running the dashboard
```
pip install -r requirements.txt
streamlit run app.py
```
Just make sure `train.csv` is sitting in the same folder as `app.py`.

## Deploying on Streamlit Community Cloud
1. Push this whole folder to a GitHub repo (public, so Streamlit Cloud can see it).
2. Go to share.streamlit.io, sign in with GitHub, click "New app".
3. Point it at the repo and set the main file to `app.py`.
4. Deploy - first build takes a few minutes since it has to install Prophet and XGBoost.
5. Grab the live link for the submission form.

## Quick note on which model I used
I tested SARIMA, Prophet, and XGBoost in Task 3 by holding out the last 3 real months and comparing forecasts against them. XGBoost had the lowest error (MAPE), so that's what I used going forward for Task 4 and the dashboard. All 3 models and their numbers are in the notebook if you want to see the comparison.
