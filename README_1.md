# Ilé: Nigerian Property Price Estimator

A machine learning project that estimates the listing price of a Nigerian property (in naira) from details such as location, property type, rooms and parking. The trained XGBoost model is served through a FastAPI endpoint and used by a small animated web page.

> **Live demo:** _add your GitHub Pages link here_
> **API docs:** _add your Render link here, ending in `/docs`_
>
> The API runs on free hosting, so the first request after a quiet period can take about a minute while the server wakes up.

![Screenshot of the estimator](docs/screenshot.png)
<!-- Save a screenshot of the result page as docs/screenshot.png -->

## What it does

1. You choose a state, district and property type, then set bedrooms, bathrooms, toilets and parking.
2. The page sends those details to the API.
3. The API runs the saved XGBoost model and returns a predicted price.
4. The page shows the price with an animated count-up.

## How it was built

| Stage | What I did |
|---|---|
| Data | 1,668 Nigerian property listings, reduced to 1,611 after removing duplicate property references |
| Cleaning | Parsed dates and area text, handled missing values, dropped columns that were almost empty (for example service charge, about 99.5% missing) |
| Leakage control | Left out fields that would not be known when a property is first listed (such as days on the market) |
| Features | 12 inputs: market status, type, bedrooms, bathrooms, toilets, parking spaces, district, state, servicing, furnishing, year and month added |
| Modelling | Linear regression baseline, then XGBoost, with a log-transformed price target and tuning of regularisation and tree settings |
| Evaluation | Cross-validation, price-range error analysis, feature importance and feature-ablation tests, then a final held-out test set |
| Deployment | Model saved with joblib, served by FastAPI, containerised with Docker, hosted on Render, with a static front end on GitHub Pages |

## Results

Final held-out test set (242 listings the model never saw during development):

| | MAE | RMSE | R² |
|---|---:|---:|---:|
| **XGBoost** | ₦249.4M | ₦1.04B | 0.30 |
| Predict-the-average baseline | ₦638.5M | ₦1.22B | -0.03 |

The model beats the baseline on every metric, but it is a first version and has clear limits.

## Limitations (please read)

- **It is not a valuation tool.** Do not use it to buy, sell or price a property.
- **Accuracy is uneven.** It is much closer on ordinary listings (roughly ₦500M and below) than on rare luxury listings, where errors can run into billions.
- **A few extreme prices distort the results.** The data contains listings up to ₦170B and some suspicious records (for example 50-bedroom houses). They make the error metrics unstable: cross-validation results swung a lot depending on which extreme listings landed in each fold.
- **Small dataset.** About 1,600 listings across many districts, so some districts have very few examples.
- **Area size was mostly missing** (about 87% to 92%), so it could not be relied on.

## Run it on your computer

```bash
git clone https://github.com/Marvee-dagreat/nigeria-property-price-estimator.git
cd nigeria-property-price-estimator
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000/docs` to see the API and try a prediction, or open `docs/index.html` in your browser for the web page.

## Project structure

```
app.py                      FastAPI app that loads the model and serves predictions
property_price_model.pkl    Saved trained model
requirements.txt            Python packages (pinned versions)
Dockerfile                  Container recipe used for deployment
docs/index.html             The animated front end (served by GitHub Pages)
```

## Roadmap

- Investigate and document rules for the suspicious listings, then retrain
- Collect more listings and fix the missing area data
- Report prediction ranges instead of a single number
- Compare other models (LightGBM, CatBoost) under the same test protocol
- Track experiments and add automated tests

## Author

Marvelous · [GitHub: Marvee-dagreat](https://github.com/Marvee-dagreat)
