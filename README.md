
# Vektra Alpha Mobile

A mobile-friendly Streamlit application for the Vektra Global News Alpha Engine.

## What it does

- Runs in a web browser on iPhone or Android.
- Pulls recent company news from Finnhub.
- Pulls market prices and volume through yfinance.
- Scores stocks using the Vektra news-alpha engine.
- Displays ranked signals, probability, expected return and explanations.
- Can refresh automatically.

## Important limitation

An iPhone cannot reliably run a Python process continuously in the background.
For continuous monitoring, deploy this app to a cloud host and open its web address
on your phone. You can then add the page to your iPhone Home Screen.

## Fastest deployment route: Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files from this folder.
3. Open Streamlit Community Cloud and create a new app.
4. Select `app.py` as the entry point.
5. Add the following secret:

```toml
FINNHUB_API_KEY="your-api-key"
```

6. Deploy the app.
7. Open the app URL in Safari on your iPhone.
8. Tap Share → Add to Home Screen.

## Run on a computer first

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the network URL shown by Streamlit on your phone while both devices are
connected to the same Wi-Fi network.

## Data

Finnhub is used only as an example live-news source. A production system should use
licensed point-in-time market and news data and should not depend on yfinance.

## Security

Never paste API keys directly into code committed to GitHub. Use Streamlit secrets
or environment variables.
