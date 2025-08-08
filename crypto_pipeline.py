#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 20:12:32 2025

@author: test
"""


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import requests
import pandas as pd
import numpy as np
import re
import time
import random
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# config

API_KEY = "ca28d0c8038e074b58ba188a33bdefad11bf7dbbfc739fe5942f8a3323ee075a"  
SYMBOL = "BTC"
WEEK_FOLDER = "week5_crypto"
LIMIT_DAYS = 2000
NEWS_LIMIT = 100


#  output folders

def build_week_dirs(week_folder: str = "week5_crypto") -> dict:
    root = Path.cwd().resolve()
    week_dir = root / week_folder
    results_dir = week_dir / "results" / "clean_data"
    fig_dir = week_dir / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"data_dir": results_dir, "fig_dir": fig_dir}

# price data

def get_daily_price_data(symbol: str, api_key: str, limit: int = 2000, currency: str = "USD") -> pd.DataFrame:
    url = f"https://data-api.coindesk.com/index/cc/v1/historical/days?market=cadli&instrument={symbol}-{currency}&limit={limit}&aggregate=1&fill=true&apply_mapping=true"
    headers = {"authorization": f"Apikey {api_key}"}
    response = requests.get(url, headers=headers, timeout=30)
    data = response.json()

    if "Data" not in data:
        print(f"No data for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(data["Data"])
    df["date"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
    df = df.rename(columns={
        "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close",
        "VOLUME": "btc_volume", "QUOTE_VOLUME": "usd_volume"
    })
    df["symbol"] = symbol
    return df[["symbol", "date", "open", "high", "low", "close", "usd_volume", "btc_volume"]]


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_7d"] = df["log_return"].rolling(7).std() * np.sqrt(365)
    df["return_7d"] = df["close"].pct_change(periods=7)
    return df


# news download

def get_crypto_news(api_key: str, limit: int = 100) -> pd.DataFrame:
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "Data" not in data:
            return pd.DataFrame()

        rows = []
        for article in data["Data"]:
            rows.append({
                "published_on": pd.to_datetime(article.get("published_on", 0), unit="s"),
                "title": article.get("title", ""),
                "body": article.get("body", ""),
                "source": article.get("source", ""),
                "categories": article.get("categories", ""),
                "url": article.get("url", "")
            })
        return pd.DataFrame(rows)

    except Exception as e:
        print("Error downloading news:", e)
        return pd.DataFrame()


# clean text

def clean_news_text(df: pd.DataFrame) -> pd.DataFrame:
    def clean(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)
        text = re.sub(r"\\s+", " ", text)
        return text.strip()

    df["clean_title"] = df["title"].apply(clean)
    df["clean_body"] = df["body"].apply(clean)
    return df


# sentiment analysis

def add_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["clean_title"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
    return df


# csv

def save_data(df: pd.DataFrame, data_dir: Path, filename: str):
    file_path = data_dir / filename
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} rows to {file_path}")


# main function

def main():
    data_dir = build_week_dirs(WEEK_FOLDER)


    df_price = get_daily_price_data(SYMBOL, API_KEY, limit=LIMIT_DAYS)
    if not df_price.empty:
        df_price = add_price_features(df_price)
        save_data(df_price, data_dir["data_dir"], f"{SYMBOL.lower()}_price_features.csv")
    else:
        print("No price data found.")

  
    df_news = get_crypto_news(API_KEY, limit=NEWS_LIMIT)
    if not df_news.empty:
        df_news = clean_news_text(df_news)
        df_news = add_sentiment_scores(df_news)
        save_data(df_news, data_dir["data_dir"], "crypto_news_sentiment.csv")
    else:
        print("No news data found.")

    # gemini
    summary_path = data_dir["data_dir"] / "gemini_summary.txt"
    latest_sentiment = df_news["sentiment_score"].mean() if not df_news.empty else "N/A"
    latest_return = df_price["return_7d"].iloc[-1] if not df_price.empty else "N/A"

    with open(summary_path, "w") as f:
        f.write("Sentiment-Enhanced Crypto Portfolio Summary\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Latest BTC 7-Day Return: {latest_return:.4f}\n")
        f.write(f"Average News Sentiment Score: {latest_sentiment:.4f}\n")
        f.write("Recommendation: ")
        if isinstance(latest_sentiment, float):
            if latest_sentiment > 0.05:
                f.write("Consider increasing allocation to BTC due to positive sentiment.\n")
            elif latest_sentiment < -0.05:
                f.write("Consider reducing BTC exposure due to negative sentiment.\n")
            else:
                f.write("Neutral sentiment detected – hold current position.\n")
        else:
            f.write("No sentiment data available.\n")

 
    gemini_prompt_path = data_dir["data_dir"] / "gemini_prompt.txt"
    with open(gemini_prompt_path, "w") as f:
        f.write("PROMPT FOR GEMINI:\n")
        f.write("I have run a crypto portfolio optimization model using sentiment analysis from news data.\n")
        f.write("Here is the summary output. Please help me understand what this means for my portfolio:\n\n")
        with open(summary_path, "r") as summary:
            f.write(summary.read())

    print("ETL + Feature Engineering Completed.")
    print("Final Gemini summary written to:", summary_path)

if __name__ == "__main__":
    main()
