#pip install nltk textblob
#pip install nltk textblob

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
#from newsapi import NewsApiClient
from newsapi.newsapi_client import NewsApiClient
from datetime import datetime, timedelta

# First we will import the necessary Library
import datetime
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import os
import math

nltk.download('vader_lexicon')
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from itertools import product

import os
import pandas as pd
import numpy as np
import math
import datetime as dt

# For Evalution we will use these library

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# For model building we will use these library

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM


# For PLotting we will use these library

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots







st.set_page_config(initial_sidebar_state="expanded")


st.title('Bitcoin Price Analysis & Prediciton')
st.sidebar.info('Welcome to the Bitcoin Price Prediction App. Choose your options below')

def main():
    option = st.sidebar.selectbox('Make a choice', ['Recent data','Latest prices','Predict'])
    if option == 'Recent data':
            recent_data()
    elif option == 'Latest prices':
            dataframe()
    else:
            predict()



@st.cache_resource
def download_btc_data(ndays,time_step):
    END_DATE = datetime.now().strftime('%Y-%m-%d')  # Aujourd'hui
    START_DATE = (datetime.now() - timedelta(days=ndays)).strftime('%Y-%m-%d')
    if time_step == 1440:
        stock_data = yf.download('BTC-USD', start=START_DATE, end=END_DATE, interval='1d')
    else:
        stock_data = yf.download('BTC-USD', start=START_DATE, end=END_DATE, interval=f'{int(time_step)}m')

    stock_data['date'] = stock_data.index
    return stock_data

@st.cache_resource
def download_news_data(ndays,time_step,API_KEY):
    # Configuration des dates dynamiques
    newsapi = NewsApiClient(api_key=API_KEY)
    

    articles = pd.DataFrame()
    for i in range(ndays):
        all_articles = newsapi.get_everything(q='btc bitcoin crypto cryptocurrency',
                                            from_param=(datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d'),
                                            to=(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                                            language='en',
                                            sort_by='publishedAt')
        articles1 = pd.DataFrame(all_articles['articles'])
        articles = pd.concat([articles, articles1], ignore_index=True)

    news_df = articles
    print("La collecte et le stockage des données sont terminés.")


    # Prepare news_df to sentiment analysis
    news_df = news_df.rename(columns={'publishedAt': 'date', 'title': 'Title', 'content': 'Text'})
    news_df['date'] = pd.to_datetime(news_df['date'])


    news_df['year'] = news_df['date'].dt.year
    news_df['day'] = news_df['date'].dt.day

    # Group columns by 'date' with a X-minute interval(Articles published each X minutes ==> BTC prediction each X minutes)
    news_df['date'] = news_df['date'].dt.floor(f'{time_step}T')

    # Format 'date' to 'year-month-day-hour-minute'
    news_df['date'] = news_df['date'].dt.strftime('%Y-%m-%d %H:%M')
    
    return news_df

@st.cache_resource
def sentiment_analysis(text):

    # NLTK
    def apply_analysis(tweet):
        return SentimentIntensityAnalyzer().polarity_scores(tweet)

    # Article Text
    text[['neg','neu','pos','compound']] = text['Text'].apply(apply_analysis).apply(pd.Series)

    def sentimental_analysis(df):
        if df['neg'] > df['pos']:
            return 'Negative'
        elif df['pos'] > df['neg']:
            return 'Positive'
        elif df['pos'] == df['neg']:
            return 'Neutral'


    ## Calculate mean of 'neg', 'pos'.. of all articles in time_step (X minutes)
    # Calculate the mean 'neg.' grouped by date
    mean_neg = text.groupby('date')['neg'].mean()
    text = pd.merge(text, mean_neg, on='date', how='left', suffixes=('', '_mean'))
    text['neg'] = text['neg_mean']
    text.drop(columns=['neg_mean'], inplace=True)


    # Calculate the mean 'pos' grouped by date
    mean_pos = text.groupby('date')['pos'].mean()
    text = pd.merge(text, mean_pos, on='date', how='left', suffixes=('', '_mean'))
    text['pos'] = text['pos_mean']
    text.drop(columns=['pos_mean'], inplace=True)

    # Calculate the mean 'compound' grouped by date
    mean_compound = text.groupby('date')['compound'].mean()
    text = pd.merge(text, mean_compound, on='date', how='left', suffixes=('', '_mean'))
    text['compound'] = text['compound_mean']
    text.drop(columns=['compound_mean'], inplace=True)

    # Calculate the mean 'neu' grouped by date
    mean_neu = text.groupby('date')['neu'].mean()
    text = pd.merge(text, mean_neu, on='date', how='left', suffixes=('', '_mean'))
    text['neu'] = text['neu_mean']
    text.drop(columns=['neu_mean'], inplace=True)


    text['Sentiment_NLTK'] = text.apply(sentimental_analysis, axis = 1)

    text.head()

    def getSubjectivity(twt):
        return TextBlob(twt).sentiment.subjectivity
    def getPolarity(twt):
        return TextBlob(twt).sentiment.polarity
    def getSentiment(score):
        if score<0:
            return 'Negative'
        elif score==0:
            return 'Neutral'
        else:
            return 'Positive'

    text['Subjectivity']=text['Text'].apply(getSubjectivity)
    text['Polarity']=text['Text'].apply(getPolarity)


    # Calculate the mean 'Polarity' of all articles in time_step (X minutes)
    mean_polarity = text.groupby('date')['Polarity'].mean()
    text = pd.merge(text, mean_polarity, on='date', how='left', suffixes=('', '_mean'))
    text['Polarity'] = text['Polarity_mean']
    text.drop(columns=['Polarity_mean'], inplace=True)



    text['Sentiment_TB']=text['Polarity'].apply(getSentiment)
    text.head()

    text_grouped = text.groupby('date')[['neg', 'neu', 'pos', 'compound', 'Subjectivity', 'Polarity']].mean()
    text = text[['date',
        'Sentiment_NLTK','Sentiment_TB']]
    text = pd.merge(text, text_grouped, on='date', how='inner')
    text.drop_duplicates(subset='date', inplace=True)

    # Fill missing time steps with neutral sentiment:

    # Specify the time step
    step = timedelta(minutes=time_step)


    # Get the current date and time
    current_datetime = datetime.now()
    # Replace the time with midnight
    midnight_datetime = current_datetime.replace(hour=0, minute=0, second=0, microsecond=0)

    # Define the date range
    start_date = (midnight_datetime - timedelta(days=ndays)).strftime('%Y-%m-%d %H:%M')
    end_date = midnight_datetime.strftime('%Y-%m-%d %H:%M')
    date_range = pd.date_range(start=start_date, end=end_date, freq=step)

    # Create a DataFrame with the complete date range
    complete_dates_df = pd.DataFrame(date_range, columns=['date'])
    text['date'] = pd.to_datetime(text['date'], format='%Y-%m-%d %H:%M')

    # Merge your existing DataFrame with the complete date range DataFrame
    merged_df = pd.merge(complete_dates_df, text, on='date', how='left')

    # Fill missing values with defaults
    merged_df['neg'].fillna(merged_df['neg'].mean(), inplace=True)
    merged_df['neu'].fillna(merged_df['neu'].mean(), inplace=True)
    merged_df['pos'].fillna(merged_df['pos'].mean(), inplace=True)
    merged_df['compound'].fillna(0, inplace=True)
    merged_df['Subjectivity'].fillna(0, inplace=True)
    merged_df['Polarity'].fillna(0, inplace=True)
    merged_df['Sentiment_NLTK'].fillna('Neutral', inplace=True)
    merged_df['Sentiment_TB'].fillna('Neutral', inplace=True)
    
    sentiment = merged_df
    sentiment['Negative'] = sentiment['neg']
    sentiment['Positive'] = sentiment['pos']
    sentiment['Neutral'] = sentiment['neu']

    return sentiment


API_KEY = st.sidebar.text_input('Enter your NewsAPI API key',value='7d50571799064bdea99f596876917fb8')
#API_KEY = 'd195cbc01116434d8c32b702d4fe31ef'

value_mapping = {'1 hour': 60, '1 day': 1440}
time_step_name = st.sidebar.selectbox('Choose a value', list(value_mapping.keys()), index=0)
time_step = value_mapping[time_step_name]

ndays = st.sidebar.number_input('Enter the duration in days(max 30)', value=15, min_value=0, max_value=30)

END_DATE = datetime.now().strftime('%Y-%m-%d')  # Aujourd'hui
START_DATE = (datetime.now() - timedelta(days=ndays)).strftime('%Y-%m-%d')

st.sidebar.text(f'Start Date: {END_DATE}')
st.sidebar.text(f'End Date: {START_DATE}')


btc_data = download_btc_data(ndays,time_step)
news_df = download_news_data(ndays,time_step,API_KEY)
sentiment = sentiment_analysis(news_df)
overall = pd.merge(btc_data,sentiment,on='date',how='inner')



def recent_data():
    st.header('Bitcoin Price and Sentiment')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close Price','Sentiment Index','Price + Sentiment'])

    data = btc_data
  
    # #     # Create a candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'])])


    # Update layout for a stock market style
    fig.update_layout(xaxis_title='date',
                    yaxis_title='Close Price',
                    xaxis_rangeslider_visible=False,
                    template='plotly_dark')  # Use a dark theme for a stock market style



    if option == 'Close Price':
        st.write('Close Price')
        st.plotly_chart(fig)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'Sentiment Index':
        st.write('Sentiment Index')
        #st.line_chart(sentiment[['date','Negative','Neutral','Positive']],x='date',color=['#00FF00','#FFFFFF','#FF0000'])
        st.line_chart(sentiment[['date','Negative','Positive']],x='date',color=['#FF0000','#00FF00'])
    elif option == 'Price + Sentiment':
        st.write('BTC Price and Sentiment Index')
        # #     # Create a candlestick chart
        # fig_overall = go.Figure(data=[go.Candlestick(x=data.index,
        #                                     open=data['Open'],
        #                                     high=data['High'],
        #                                     low=data['Low'],
        #                                     close=data['Close']),go.Line(x=data.index, y=sentiment['Negative']*150000, mode='lines', name='Negative', line=dict(color='#FF0000')),
        #                                     go.Line(x=sentiment['date'], y=sentiment['Positive']*150000, mode='lines', name='Positive', line=dict(color='#00FF00'))
        #                                     ])


        # # Update layout for a stock market style
        # fig_overall.update_layout(xaxis_title='date',
        #                 yaxis_title='Close Price',
        #                 xaxis_rangeslider_visible=False,
        #                 template='plotly_dark')  # Use a dark theme for a stock market style
        # st.plotly_chart(fig_overall)
        fig_overall = go.Figure(data=[go.Candlestick(x=data.index,
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'],
                                                    name='Close Price')])
        
        fig_overall.add_trace(go.Line(x=data.index, y=sentiment['Negative'], mode='lines', name='Negative', line=dict(color='#FF0000'),yaxis='y2'))
        fig_overall.add_trace(go.Line(x=data.index, y=sentiment['Positive'], mode='lines', name='Positive', line=dict(color='#00FF00'),yaxis='y2'))

        fig_overall.update_layout(
        yaxis2=dict(
            title="Sentiment Intensity",
            overlaying="y",
            side="right",
            range=[0, 0.5],  # Set the range from 0 to 2
        )
    )



        # Update layout for a stock market style
        fig_overall.update_layout(xaxis_title='date',
                        yaxis_title='Close Price',
                        xaxis_rangeslider_visible=False,
                        template='plotly_dark')  # Use a dark theme for a stock market style
        fig_overall.update_layout(legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=1.1
                        ))
        #fig_overall.update_layout(height=600, width=960)
        st.plotly_chart(fig_overall)



def dataframe():
    st.header('Recent Data')
    data = btc_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    st.dataframe(data.tail(10))


def predict():
    st.header('Bitcoin Price Forecasting')
    days_to_forecast = st.number_input('How many days forecast?', value=5)
    days_to_forecast = int(days_to_forecast)

    forecast_dates,forecast_values,btc_predict = tm_model(ndays,days_to_forecast)
    # Display the forecasted values in Streamlit
    st.write(f"Forecasted Close Prices for the Next {days_to_forecast} Days:")

    # #     # Create a candlestick chart
    fig_predict = go.Figure(data=[go.Candlestick(x=btc_predict.index,
                                        open=btc_predict['Open'],
                                        high=btc_predict['High'],
                                        low=btc_predict['Low'],
                                        close=btc_predict['Close'])])


    # Update layout for a stock market style
    fig_predict.update_layout(xaxis_title='date',
                    yaxis_title='Close Price',
                    xaxis_rangeslider_visible=False,
                    template='plotly_dark')  # Use a dark theme for a stock market style

    # Add forecasted close prices to the chart
    fig_predict.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, name='Forecast'))

    # # Set chart title
    # fig.update_layout(title='Bitcoin Close Price Forecast')

    # Display the chart in Streamlit
    st.plotly_chart(fig_predict)

@st.cache_resource
def tm_model(ndays,days_to_forecast):
    btc_predict = download_btc_data(1000,1440)
        # Extract the close prices
    close_prices = btc_predict['Close']

    # Define ARIMA parameters
    p = 2  # Order of the AR term
    d = 1  # Order of the differencing
    q = 2  # Order of the MA term

    # # Train the ARIMA model
    # model = ARIMA(close_prices, order=(p, d, q))
    # model_fit = model.fit()

    # Division des données en ensembles d'entraînement et de test
    train_size = int(len(close_prices) * 0.8)
    train, test = close_prices[0:train_size], close_prices[train_size:]

    history = [x for x in train]  # Historique des observations pour l'entraînement itératif
    predictions = []  # Pour stocker les prédictions

    # Prédictions en marchant à travers le temps dans l'ensemble de test
    for t in range(len(test)):
        model = ARIMA(history, order=(p,d,q))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test.iloc[t])  # Ajouter l'observation réelle pour la prochaine itération

    # Forecast the next 5 days
    forecast_steps = days_to_forecast
    forecast = model_fit.get_forecast(steps=forecast_steps)

    # Get the forecasted values and dates
    forecast_values = forecast.predicted_mean
    # Generate continuous dates for the forecast period
    forecast_dates = pd.date_range(close_prices.index[-1] + timedelta(days=1), periods=forecast_steps)
    return forecast_dates,forecast_values,btc_predict

if __name__ == '__main__':
    main()
