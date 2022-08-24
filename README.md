# stock_Market_prediction
## Stock Market prediction using ML



# Contents:

# Introduction

# Background to Machine Learning                                                                       
# Background to Deep Learning                                                                             
# Background to Technical Analysis                                                                       
# People Associated with Technical Analysis                                                 
# Features 	     
# The Project                                                                                                
      Tech Stack                                                                                                            
      Data Collection                                                                                                   
      Normalizing the Data using scikit-learn                                                           
      LSTMs                                                                                                                     
	
# Result	
# P&G	
# Microsoft                                                                                                               
# Apple                                                                                                                     
# Tesla	

# Conclusion and Future Plans	
# Bibliography	
# References	

#  Chapter 1 Introduction


The stock market allows companies to raise money by offering stock shares and corporate bonds. It lets common investors participate in the financial achievements of the companies, make profits through capital gains, and earn money through dividends — although losses are also possible. 

While it is impossible to predict the movement of the stock market with absolute certainty, its movements do tend to echo over time. This means that there are trends that could be noticed and observed to help traders determine the direction in which the market is headed. Unsurprisingly, for stockbrokers and professionals in the finance industry, understanding these trends properly with the help of prediction software for forecasting is crucial for decision making. These trends can be best observed by using technical analysis,(Predicting the future trend of a stock price by looking at it historical data.) a discipline that enables predicting future stock trends from historical price data. This has proven to be absolutely essential for thousands of investors and traders all over the world and has allowed them to reap huge profits.

Predicting how the stock market will move is one of the most challenging issues due to many factors that are involved in the stock prediction, such as interest rates, politics, uncertain occurence and economic growth that make the stock market volatile and very hard to predict accurately. The prediction of shares offers huge chances for profit and is a major motivation for research in this area. Since stock investment is a major financial market activity, a lack of accurate knowledge and detailed information would lead to an inevitable loss of investment. The prediction of the stock market is a difficult task as market movements are always subject to uncertainties. Stock market prediction methods are divided into two main categories: technical and fundamental analysis. Technical analysis focuses on analyzing historical stock prices to predict future stock values (i.e. it focuses on the direction of prices). On the other hand, fundamental analysis relies mostly on analyzing unstructured textual information like financial news and earning reports.
Background to Machine Learning

Machine learning is a field of machine intelligence concerned with the design and development of algorithms and models that allow computers to learn without being explicitly programmed. Machine learning has many applications including those related to regression, classification, clustering, natural language processing, audio and video related, computer vision, etc. Machine learning requires training one or more models using different algorithms.

Machine learning (ML) extracts meaningful insights from raw data to quickly solve complex, data-rich business problems. ML algorithms learn from the data iteratively and allow computers to find different types of hidden insights without being explicitly programmed to do so. ML is evolving at such a rapid rate and is mainly being driven by new computing technologies.

Machine learning in business helps in enhancing business scalability and improving business operations for companies across the globe. Artificial intelligence tools and numerous ML algorithms have gained tremendous popularity in the business analytics community. Factors such as growing volumes, easy availability of data, cheaper and faster computational processing, and affordable data storage have led to a massive machine learning boom. Therefore, organizations can now benefit by understanding how businesses can use machine learning and implement the same in their own processes.

Our model utilizes a type of Machine Learning called Deep Learning.

Background to Deep Learning

Put simply, deep learning is a subset of machine learning which teaches machines to do what humans are naturally born with: learn by example. Though the technology is often considered a set of algorithms which ‘mimics the brain’, a more appropriate description would be a set of algorithms which ‘learns in layers’. It involves learning through layers that enable a computer to develop a hierarchy of complicated concepts from simpler concepts.

Deep learning is the central technology behind a lot of high-end innovations like driverless cars, voice control in devices like tablets, smartphones, hands-free speakers etc and many more. It’s offering results which weren’t possible before or even with traditional machine learning techniques. 


Background to Technical Analysis 

Before knowing the common technical patterns, one should know the basics to enable data scientists to do technical research. The data scientist should be familiar with the following terms-

Uptrend: When there are multiple Higher High and Higher Low. As every new Higher High is higher than the previous one then the stock is said to be in an Uptrend.
 

Downtrend: When there are multiple Lower Low and Higher Low. As every new Lower Low is lower than the previous one then the stock is said to be in a Downtrend.

 
Sideway Trend: When there are multiple Highs and Lows at the same point then the stock is said to have gone sideways. If this is happening at a very short time frame then this often indicates stagnation in the stock and the market is said to be less volatile. It is considered the ideal time to make very short-term predictions and book profit.
 
 

Support: This is the price point which the stock is not able to fall below. This acts as the support for the stock as when it reaches this point, the chances are it will bounce up from here. Thus, a data scientist needs to train their Machine Learning model to calculate or feed such points in their input data because if this point is breached then the stock can go into free fall for some time and the model should sell the stock (yes, in the stock market on can also sell a stock which they haven’t bought. Its explanation is beyond the scope of this article but if you sell the stock (aka short the stock) then profit is made if the price drops). And if the stock price bounces up from the support then the price will rise up from here and the predictive model should buy the stock. This bouncing back technically is known as trend reversal. 
 

Resistance: Similarly resistance is a virtual point which the share is not able to cross. When the stock price get stuck between them then the share is known to have gone sideways.
 

Now, We’ll discuss some common patterns found during Technical Research.

With this understanding, we can quickly understand the common technical research that people do, and a data scientist should also look for and train their model to do the same.

(i) Head and Shoulders

This is when the stock reaches a peak and is surrounded by smaller peaks on each side. Here when the third peak is achieved, generally, the stock prices tend to fall.


 

(ii) Double Top

When a stock price goes up and hits a peak at the same point twice, then this pattern is known as Double top. When a stock hits a double top, there are high chances that the price will fall.

 
(iii) Double Bottom

It is the opposite of Double top, and when the second bottom is hit, the price goes with an uptrend.

 


(iv) Descending Triangle

The stock price will fall dramatically when the support remains stagnant, and the resistance points are made at a lower point.
 

(v) Ascending Triangle

It is the opposite of the Descending Triangle.

 
(vi) Symmetrical Triangle

When the highs and lows become very close to each other, and there is no space for the stock to move (forming a triangle), breakout happens, and the stock price goes dramatically up or down.

 



People associated with Technical Analysis

John Magee

John Magee wrote the bible of technical analysis, "Technical Analysis of Stock Trends" (1948). Magee was one of the first to trade solely on the stock price and its pattern on the historical charts. Magee charted everything: individual stocks, averages, trading volumes – basically anything that could be graphed. He then poured over these charts to identify broad patterns and specific shapes like weak triangles, flags, bodies, shoulders and so on. From his 40s to his death at 86, however, Magee was one of the most disciplined technical analysts around, refusing to even read a current newspaper lest it interfere with the signals of his charts. Later on, technical analysts built upon his work to develop the field of technical analysis and made it what it is today by employing diverse and effective algorithms to arrive at more concrete and accurate decisions. 

William P. Hamilton

William Peter Hamilton (1867-1929) was the fourth editor of the Wall Street Journal and a proponent of the Dow Theory. His greatest achievement was to understand the Dow Theory, introduced by Charles Dow, and make important additions to it to call market trends. The Dow Theory was a collection of market trends linked heavily to oceanic metaphors. The fundamental, long-term trend of four or more years was the tide of the market – either rising (bullish) or falling (bearish). This was followed by shorter-term waves that lasted between a week and a month. And, lastly, there were the splashes and tiny ripples of choppy water insignificant day-to-day fluctuations. 
Hamilton used these measures in addition to a few rules – such as the railroad average and the industrial average confirming each other's direction – to call bull and bear markets with laudable accuracy. 


# Chapter 2 


# Features 



1.	Collects data using the pandas_datareader() library from Tiingo.

2.	The data collected is from the last 5 years from the present date.

3.	The dataset is then preprocessed efficiently to homogenize the data available for analysis. 

4.	Usage of a stacked LSTM (Long Short Term Memory) model which enables the model to learn important characteristics and improve accuracy in time-series classification.

5.	Model will predict the training and test data output and visualize it.

6.	Additionally, the model will predict the price of the stock being observed for the next 30 days and visualize it.







 #   Chapter 3
 
 
   The Project


While the field of technical analysis is now widely popular among industry professionals, it is still common to see the everyday investor make investments solely on the basis of their gut instincts. Additionally, a lot of people all over the globe still remain skeptical about investing in the stock market due to its volatility. However, the advent of Artificial Intelligence, Machine Learning, Deep Learning and related fields have made it possible to reduce the unpredictability factor and the volatility associated with the stock market to a large extent.

So, we have tried to create a similar model which takes the performance of a stock over the last 5 years as input to help predict its performance over the next 30 days. We hope our project will help reassure reluctant investors about the risks involved in investing and help reaffirm the importance of technical analysis to investors who continue to be reliant on their instincts.

Tech Stack
pandas_datareader() for data collection

●	Pandas Datareader is a Python package that allows us to create a pandas DataFrame object by leveraging various data sources from the internet. It is popularly used for working with realtime stock price datasets.

●	Some of the sources used for obtaining data about stocks are Yahoo Finance, Google Finance, Quandl, Morningstar, IEX, Tiingo et cetera. For this project, we have used Tiingo to collect data relevant to the stock being analysed. 




Matplotlib for visualization
●	Matplotlib is the basic visualizing or plotting library of python. Matplotlib is a useful and a powerful tool for executing a variety of tasks. It can help us create different types of visualization reports like line plots, scatter plots, histograms, bar charts, pie charts, box plots, and many more different plots. It also supports 3-dimensional plotting.
●	Most of the Matplotlib utilities lie under the pyplot submodule, and are usually imported under the plt alias:
import matplotlib.pyplot as plt.

   scikit-learn for normalizing data

●	Scikit-Learn, also known as sklearn is a python library to implement machine learning models and statistical modelling. Through scikit-learn, we can implement various machine learning models for regression, classification, clustering, and statistical tools for analyzing these models. It also provides functionality for dimensionality reduction, feature selection, feature extraction, ensemble techniques, and inbuilt datasets. We will be looking into these features one by one. This library is built upon NumPy, SciPy, and Matplotlib

●	scikit-learn is an indispensable part of the Python machine learning toolkit at various organizations. It is very widely used across all parts of the bank for classification, predictive analytics, and many other machine learning tasks including normalization of data.

●	Normalization in machine learning is the process of translating data into the range [0, 1] (or any other range) or simply transforming data onto the unit sphere.





   
Keras for Models


●	Keras is a high-level, deep learning API developed by Google for implementing neural networks. It is written in Python and is used to make the implementation of neural networks easy. It also supports multiple backend neural network computation.

●	TensorFlow has adopted Keras as its official high-level API. Keras is embedded in TensorFlow and can be used to perform deep learning fast as it provides inbuilt modules for all neural network computations. At the same time, computation involving tensors, computation graphs, sessions, etc. can be custom made using the Tensorflow Core API, which gives us total flexibility and control over our application and lets us implement our ideas in a relatively short time.

●	Apart from this, Keras has numerous useful features. It runs smoothly on both CPU and GPU, it supports almost all neural network models, and it is modular in nature, which makes it expressive, flexible, and apt for innovative research.

●	The research community for Keras is vast and highly developed. The documentation and help available are far more extensive than other deep learning frameworks. 

●	Keras is used commercially by many companies like Netflix, Uber, Square, Yelp, etc which have deployed products in the public domain which are built using Keras. 

   Google Colab for IDE

●	Colaboratory, or “Colab” for short, are Jupyter Notebooks hosted by Google that allow us to write and execute Python code through our browser. It is easy to use a Colab and linked with your Google account. Colab provides free access to GPUs and TPUs, requires zero configuration, and ease of sharing our code with the community.

●	Google Colab has many interactive tutorials to help budding data scientists learn more about machine learning and neural networks. It also allows us to import datasets from foreign sources such as kaggle. Additionally, it also saves our notebooks to google drive, thereby keeping our data safe and backed-up.

●	Another great feature that Google Colab offers is the collaboration feature. If you are working with multiple developers on a project, it is great to use Google Colab notebook. Just like collaborating on a Google Docs document, you can co-code with multiple developers using a Google Colab notebook. Besides, you can also share your completed work with other developers.
Data Collection

The data for the stock was collected by using the pandas_datareader() package. As mentioned before, its major advantage is that it converts data into a pandas DataFrame object. Accessing data from Tiingo requires an API key which can be obtained for free by signing up on the Tiingo website. Generating this key allows the user to make 50 free requests for stocks data per day.

The data we have collected is for the last 5 years from present day. This helps our model make the most relevant predictions which should help investors make up their minds about investing in a particular security.

Normalizing the data using scikit-learn()

The data that we have obtained is not normalized and the range for each column varies. Normalizing data helps the algorithm to converge, as in, to find the local or global minimum effectively. For this, we have used MinMaxScaler from scikit-learn. But we do this after we are done splitting the dataset into training and testing datasets.
LSTMs

Long Short Term Memory networks, or LSTMs in short, are a special kind of Recurrent Neural Network (RNN) capable of learning long-term dependencies. LSTMs were introduced by Hochreiter & Schmidhuber (1997) and have only gone from strength to strength since then in terms of their usefulness. LSTMs work extremely effectively on a lot of problems, and are widely used nowadays by students and professionals.

In this project, we’ll be using Stacked LSTMs. Stacking LSTM hidden layers makes the model deeper, thereby validating its description as a deep learning technique. Generally, it is the depth of a neural network that is attributed to the success of an approach on a prediction problem. 

Stacked LSTMs or Deep LSTMs were introduced by Graves, et al. in their application of LSTMs for speech recognition, when their LSTM model ended up beating an important benchmark on a challenging standard problem. 
Stacked LSTMs have now become a stable technique for difficult sequence prediction problems. A Stacked LSTM architecture can be defined as an LSTM model comprising of multiple LSTM layers. An LSTM layer above provides a sequence output rather than a single value output to the LSTM layer below. Specifically speaking, it provides one output per input time step, rather than one output time step for all input time steps. 

LSTMs consume input in format [ samples, timesteps, Features ], i.e., a 3- dimensional array. 
●	Samples tells us the number of samples of input we want our Neural Network to see before updating weights. It has been verified that using a vert small batch size reduces the speed of training, whereas usinga very big batch size (for example, an entire dataset) reduces the model’s ability too generalize between different data while aso consuming more memory.
●	Timesteps define how many units back in time we want our network to study. In our case we will be using 100 as timestep i.e. we will look into 100 previous days of data to predict next day’s price.
●	Features is the number of attributes used to represent each time step.

Creating our model using LSTM

Here is a small snippet of the code involved

 


This shows that the LSTM network has been created.


 
Predicting Test Data and Plotting the Output (for P&G)

 

In this plot, the blue line indicates the closing price of P&G over the course of the last 5 years. The orange line indicates the results of the predictions made on the training dataset. The green line indicates the results of the predictions made on the testing dataset.



Predicting Test Data and Plotting the Output (for Microsoft)


 


In this plot, the blue line indicates the closing price of Microsoft over the course of the last 5 years. The orange line indicates the results of the predictions made on the training dataset. The green line indicates the results of the predictions made on the testing dataset.

Predicting Test Data and Plotting the Output (for Apple)
 

In this plot, the blue line indicates the closing price of Apple over the course of the last 5 years. The orange line indicates the results of the predictions made on the training dataset. The green line indicates the results of the predictions made on the testing dataset.

Predicting Test Data and Plotting the Output (for Tesla)

 
In this plot, the blue line indicates the closing price of Tesla over the course of the last 5 years. The orange line indicates the results of the predictions made on the training dataset. The green line indicates the results of the predictions made on the testing dataset. 
          
          
          
 #   Chapter 4
                
                
  #    Result

For P&G
Now that our model has been trained and tested, it is ready to give us a reliable prediction about the closing price of the P&G stock for the next 30 days.
Therefore, the P&G stock direction and trend for the next 30 days according to our model will be something like this,

 

The wave from day 60 to day 90 indicates the pattern and the direction of the flow of the P&G stock over the next 30 days.

The wave from day 60 to day 90 indicates the pattern and the direction of the flow of the P&G stock over the next 30 days.
And now the same plot including data from the past 5 years and our prediction for the next 30 days.

 
For Microsoft

 

The wave from day 60 to day 90 indicates the pattern and the direction of the flow of the Microsoft stock over the next 30 days.

The wave from day 60 to day 90 indicates the pattern and the direction of the flow of the Microsoft stock over the next 30 days.
And now the same plot including data from the past 5 years and our prediction for the next 30 days.
 


For Apple
 

The wave from day 60 to day 90 indicates the pattern and the direction of the flow of the Apple stock over the next 30 days.

The wave from day 60 to day 90 indicates the pattern and the direction of the flow of the Apple stock over the next 30 days.
And now the same plot including data from the past 5 years and our prediction for the next 30 days.
 




For Tesla
 
The wave from day 60 to day 90 indicates the pattern and the direction of the flow of the Tesla stock over the next 30 days.

The wave from day 60 to day 90 indicates the pattern and the direction of the flow of the Tesla stock over the next 30 days.
And now the same plot including data from the past 5 years and our prediction for the next 30 days.

 


#   Chapter 5


Conclusion and Future Plans

# Conclusion
In conclusion, based on predictions made by our model which leverages stacked LSTMs, we can make sound and reliable predictions to help investors and traders gain a better understanding of the expected flow of the stock market in general. 

Future Work
Currently, our group leverages stacked LSTMs to make predictions about the price of a security listed on the stock market based on its price alone. To further evolve and enhance this model, our group will now endeavour to further increase the accuracy of the predictions made by our model.
We wish to do this by including sentiment analysis of tweets that can cause unexpected but sustained ripples in the market. These fluctuations can also play a major role in the value of securities listed on the stock market.
Additionally, we also wish to include the effects of the sentiments of news headlines on the price of a security. This will further increase the reliability of the predictions made by our model.
Furthermore, we wish to expand our model to make predictions about the Indian stock market too. This will help us increase the relevance of our model since the country we're settled in is India.







# Chapter 6


 # Bibliography

[1]	https://www.sciencedirect.com/science/article/pii/S1877050920304865
[2]	https://colah.github.io/posts/2015-08-Understanding-LSTMs/
[3]	https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
[4]	https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944
[5]	https://pypi.org/project/pandas-datareader/
[6]	https://api.tiingo.com/documentation/appendix/developers
[7]	https://www.investopedia.com/articles/financial-theory/10/pioneers-technical-analysis.asp
[8]	https://www.fool.com/investing/how-to-invest/stocks/stock-market-volatility/#:~:text=Stock%20market%20volatility%20is%20a,varies%20from%20its%20average%20price
[9]	https://www.simplilearn.com/tutorials/deep-learning-tutorial/what-is-keras
[10]	https://www.intechopen.com/chapters/72381
[11]	https://towardsdatascience.com/4-reasons-why-you-should-use-google-colab-for-your-next-project-b0c4aaad39ed
[12]	https://becominghuman.ai/deep-learning-and-its-5-advantages-eaeee1f31c86
