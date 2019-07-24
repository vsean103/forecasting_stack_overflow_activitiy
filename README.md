# Forecasting Stack Overflow Activity with Time Series and Deep Learning

### Abstract:

Stack Overflow (SO) has become the most widely used question-and-answer platform in the past decade. The success of SO motivated researchers to explore and analyze its rich dataset. In this paper, we analyze SO activity using the SOTorrent dataset by applying seasonal decomposition to investigate trends in growth and seasonality relating to SO post activity. We base our time series analysis on the number of posts per day, as well as total posts per week and month. Our results expose seasonal behaviors in weekly and yearly periods through an additive time series model. Knowing the popularity of days and seasons can be important in maximizing post and advertisement exposure, as well as aid in anticipating network load. Thus, we extend our analysis to accurately forecast the SO activity of posts using Long Short Term Memory (LSTM) neural networks.

## Getting Started

### Prerequisites

Some python packages/libraries: numpy, pandas, keras

## Data

The data that we used in this project is publicly available at https://empirical-software.engineering/projects/sotorrent/

## Methods

We use 2 methods in our analysis.
- Seasonal and Trend Decompo- sition using Loess (STL) [1]
- Long Short Term Memory (LSTM) [2]

## References

[1] R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning, “Stl: A seasonal-trend decomposition procedure based on loess (with discussion),” Journal of Official Statistics, vol. 6, pp. 3–73, 1990.

[2] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, pp. 1735–1780, Nov. 1997. [Online]. Available: http://dx.doi.org/10.1162/neco.1997.9.8.1735

## Authors

* Viseth Sean
* Natalie Best

## Acknowledgments

* Supervisor: Dr. Erik Linstead (Chapman University)
