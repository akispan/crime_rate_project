# crime_rate_project

## Description:

We collected 1,800 Press Releases from police from Jenuary 2019 to April 2021 and used them to estimate the crime rate between different areas in Greece. We used a model for Named Entity Recognition and finally we 
applied data visualization on map with heatmap. 

## Prerequisites:
Data and code were applied on Colab Notebook. 
### The modules need to be installed are:

spacy, spacy download en_core_web_sm, geopandas, geopy, spacy.cli.download("el_core_news_sm")

### The modules need to be imported are:

drive, json, spacy, spacy.cli, defaultdict, pandas, plotly.express, geopandas, geopy, Nominatim

## Packages Installation

In order to install the packages needed for this project that are not by default installed in colab, you need to use the command: !pip install <package>

## DATA:

Data are json formatted. The Releases are technically a list of lists grouped by date like the following example:
[
	['Jenuary 2019',
		[{'title':'A crime was commited today in Athens' subway',
		  'report': 'Some information about the crime in here'},
		  {'title':'A crime was commited today in Athens' subway',
		  'report': 'Some information about the crime in here'}
		],
	]
]

## Pipeline

After data collection, we manipulated them for better use in our model. We used the "el_core_news_sm" model from spacy, created by Prokopis Prokopidis for Named Entities Recognition.
At the end, we created a heatmap on map to visualize out results and also we created graphs per area to visualize how the crime rate changes from month to month.

## Authors
Akis Panagiotou

## Acknowledgments
I would like to express my gratitude and appreciation for Dimitris Pappas whose guidance, support and encouragement has been invaluable throughout this project.