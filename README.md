# crime_rate_project

## Description:

We collected 1,800 Press Releases from police from Jenuary 2019 to April 2021 and used them to estimate the crime rate between different areas in Greece. We used a model for Named Entity Recognition and finally we 
applied data visualization on map with heatmap. 

Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites:
Data and code were applied on Colab Notebook. 
### The modules need to be installed are:

spacy, spacy download en_core_web_sm, geopandas, geopy, spacy.cli.download("el_core_news_sm")

### The modules need to be imported are:

drive, json, spacy, spacy.cli, defaultdict, pandas, plotly.express, geopandas, geopy, Nominatim

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


Give examples
Installing
A step by step series of examples that tell you how to get a development env running

Say what the step will be

Give the example
And repeat

until finished
End with an example of getting some data out of the system or using it for a little demo

Running the tests
Explain how to run the automated tests for this system

Break down into end to end tests
Explain what these tests test and why

Give an example
And coding style tests
Explain what these tests test and why

Give an example
Deployment
Add additional notes about how to deploy this on a live system

Built With
Dropwizard - The web framework used
Maven - Dependency Management
ROME - Used to generate RSS Feeds
Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

Versioning
We use SemVer for versioning. For the versions available, see the tags on this repository.

Authors
Billie Thompson - Initial work - PurpleBooth
See also the list of contributors who participated in this project.

License
This project is licensed under the MIT License - see the LICENSE.md file for details

Acknowledgments
Hat tip to anyone whose code was used
Inspiration
etc