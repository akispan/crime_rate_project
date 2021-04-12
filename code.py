# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)

import spacy
import spacy.cli
from collections import defaultdict
%matplotlib inline
spacy.cli.download("el_core_news_sm")
import json
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import geopandas
import geopy
from geopy import Nominatim

# !pip install spacy
# !pip install spacy download en_core_web_sm
# !pip install geopandas
# !pip install geopy



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def has_number(s):
  for i in s:
      if is_number(i):
          return True
  return False

def get_capitalized(item):
    letters = {'Ά':'Α','Έ':'Ε','Ή':'Η','Ί':'Ι','Ό':'Ο','Ύ':'Υ','Ώ':'Ω'}
    item = item.upper()
    for i in item:
        if i in letters.keys():
            item = item.replace(i, letters[i])
    return item

def get_tokens_from_text(mytext):
    tokens = defaultdict(int)
    nlp = spacy.load("el_core_news_sm")
    doc = nlp(mytext)
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            entity = get_capitalized(ent.text)
            tokens[entity] = 1
    return tokens



with open('/content/drive/My Drive/data_for_colab/police/data.json') as f:
    data = json.load(f)
with open('/content/drive/My Drive/data_for_colab/police/tokens_keep.json') as f:
    tokens_keep = json.load(f)
with open('/content/drive/My Drive/data_for_colab/police/name_correction.json') as f:
    name_correction = json.load(f)



original_size = 0
delete_size = 0
for item in data:

  for i in reversed(item[1]):

    original_size += 1
    if i['text'] == 'Δείτε τον πίνακα εδώ':
      item[1].remove(i)
      delete_size += 1

print('Total   size :',original_size)
print('Anused  size :',delete_size)
print('Process size :',original_size - delete_size)


for month in data:

    for report in month[1]:

        if 'ΔΕΛΤΙΟ ΤΥΠΟΥ' in report['text']:
            report['text'] = report['text'].split('ΔΕΛΤΙΟ ΤΥΠΟΥ')[1].strip()
        elif 'ΑΝΑΚΟΙΝΩΣΗ' in report['text']:
            report['text'] = report['text'].split('ΑΝΑΚΟΙΝΩΣΗ')[1].strip()


print('--- Find Tokens extraction Started ---')

tokens_dict = defaultdict(int)

pbar = tqdm(data, total=len(data))
for month in pbar:

    tokens_per_month = defaultdict(int)
    # print('month')
    for report in month[1]:

        text = report['text']
        tokens = get_tokens_from_text(text)
        # print(tokens)
        for k in tokens:
            if k in tokens_keep:
                if k in name_correction:
                    tokens_per_month[name_correction[k]] += 1
                else:
                    tokens_per_month[k] += 1
    tokens_dict.update({month[0]: tokens_per_month})

tokens = defaultdict(int)

for res in tokens_dict.values():
    for k, val in res.items():
        tokens[k] += val


locator = Nominatim(user_agent='myGeocoder')

latitude = []
longitude = []
times = []

for loc in sorted(tokens.keys()):

    # print(loc)
    location = locator.geocode(loc)

    try:
        latitude.append(location.latitude)
        longitude.append(location.longitude)
        times.append(tokens[loc])
        # print('Latitude, Longitude : {} {}'.format(location.latitude, location.longitude))
    except:
        print(loc)



df = pd.DataFrame(zip(latitude, longitude, times), columns = ['Latitude', 'Longitude', 'Magnitude'])

fig = px.density_mapbox(df, lat='Latitude', lon='Longitude', z='Magnitude',
                        color_continuous_midpoint = 0,
                        radius=30,
                        center=dict(lat=37.97728639084097, lon=23.726664468538466),
                        zoom=6,
                        width=1000,
                        height=1000,
                        mapbox_style="stamen-terrain")

fig.update_layout(margin=dict(b=0, t=0, l=0, r=0))
fig.show()