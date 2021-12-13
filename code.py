# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)

# !pip install greek-stemmer
# !pip install pyldavis
# !pip install spacy
# !pip install spacy download en_core_web_sm
# !pip install geopandas
# !pip install geopy

nltk.download('stopwords')
spacy.cli.download("el_core_news_sm")
import spacy
import spacy.cli
from collections import defaultdict

import os
import re
import json
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import geopandas
import geopy
from geopy import Nominatim
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import nltk
from greek_stemmer import GreekStemmer
from nltk.corpus import stopwords
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pickle 

# Load data from Google Drive
with open('/content/drive/My Drive/data_for_colab/police2/data.json') as f:
    data = json.load(f)
    
# We deleted text without interest because it contains only tables

original_size = 0
deleted_size = 0
for item in data:
  for i in reversed(item[1]):
    original_size += 1
    if 'Δείτε τον πίνακα εδώ' in i['text'] or 'Δείτε εδώ αναλυτικό'in i['text']:
      item[1].remove(i)
      deleted_size += 1
print('Total   size :',original_size)
print('Anused  size :',deleted_size)
print('Process size :',original_size - deleted_size)

# Data Cleaning: In some texts we need to cut the beginning sentence because it's repeated 

for month in data:
  for report in month[1]:
    if 'ΔΕΛΤΙΟ ΤΥΠΟΥ' in report['text']:
      report['text'] = report['text'].split('ΔΕΛΤΙΟ ΤΥΠΟΥ')[1].strip()
    elif 'ΑΝΑΚΟΙΝΩΣΗ' in report['text']:
      report['text'] = report['text'].split('ΑΝΑΚΟΙΝΩΣΗ')[1].strip()

# Tokens_keep is a list of GPE places to keep
# Name_correction is a dict with place names for correction
# Both dictionaries are loaded from Google Drive
with open('/content/drive/My Drive/data_for_colab/police2/tokens_keep.json') as f:
    tokens_keep = json.load(f)
with open('/content/drive/My Drive/data_for_colab/police2/name_correction.json') as f:
    name_correction = json.load(f)
    
# Functions:
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

def get_token_per_month(data):
    tokens_dict = defaultdict(int)

    pbar = tqdm(data, total=len(data))
    for month in pbar:

        tokens_per_month = defaultdict(int)
        for report in month[1]:

            text = report['text']
            tokens = get_tokens_from_text(text)
            for k in tokens:
                if k in tokens_keep:
                    if k in name_correction:
                        tokens_per_month[name_correction[k]] += 1  
                    else:
                        tokens_per_month[k] += 1 
        tokens_dict.update({month[0]: tokens_per_month})

    return tokens_dict

def get_all_tokens(tokens_dict):
    tokens = defaultdict(int)

    for res in tokens_dict.values():
        # print(res)
        for k, val in res.items():
            tokens[k] += val

    return tokens
    
print('--- Find Tokens extraction Started ---')
tokens_dict = get_token_per_month(data)
tokens      = get_all_tokens(tokens_dict)


def get_coordinates(tokens):
    locator = Nominatim(user_agent='my-application')

    latitude  = []
    longitude = []
    times     = []

    for loc in sorted(tokens.keys()):

      #location = locator.geocode(loc)
      
      try:
        location = locator.geocode(loc)
        latitude.append(location.latitude)
        longitude.append(location.longitude)
        times.append(tokens[loc])
      except:
        print(loc)

    return latitude, longitude, times
    
def show_map(latitude, longitude, times):
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
    
show_map(latitude, longitude, times)

# We want to plot a Graph that shows the frequency of actions taken place for a specific area in 
# the last 3 years.
years = ['2019','2020','2021']
months = ['Ιανουάριος ','Φεβρουάριος ','Μάρτιος ','Απρίλιος ','Μάιος ','Ιούνιος ','Ιούλιος ', 'Αύγουστος ','Σεπτέμβριος ','Οκτώβριος ','Νοέμβριος ','Δεκέμβριος ']

dates = []          
for year in years:
  
  if year == '2021':
    for month in months[:10]:
      dates.append(month + year)
  else:
    for month in months:
      dates.append(month + year)
print(dates)


areas = list(sorted(tokens))

graph_data = {}

for area in areas:

    month_array = [0]*34

    for i, date in zip(range(34), dates):
        if area in tokens_dict[date]:
            month_array[i] = tokens_dict[date][area]

    graph_data.update({area:month_array})



def plot_line(dates, graph_data, area):
  
  plt.rcParams["figure.figsize"] = (20,10)
  plt.rcParams["axes.grid"] = True
  matplotlib.rc('xtick', labelsize=15) 
  matplotlib.rc('ytick', labelsize=15) 
  fig, ax = plt.subplots()
  fig.suptitle('Γραφημα ανα μήνα για την περιοχη {}'.format(area), fontsize=16)
  plt.xlabel('μήνες', fontsize=16)
  plt.ylabel('εμφανίσεις', fontsize=16)
  ndx = [x for x in range (len(dates))]
  plt.xticks(ndx, dates, rotation='vertical')
  plt.plot( dates , graph_data)
  # plt.legend(shadow=True, fontsize="large", loc="best")
  # plt.ylim(0,1.1)   #  creates y scale from 0 to 1
  plt.show()


plot_line(dates, graph_data['ΑΘΗΝΑ'], 'ΑΘΗΝΑ')



li = []
for item in data:
    for report in item[1]:
        # print(type(report['title']), type(report['text']))
        res = report['title'] + ' ' + report['text']
        li.append(res)

df = pd.DataFrame(li, columns=['Titles_and_Text'])



stop = ['μη','εαυτου','αν','αλλ’','αλλα','αλλοσ','απο','αρα','αυτοσ','δ’','δε','δη','δια','δαι','δαισ','ετι', 'εγω','εκ','εμοσ','εν','επι','εἰ','εἰμι','ειμι','εἰσ','γαρ','γε','γα^','η','η','και','κατα','μεν','μετα','ο','οδε','οσ','οστισ','οτι','ουτωσ','ουτοσ','ουτε','ουν','ουδεισ','οἱ','ου','ουδε','ουκ','περι','προσ','συ','συν','τα','τε','την','τησ','τῇ','τι','τι','τισ','τισ','το','τοι','τοιουτοσ','τον','τουσ','του','τῶν','τῷ','ὑμοσ','ὑπερ','ὑπο','ὡσ','ὦ','ὥστε','εαν','παρα','σοσ','ο','η','το','οι','τα','του','τησ','των','τον','την','και','κι','κ','ειμαι','εισαι','ειναι','ειμαστε','ειστε','στο','στον','στη','στην','μα','αλλα','απο','για','προσ','με','σε','ωσ','παρα','αντι','κατα','μετα','θα','να','δε','δεν','μη','μην','επι','ενω','εαν','αν','τοτε','που','πωσ','ποιοσ','ποια','ποιο','ποιοι','ποιεσ','ποιων','ποιουσ','αυτοσ','αυτη','αυτο','αυτοι','αυτων','αυτουσ','αυτεσ','αυτα','εκεινοσ','εκεινη','εκεινο','εκεινοι','εκεινεσ','εκεινα','εκεινων','εκεινουσ','οπωσ','ομωσ','ισωσ','οσο','οτι','Α∆ΙΑΚΟΠΑ', 'ΑΙ', 'ΑΚΟΜΑ', 'ΑΚΟΜΗ', 'ΑΚΡΙΒΩΣ', 'ΑΛΗΘΕΙΑ', 'ΑΛΗΘΙΝΑ', 'ΑΛΛΑ', 'ΑΛΛΑΧΟΥ', 'ΑΛΛΕΣ', 'ΑΛΛΗ', 'ΑΛΛΗΝ', 'ΑΛΛΗΣ', 'ΑΛΛΙΩΣ', 'ΑΛΛΙΩΤΙΚΑ', 'ΑΛΛΟ', 'ΑΛΛΟΙ', 'ΑΛΛΟΙΩΣ', 'ΑΛΛΟΙΩΤΙΚΑ', 'ΑΛΛΟΝ', 'ΑΛΛΟΣ', 'ΑΛΛΟΤΕ', 'ΑΛΛΟΥ', 'ΑΛΛΟΥΣ', 'ΑΛΛΩΝ', 'ΑΜΑ', 'ΑΜΕΣΑ', 'ΑΜΕΣΩΣ', 'ΑΝ', 'ΑΝΑ', 'ΑΝΑΜΕΣΑ', 'ΑΝΑΜΕΤΑΞΥ', 'ΑΝΕΥ', 'ΑΝΤΙ', 'ΑΝΤΙΠΕΡΑ', 'ΑΝΤΙΣ', 'ΑΝΩ', 'ΑΝΩΤΕΡΩ', 'ΑΞΑΦΝΑ', 'ΑΠ', 'ΑΠΕΝΑΝΤΙ', 'ΑΠΟ', 'ΑΠΟΨΕ', 'ΑΡΑ', 'ΑΡΑΓΕ', 'ΑΡΓΑ', 'ΑΡΓΟΤΕΡΟ', 'ΑΡΙΣΤΕΡΑ', 'ΑΡΚΕΤΑ', 'ΑΡΧΙΚΑ', 'ΑΣ', 'ΑΥΡΙΟ', 'ΑΥΤΑ', 'ΑΥΤΕΣ', 'ΑΥΤΗ', 'ΑΥΤΗΝ', 'ΑΥΤΗΣ', 'ΑΥΤΟ', 'ΑΥΤΟΙ', 'ΑΥΤΟΝ', 'ΑΥΤΟΣ', 'ΑΥΤΟΥ', 'ΑΥΤΟΥΣ', 'ΑΥΤΩΝ', 'ΑΦΟΤΟΥ', 'ΑΦΟΥ', 'ΒΕΒΑΙΑ', 'ΒΕΒΑΙΟΤΑΤΑ', 'ΓΙ', 'ΓΙΑ', 'ΓΡΗΓΟΡΑ', 'ΓΥΡΩ', '∆Α', '∆Ε', '∆ΕΙΝΑ', '∆ΕΝ', '∆ΕΞΙΑ', '∆ΗΘΕΝ', '∆ΗΛΑ∆Η', '∆Ι', '∆ΙΑ', '∆ΙΑΡΚΩΣ', '∆ΙΚΑ', '∆ΙΚΟ', '∆ΙΚΟΙ', '∆ΙΚΟΣ', '∆ΙΚΟΥ', '∆ΙΚΟΥΣ', '∆ΙΟΛΟΥ', '∆ΙΠΛΑ', '∆ΙΧΩΣ', 'ΕΑΝ', 'ΕΑΥΤΟ', 'ΕΑΥΤΟΝ', 'ΕΑΥΤΟΥ', 'ΕΑΥΤΟΥΣ', 'ΕΑΥΤΩΝ', 'ΕΓΚΑΙΡΑ', 'ΕΓΚΑΙΡΩΣ', 'ΕΓΩ', 'Ε∆Ω', 'ΕΙ∆ΕΜΗ', 'ΕΙΘΕ', 'ΕΙΜΑΙ', 'ΕΙΜΑΣΤΕ', 'ΕΙΝΑΙ', 'ΕΙΣ', 'ΕΙΣΑΙ', 'ΕΙΣΑΣΤΕ', 'ΕΙΣΤΕ', 'ΕΙΤΕ', 'ΕΙΧΑ', 'ΕΙΧΑΜΕ', 'ΕΙΧΑΝ', 'ΕΙΧΑΤΕ', 'ΕΙΧΕ', 'ΕΙΧΕΣ', 'ΕΚΑΣΤΑ', 'ΕΚΑΣΤΕΣ', 'ΕΚΑΣΤΗ', 'ΕΚΑΣΤΗΝ', 'ΕΚΑΣΤΗΣ', 'ΕΚΑΣΤΟ', 'ΕΚΑΣΤΟΙ', 'ΕΚΑΣΤΟΝ', 'ΕΚΑΣΤΟΣ', 'ΕΚΑΣΤΟΥ', 'ΕΚΑΣΤΟΥΣ', 'ΕΚΑΣΤΩΝ', 'ΕΚΕΙ', 'ΕΚΕΙΝΑ', 'ΕΚΕΙΝΕΣ', 'ΕΚΕΙΝΗ', 'ΕΚΕΙΝΗΝ', 'ΕΚΕΙΝΗΣ', 'ΕΚΕΙΝΟ', 'ΕΚΕΙΝΟΙ', 'ΕΚΕΙΝΟΝ', 'ΕΚΕΙΝΟΣ', 'ΕΚΕΙΝΟΥ', 'ΕΚΕΙΝΟΥΣ', 'ΕΚΕΙΝΩΝ', 'ΕΚΤΟΣ', 'ΕΜΑΣ', 'ΕΜΕΙΣ', 'ΕΜΕΝΑ', 'ΕΜΠΡΟΣ', 'ΕΝ', 'ΕΝΑ', 'ΕΝΑΝ', 'ΕΝΑΣ', 'ΕΝΟΣ', 'ΕΝΤΕΛΩΣ', 'ΕΝΤΟΣ', 'ΕΝΤΩΜΕΤΑΞΥ', 'ΕΝΩ', 'ΕΞ', 'ΕΞΑΦΝΑ', 'ΕΞΗΣ', 'ΕΞΙΣΟΥ', 'ΕΞΩ', 'ΕΠΑΝΩ', 'ΕΠΕΙ∆Η', 'ΕΠΕΙΤΑ', 'ΕΠΙ', 'ΕΠΙΣΗΣ', 'ΕΠΟΜΕΝΩΣ', 'ΕΣΑΣ', 'ΕΣΕΙΣ', 'ΕΣΕΝΑ', 'ΕΣΤΩ', 'ΕΣΥ', 'ΕΤΕΡΑ', 'ΕΤΕΡΑΙ', 'ΕΤΕΡΑΣ', 'ΕΤΕΡΕΣ', 'ΕΤΕΡΗ', 'ΕΤΕΡΗΣ', 'ΕΤΕΡΟ', 'ΕΤΕΡΟΙ', 'ΕΤΕΡΟΝ', 'ΕΤΕΡΟΣ', 'ΕΤΕΡΟΥ', 'ΕΤΕΡΟΥΣ', 'ΕΤΕΡΩΝ', 'ΕΤΟΥΤΑ', 'ΕΤΟΥΤΕΣ', 'ΕΤΟΥΤΗ', 'ΕΤΟΥΤΗΝ', 'ΕΤΟΥΤΗΣ', 'ΕΤΟΥΤΟ', 'ΕΤΟΥΤΟΙ', 'ΕΤΟΥΤΟΝ', 'ΕΤΟΥΤΟΣ', 'ΕΤΟΥΤΟΥ', 'ΕΤΟΥΤΟΥΣ', 'ΕΤΟΥΤΩΝ', 'ΕΤΣΙ', 'ΕΥΓΕ', 'ΕΥΘΥΣ', 'ΕΥΤΥΧΩΣ', 'ΕΦΕΞΗΣ', 'ΕΧΕΙ', 'ΕΧΕΙΣ', 'ΕΧΕΤΕ', 'ΕΧΘΕΣ', 'ΕΧΟΜΕ', 'ΕΧΟΥΜΕ', 'ΕΧΟΥΝ', 'ΕΧΤΕΣ', 'ΕΧΩ', 'ΕΩΣ', 'Η', 'Η∆Η', 'ΗΜΑΣΤΑΝ', 'ΗΜΑΣΤΕ', 'ΗΜΟΥΝ', 'ΗΣΑΣΤΑΝ', 'ΗΣΑΣΤΕ', 'ΗΣΟΥΝ', 'ΗΤΑΝ', 'ΗΤΑΝΕ', 'ΗΤΟΙ', 'ΗΤΤΟΝ', 'ΘΑ', 'Ι', 'Ι∆ΙΑ', 'Ι∆ΙΑΝ', 'Ι∆ΙΑΣ', 'Ι∆ΙΕΣ', 'Ι∆ΙΟ', 'Ι∆ΙΟΙ', 'Ι∆ΙΟΝ', 'Ι∆ΙΟΣ', 'Ι∆ΙΟΥ', 'Ι∆ΙΟΥΣ', 'Ι∆ΙΩΝ', 'Ι∆ΙΩΣ', 'ΙΙ', 'ΙΙΙ', 'ΙΣΑΜΕ', 'ΙΣΙΑ', 'ΙΣΩΣ', 'ΚΑΘΕ', 'ΚΑΘΕΜΙΑ', 'ΚΑΘΕΜΙΑΣ', 'ΚΑΘΕΝΑ', 'ΚΑΘΕΝΑΣ', 'ΚΑΘΕΝΟΣ', 'ΚΑΘΕΤΙ', 'ΚΑΘΟΛΟΥ', 'ΚΑΘΩΣ', 'ΚΑΙ', 'ΚΑΚΑ', 'ΚΑΚΩΣ', 'ΚΑΛΑ', 'ΚΑΛΩΣ', 'ΚΑΜΙΑ', 'ΚΑΜΙΑΝ', 'ΚΑΜΙΑΣ', 'ΚΑΜΠΟΣΑ', 'ΚΑΜΠΟΣΕΣ', 'ΚΑΜΠΟΣΗ', 'ΚΑΜΠΟΣΗΝ', 'ΚΑΜΠΟΣΗΣ', 'ΚΑΜΠΟΣΟ', 'ΚΑΜΠΟΣΟΙ', 'ΚΑΜΠΟΣΟΝ', 'ΚΑΜΠΟΣΟΣ', 'ΚΑΜΠΟΣΟΥ', 'ΚΑΜΠΟΣΟΥΣ', 'ΚΑΜΠΟΣΩΝ', 'ΚΑΝΕΙΣ', 'ΚΑΝΕΝ', 'ΚΑΝΕΝΑ', 'ΚΑΝΕΝΑΝ', 'ΚΑΝΕΝΑΣ', 'ΚΑΝΕΝΟΣ', 'ΚΑΠΟΙΑ', 'ΚΑΠΟΙΑΝ', 'ΚΑΠΟΙΑΣ', 'ΚΑΠΟΙΕΣ', 'ΚΑΠΟΙΟ', 'ΚΑΠΟΙΟΙ', 'ΚΑΠΟΙΟΝ', 'ΚΑΠΟΙΟΣ', 'ΚΑΠΟΙΟΥ', 'ΚΑΠΟΙΟΥΣ', 'ΚΑΠΟΙΩΝ', 'ΚΑΠΟΤΕ', 'ΚΑΠΟΥ', 'ΚΑΠΩΣ', 'ΚΑΤ', 'ΚΑΤΑ', 'ΚΑΤΙ', 'ΚΑΤΙΤΙ', 'ΚΑΤΟΠΙΝ', 'ΚΑΤΩ', 'ΚΙΟΛΑΣ', 'ΚΛΠ', 'ΚΟΝΤΑ', 'ΚΤΛ', 'ΚΥΡΙΩΣ', 'ΛΙΓΑΚΙ', 'ΛΙΓΟ', 'ΛΙΓΩΤΕΡΟ', 'ΛΟΓΩ', 'ΛΟΙΠΑ', 'ΛΟΙΠΟΝ', 'ΜΑ', 'ΜΑΖΙ', 'ΜΑΚΑΡΙ', 'ΜΑΚΡΥΑ', 'ΜΑΛΙΣΤΑ', 'ΜΑΛΛΟΝ', 'ΜΑΣ', 'ΜΕ', 'ΜΕΘΑΥΡΙΟ', 'ΜΕΙΟΝ', 'ΜΕΛΕΙ', 'ΜΕΛΛΕΤΑΙ', 'ΜΕΜΙΑΣ', 'ΜΕΝ', 'ΜΕΡΙΚΑ', 'ΜΕΡΙΚΕΣ', 'ΜΕΡΙΚΟΙ', 'ΜΕΡΙΚΟΥΣ', 'ΜΕΡΙΚΩΝ', 'ΜΕΣΑ', 'ΜΕΤ', 'ΜΕΤΑ', 'ΜΕΤΑΞΥ', 'ΜΕΧΡΙ', 'ΜΗ', 'ΜΗ∆Ε', 'ΜΗΝ', 'ΜΗΠΩΣ', 'ΜΗΤΕ', 'ΜΙΑ', 'ΜΙΑΝ', 'ΜΙΑΣ', 'ΜΟΛΙΣ', 'ΜΟΛΟΝΟΤΙ', 'ΜΟΝΑΧΑ', 'ΜΟΝΕΣ', 'ΜΟΝΗ', 'ΜΟΝΗΝ', 'ΜΟΝΗΣ', 'ΜΟΝΟ', 'ΜΟΝΟΙ', 'ΜΟΝΟΜΙΑΣ', 'ΜΟΝΟΣ', 'ΜΟΝΟΥ', 'ΜΟΝΟΥΣ', 'ΜΟΝΩΝ', 'ΜΟΥ', 'ΜΠΟΡΕΙ', 'ΜΠΟΡΟΥΝ', 'ΜΠΡΑΒΟ', 'ΜΠΡΟΣ', 'ΝΑ', 'ΝΑΙ', 'ΝΩΡΙΣ', 'ΞΑΝΑ', 'ΞΑΦΝΙΚΑ', 'Ο', 'ΟΙ', 'ΟΛΑ', 'ΟΛΕΣ', 'ΟΛΗ', 'ΟΛΗΝ', 'ΟΛΗΣ', 'ΟΛΟ', 'ΟΛΟΓΥΡΑ', 'ΟΛΟΙ', 'ΟΛΟΝ', 'ΟΛΟΝΕΝ', 'ΟΛΟΣ', 'ΟΛΟΤΕΛΑ', 'ΟΛΟΥ', 'ΟΛΟΥΣ', 'ΟΛΩΝ', 'ΟΛΩΣ', 'ΟΛΩΣ∆ΙΟΛΟΥ', 'ΟΜΩΣ', 'ΟΠΟΙΑ', 'ΟΠΟΙΑ∆ΗΠΟΤΕ', 'ΟΠΟΙΑΝ', 'ΟΠΟΙΑΝ∆ΗΠΟΤΕ', 'ΟΠΟΙΑΣ', 'ΟΠΟΙΑΣ∆ΗΠΟΤΕ', 'ΟΠΟΙ∆ΗΠΟΤΕ', 'ΟΠΟΙΕΣ', 'ΟΠΟΙΕΣ∆ΗΠΟΤΕ', 'ΟΠΟΙΟ', 'ΟΠΟΙΟ∆ΗΠΟΤΕ', 'ΟΠΟΙΟΙ', 'ΟΠΟΙΟΝ', 'ΟΠΟΙΟΝ∆ΗΠΟΤΕ', 'ΟΠΟΙΟΣ', 'ΟΠΟΙΟΣ∆ΗΠΟΤΕ', 'ΟΠΟΙΟΥ', 'ΟΠΟΙΟΥ∆ΗΠΟΤΕ', 'ΟΠΟΙΟΥΣ', 'ΟΠΟΙΟΥΣ∆ΗΠΟΤΕ', 'ΟΠΟΙΩΝ', 'ΟΠΟΙΩΝ∆ΗΠΟΤΕ', 'ΟΠΟΤΕ', 'ΟΠΟΤΕ∆ΗΠΟΤΕ', 'ΟΠΟΥ', 'ΟΠΟΥ∆ΗΠΟΤΕ', 'ΟΠΩΣ', 'ΟΡΙΣΜΕΝΑ', 'ΟΡΙΣΜΕΝΕΣ', 'ΟΡΙΣΜΕΝΩΝ', 'ΟΡΙΣΜΕΝΩΣ', 'ΟΣΑ', 'ΟΣΑ∆ΗΠΟΤΕ', 'ΟΣΕΣ', 'ΟΣΕΣ∆ΗΠΟΤΕ', 'ΟΣΗ', 'ΟΣΗ∆ΗΠΟΤΕ', 'ΟΣΗΝ', 'ΟΣΗΝ∆ΗΠΟΤΕ', 'ΟΣΗΣ', 'ΟΣΗΣ∆ΗΠΟΤΕ', 'ΟΣΟ', 'ΟΣΟ∆ΗΠΟΤΕ', 'ΟΣΟΙ', 'ΟΣΟΙ∆ΗΠΟΤΕ', 'ΟΣΟΝ', 'ΟΣΟΝ∆ΗΠΟΤΕ', 'ΟΣΟΣ', 'ΟΣΟΣ∆ΗΠΟΤΕ', 'ΟΣΟΥ', 'ΟΣΟΥ∆ΗΠΟΤΕ', 'ΟΣΟΥΣ', 'ΟΣΟΥΣ∆ΗΠΟΤΕ', 'ΟΣΩΝ', 'ΟΣΩΝ∆ΗΠΟΤΕ', 'ΟΤΑΝ', 'ΟΤΙ', 'ΟΤΙ∆ΗΠΟΤΕ', 'ΟΤΟΥ', 'ΟΥ', 'ΟΥ∆Ε', 'ΟΥΤΕ', 'ΟΧΙ', 'ΠΑΛΙ', 'ΠΑΝΤΟΤΕ', 'ΠΑΝΤΟΥ', 'ΠΑΝΤΩΣ', 'ΠΑΡΑ', 'ΠΕΡΑ', 'ΠΕΡΙ', 'ΠΕΡΙΠΟΥ', 'ΠΕΡΙΣΣΟΤΕΡΟ', 'ΠΕΡΣΙ', 'ΠΕΡΥΣΙ', 'ΠΙΑ', 'ΠΙΘΑΝΟΝ', 'ΠΙΟ', 'ΠΙΣΩ', 'ΠΛΑΙ', 'ΠΛΕΟΝ', 'ΠΛΗΝ', 'ΠΟΙΑ', 'ΠΟΙΑΝ', 'ΠΟΙΑΣ', 'ΠΟΙΕΣ', 'ΠΟΙΟ', 'ΠΟΙΟΙ', 'ΠΟΙΟΝ', 'ΠΟΙΟΣ', 'ΠΟΙΟΥ', 'ΠΟΙΟΥΣ', 'ΠΟΙΩΝ', 'ΠΟΛΥ', 'ΠΟΣΕΣ', 'ΠΟΣΗ', 'ΠΟΣΗΝ', 'ΠΟΣΗΣ', 'ΠΟΣΟΙ', 'ΠΟΣΟΣ', 'ΠΟΣΟΥΣ', 'ΠΟΤΕ', 'ΠΟΥ', 'ΠΟΥΘΕ', 'ΠΟΥΘΕΝΑ', 'ΠΡΕΠΕΙ', 'ΠΡΙΝ', 'ΠΡΟ', 'ΠΡΟΚΕΙΜΕΝΟΥ', 'ΠΡΟΚΕΙΤΑΙ', 'ΠΡΟΠΕΡΣΙ', 'ΠΡΟΣ', 'ΠΡΟΤΟΥ', 'ΠΡΟΧΘΕΣ', 'ΠΡΟΧΤΕΣ', 'ΠΡΩΤΥΤΕΡΑ', 'ΠΩΣ', 'ΣΑΝ', 'ΣΑΣ', 'ΣΕ', 'ΣΕΙΣ', 'ΣΗΜΕΡΑ', 'ΣΙΓΑ', 'ΣΟΥ', 'ΣΤΑ', 'ΣΤΗ', 'ΣΤΗΝ', 'ΣΤΗΣ', 'ΣΤΙΣ', 'ΣΤΟ', 'ΣΤΟΝ', 'ΣΤΟΥ', 'ΣΤΟΥΣ', 'ΣΤΩΝ', 'ΣΥΓΧΡΟΝΩΣ', 'ΣΥΝ', 'ΣΥΝΑΜΑ', 'ΣΥΝΕΠΩΣ', 'ΣΥΝΗΘΩΣ', 'ΣΥΧΝΑ', 'ΣΥΧΝΑΣ', 'ΣΥΧΝΕΣ', 'ΣΥΧΝΗ', 'ΣΥΧΝΗΝ', 'ΣΥΧΝΗΣ', 'ΣΥΧΝΟ', 'ΣΥΧΝΟΙ', 'ΣΥΧΝΟΝ', 'ΣΥΧΝΟΣ', 'ΣΥΧΝΟΥ', 'ΣΥΧΝΟΥ', 'ΣΥΧΝΟΥΣ', 'ΣΥΧΝΩΝ', 'ΣΥΧΝΩΣ', 'ΣΧΕ∆ΟΝ', 'ΣΩΣΤΑ', 'ΤΑ', 'ΤΑ∆Ε', 'ΤΑΥΤΑ', 'ΤΑΥΤΕΣ', 'ΤΑΥΤΗ', 'ΤΑΥΤΗΝ', 'ΤΑΥΤΗΣ', 'ΤΑΥΤΟ', 'ΤΑΥΤΟΝ', 'ΤΑΥΤΟΣ', 'ΤΑΥΤΟΥ', 'ΤΑΥΤΩΝ', 'ΤΑΧΑ', 'ΤΑΧΑΤΕ', 'ΤΕΛΙΚΑ', 'ΤΕΛΙΚΩΣ', 'ΤΕΣ', 'ΤΕΤΟΙΑ', 'ΤΕΤΟΙΑΝ', 'ΤΕΤΟΙΑΣ', 'ΤΕΤΟΙΕΣ', 'ΤΕΤΟΙΟ', 'ΤΕΤΟΙΟΙ', 'ΤΕΤΟΙΟΝ', 'ΤΕΤΟΙΟΣ', 'ΤΕΤΟΙΟΥ', 'ΤΕΤΟΙΟΥΣ', 'ΤΕΤΟΙΩΝ', 'ΤΗ', 'ΤΗΝ', 'ΤΗΣ', 'ΤΙ', 'ΤΙΠΟΤΑ', 'ΤΙΠΟΤΕ', 'ΤΙΣ', 'ΤΟ', 'ΤΟΙ', 'ΤΟΝ', 'ΤΟΣ', 'ΤΟΣΑ', 'ΤΟΣΕΣ', 'ΤΟΣΗ', 'ΤΟΣΗΝ', 'ΤΟΣΗΣ', 'ΤΟΣΟ', 'ΤΟΣΟΙ', 'ΤΟΣΟΝ', 'ΤΟΣΟΣ', 'ΤΟΣΟΥ', 'ΤΟΣΟΥΣ', 'ΤΟΣΩΝ', 'ΤΟΤΕ', 'ΤΟΥ', 'ΤΟΥΛΑΧΙΣΤΟ', 'ΤΟΥΛΑΧΙΣΤΟΝ', 'ΤΟΥΣ', 'ΤΟΥΤΑ', 'ΤΟΥΤΕΣ', 'ΤΟΥΤΗ', 'ΤΟΥΤΗΝ', 'ΤΟΥΤΗΣ', 'ΤΟΥΤΟ', 'ΤΟΥΤΟΙ', 'ΤΟΥΤΟΙΣ', 'ΤΟΥΤΟΝ', 'ΤΟΥΤΟΣ', 'ΤΟΥΤΟΥ', 'ΤΟΥΤΟΥΣ', 'ΤΟΥΤΩΝ', 'ΤΥΧΟΝ', 'ΤΩΝ', 'ΤΩΡΑ', 'ΥΠ', 'ΥΠΕΡ', 'ΥΠΟ', 'ΥΠΟΨΗ', 'ΥΠΟΨΙΝ', 'ΥΣΤΕΡΑ', 'ΦΕΤΟΣ', 'ΧΑΜΗΛΑ', 'ΧΘΕΣ', 'ΧΤΕΣ', 'ΧΩΡΙΣ', 'ΧΩΡΙΣΤΑ', 'ΨΗΛΑ', 'Ω', 'ΩΡΑΙΑ', 'ΩΣ', 'ΩΣΑΝ', 'ΩΣΟΤΟΥ', 'ΩΣΠΟΥ', 'ΩΣΤΕ', 'ΩΣΤΟΣΟ', 'ΩΧ', 'απ', 'απο', 'γι', 'για', 'δι', 'δια', 'εις', 'εκ', 'ενα', 'εναν', 'ενας', 'ενος', 'εξ', 'επ', 'επι', 'καθ', 'και', 'κατ', 'κατα', 'με', 'μεσα', 'μια', 'μια', 'μια', 'μιαν', 'μιας', 'περι', 'σε', 'στα', 'στη', 'στην', 'στις', 'στο', 'στον', 'στους', 'τα', 'τη', 'την', 'της', 'τις', 'το', 'τον', 'του', 'τους', 'των', 'υπο']
stop.sort()
stopwords = []
for w in stop:
  stopwords.append(w.upper())
print(type(stopwords))
# for w in stopwords:
#   key = ' '+ w +' '
#   if key in test:
#     test = test.replace(key, ' ') 

li = []
for item in data:
    for report in item[1]:
        # print(type(report['title']), type(report['text']))
        res = report['title'] + ' ' + report['text']

        res = get_capitalized(res)
        patterns = ["\d*[.|/|-]\d*[.|/|-]\d*",
                    ' \d*Μ\. ',
                    '\d\d[.|/|-]\d\d',
                    " \d* ",
                    " . "]
        for pattern in patterns:
            all_matches = re.finditer(pattern, res)
            for match in all_matches:
                # print('-'*20)
                res = res.replace(match.group(), ' ')

        for w in stopwords:
            key = ' '+ w +' '
            if key in res:
                res = res.replace(key, ' ')

        li.append(res)

df = pd.DataFrame(li, columns=['Titles_and_Text'])


# Word Cloud for a visual representation of words

long_string = ','.join(list(df['Titles_and_Text'].values))

wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# Tokenization approach

stemmer = GreekStemmer()

# stop_words = stopwords.words('greek')

def is_greek(word):
    if word >= 'Α':
        return True
    return False


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    stemmed_words = []
    for doc in texts:
        li = []
        for word in simple_preprocess(str(doc)):
            if (is_greek(word) and word not in stopwords):
                 li.append(stemmer.stem(word.upper()))
        
        stemmed_words.append(li)
    return stemmed_words        
    # return [[stemmer.stem(word) for word in simple_preprocess(str(doc)) if (is_greek(word) and word not in stopwords2)] for doc in texts]

data = df.Titles_and_Text.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)



# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

id_words = [[(id2word[id], count) for id, count in line] for line in corpus]



lda_model = None

# number of topics
# for num_topics in range(2,21):
num_topics =8
# Build LDA model
temp_model = gensim.models.LdaMulticore(corpus=corpus,
                                      id2word=id2word,
                                      num_topics=num_topics)


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=temp_model, texts=data_words, dictionary=id2word, coherence='c_v')
temp_lda = coherence_model_lda.get_coherence()

if not lda_model or temp_lda > coherence_lda:
    lda_model = temp_model
    coherence_lda = temp_lda
    best_num_topics = num_topics
# elif temp_lda > coherence_lda:

print('num_topics: ', best_num_topics, '  Coherence Score: ', coherence_lda)





# Visualize the topics
# pyLDAvis.enable_notebook()
LDAvis_data_filepath = '/content/drive/My Drive/data_for_colab/police2/ldavis_prepared_'+str(best_num_topics)
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself

LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)




pyLDAvis.enable_notebook()    
# load the pre-prepared pyLDAvis data from disk
LDAvis_data_filepath = '/content/drive/My Drive/data_for_colab/police2/ldavis_prepared_'+str(8)
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, '/content/drive/My Drive/data_for_colab/police2/ldavis_prepared_'+ str(8) +'.html')
LDAvis_prepared

buckets_title = ['Συλληψεις', 'ΚΥΚΛΟΦΟΡΙΑΚΕΣ ΡΥΘΜΙΣΕΙΣ', 'Συγκεντρώσεις', 'Συλληψεις και ναρκωτικα', 'Συγκεντρώσεις', 'ΚΥΚΛΟΦΟΡΙΑΚΕΣ ΡΥΘΜΙΣΕΙΣ', 'Συλληψεις', 'Επεισόδια']

# Get the words per topic

topics = LDAvis_prepared.topic_info.Category.unique() 
df = LDAvis_prepared.topic_info

for t, my_title in zip(topics, buckets_title):
  words = df[df['Category'] == t]
  print('\nTopic: ', t, '  --  My title: ', my_title)
  print('Words: ')
  pprint(', '.join(words.Term[0:30]))


'''
Model prediction
'''

def get_tokensUpper(mylist):
    return [[i.upper() for i in item]for item in mylist]

mylist = get_tokensUpper(data_words)

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                      id2word=id2word,
                                      num_topics=8)
# Create a new corpus, made of previously unseen documents.

other_corpus = [id2word.doc2bow(text) for text in mylist]

# unseen_doc = other_corpus[0]
vector = lda_model[other_corpus]  # get topic probability distribution for a document


# Create map for only one topic

topic = 2

category = buckets_title[topic]
texts    = buckets[str(topic)]

tokens_dict = {}

tokens_per_month = defaultdict(int)
for text in texts:

    tokens = get_tokens_from_text(text)
    for k in tokens:
        if k in tokens_keep:
            if k in name_correction:
                tokens_per_month[name_correction[k]] += 1  
            else:
                tokens_per_month[k] += 1 
tokens_dict.update({'All months': tokens_per_month})

print('--- Find Tokens extraction Started ---')
tokens      = get_all_tokens(tokens_dict)