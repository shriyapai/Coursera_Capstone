#!/usr/bin/env python
# coding: utf-8

# # Part 1

# ### In the following code cell, we import the necessary libraries question 1 in the assignment.

# In[116]:


#Import pandas and numpy
import pandas as pd
import numpy as np

# Import the library we use to open URLs
import urllib.request

#Enter the link to the Wikipedia page that we want to scrape
url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"

# Open the url using urllib.request and put the HTML into the page variable
page = urllib.request.urlopen(url)

# Import the BeautifulSoup library
from bs4 import BeautifulSoup


# ### In the following code cell, we parse the HTML from the URL into the BeautifulSoup parse format, and identify the table we need to work with.

# In[117]:


# Parse the HTML from our URL into the BeautifulSoup parse tree format
soup = BeautifulSoup(page, "lxml")

# Now we identify the table in the HTML and call it "all_tables"
all_tables=soup.find_all("table")
all_tables

# Our table is in the class "wikitable sortable", so we find that specifically
right_table=soup.find('table', class_='wikitable sortable')


# ### Now we get to the part where we take the data from the Wikipedia table and put it into a pandas dataframe format.

# In[118]:


A=[]
B=[]
C=[]

for row in right_table.findAll('tr'):
    cells=row.findAll('td')
    if len(cells)==3:
        A.append(cells[0].find(text=True))
        B.append(cells[1].find(text=True))
        C.append(cells[2].find(text=True))
        
df=pd.DataFrame(A,columns=['PostalCode'])
df['Boroughs']=B
df['Neighborhoods']=C
df


# ### Now we have to "clean" the table and process the individual cells. In the following code block, we get rid of rows where the boroughs are not assigned, and we replace neighborhoods that are not assigned with the name of the corresponding borough.

# In[119]:


# Remove rows where the borough is "Not Assigned"
df = df.loc[(df.Boroughs != "Not assigned")]

# Reset the indexing after getting rid of some rows
df=df.reset_index(drop=True)

# Just give a name to the column of boroughs to avoid the "Setting with copy" warning later
b = df['Boroughs']
b

# In this step, we replace the unassigned neighborhoods with the names of the corresponding borough
df.loc[(df.Neighborhoods == 'Not assigned\n'),'Neighborhoods'] = b
df.loc[(df.Neighborhoods == 'Not assigned'),'Neighborhoods'] = b
df


# ### In the following code block, we group the neighborhoods by postal code and have the neighborhoods separated with a comma.

# In[120]:


df = df.groupby(['PostalCode','Boroughs'])['Neighborhoods'].apply(', '.join).reset_index()
df


# ### We now print the number of rows in the dataframe using the .shape method.

# In[121]:


df.shape


# # Part 2

# ### Here we read the data from the CSV file and create a new dataframe containing the Postal Code and Latitude and Longitude data. We call this dataframe "LL_df".

# In[122]:


# Read data from .csv file into a new dataframe
LL_df = pd.read_csv("http://cocl.us/Geospatial_data")   

# Renaming "Postal Code" in the new dataframe to "PostalCode" as in our previous dataframe
LL_df.rename(columns = {'Postal Code':'PostalCode'}, inplace = True) 
LL_df


# ### In the following code block, we merge the two dataframes based on the "PostalCode" column to get our new dataframe (called "df_new") with five columns as shown below.

# In[123]:


# Merging the two dataframes "df" and "LL_df" based on the PostalCode column
df_new = pd.merge(df, LL_df, on="PostalCode")
df_new


# # Part 3

# ## In this part we will work on segmenting and clustering the neighborhoods in Toronto. We will work with ALL BOROUGHS in our dataframe, and not only boroughs that contain the word Toronto in them.

# ### We first download all the libraries we will need.

# In[15]:


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda update -n base -c defaults conda
get_ipython().system('conda install -c conda-forge geopy --yes ')
# convert an address into latitude and longitude values
from geopy.geocoders import Nominatim 

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # map rendering library

print('Libraries imported.')


# ### In order to define an instance of the geocoder, we need to define a user_agent. We will name our agent toronto_explorer, as shown below.

# In[124]:


address = 'Toronto, ON'

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# ### Now we create a map of Toronto with neighborhoods superimposed on top.

# In[126]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_new['Latitude'], df_new['Longitude'], df_new['Boroughs'], df_new['Neighborhoods']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# ### Next, we are going to use the Foursquare API to explore the neighborhoods and segment them. In the following code block, we define our Foursquare credentials and the version.

# In[127]:


CLIENT_ID = 'SBUVNNGTLE34CIU1PMYYILJEDBGOFO0SKLVAYX22H5HT32P1' # your Foursquare ID
CLIENT_SECRET = 'TLR14KBRNSDXT3HUYGBW2HD0HPJSHICF1UNGOYTHA5IEPXOO' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# ### In the code block below, we create a function to perform the following steps:
# ### (i) get the latitude and longitude of each neighborhood
# ### (ii) get the top 100 venues that are in each neighborhood within a radius of 500 meters
# ### (iii) send the GET request and examine the resutls
# ### (iv) clean the json and structure it into a pandas dataframe
# ### We apply the above process to all the neighborhoods in and around Toronto

# In[128]:


LIMIT = 100


# In[129]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
        
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# ### In the following code block, we run the above function on each neighborhood and create a new dataframe called toronto_venues.

# In[130]:


toronto_venues = getNearbyVenues(names=df_new['Neighborhoods'],
                                   latitudes=df_new['Latitude'],
                                   longitudes=df_new['Longitude']
                                  )


# ### Now we check the size of the resulting dataframe.

# In[131]:


print(toronto_venues.shape)
toronto_venues.head()


# ### Now we check how many venues were returned for each neighborhood.

# In[132]:


toronto_venues.groupby('Neighborhood').count()


# ### In the following code block, we find out how many unique categories can be curated from all the returned venues.

# In[133]:


print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# ## Here we analyze each neighborhood.

# In[134]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# ### Below, we examine the size of the new dataframe.

# In[135]:


toronto_onehot.shape


# ### In the code block below, we group rows by neighborhood and by taking the mean of the frequency of occurrence of each category.

# In[136]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# ### Below, we confirm the new size.

# In[137]:


toronto_grouped.shape


# ### Now we print each neighborhood along with the top 5 most common venues.

# In[138]:


num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# ### Next, we put that into a pandas dataframe. But first, we write a function to sort the venues in descending order.

# In[139]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# ### In the code block below, we create the new dataframe and display the top 10 venues for each neighborhood.

# In[140]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## Here we cluster the neighborhoods.
# ### In the following code block, we run k-means to cluster the neighborhood into 5 clusters.

# In[141]:


# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# ### Now we create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[142]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df_new

# merge toronto_grouped with df_new to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhoods')

toronto_merged.head() 


# ### In the following code block, we convert the entries in the "Cluster Labels" column to integer type from float.

# In[143]:


toronto_merged['Cluster Labels'] = toronto_merged['Cluster Labels'].fillna(0.0).astype(int)


# ### And finally, we visualize the resulting clusters using the code in the following code block.

# In[144]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=10)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhoods'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## In this part, we examine each cluster

# ### Cluster 1

# In[145]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 2

# In[146]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 3

# In[147]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 4

# In[148]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 5

# In[149]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[ ]:




