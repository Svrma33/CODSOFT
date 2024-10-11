# Importing necessary libraries
import pandas as pd 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib import rcParams
import warnings

#Ignore all warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('IMDb-Movies-India.csv', encoding='ISO-8859-1')

# Cleaning the Data
print("Initial Data Shape:", df.shape)

# Finding rows with missing values in certain columns
null_rows = df[df.iloc[:, 1:9].isna().apply(lambda x: all(x), axis=1)]
print("Below are the Missing values for each column from 1 to 9:")
print(null_rows.head())

# Removing rows where all columns from 1 to 9 are missing
df = df[~df.iloc[:, 1:9].isna().apply(lambda x: all(x), axis=1)]
print("Shape after removing rows with missing values:", df.shape)
print()
# Identifying duplicate rows based on 'Name' and 'Year'
duplicate = df[df.duplicated(subset=['Name', 'Year'], keep=False)]
print("Below are the duplicate rows according to Name and Year:")
print(duplicate.head())

# Removing duplicate rows
df.drop_duplicates(subset=['Name', 'Year'], inplace=True)
print("Shape after removing duplicates:", df.shape)
print()
# Removing rows with missing values in certain columns
null_rows = df[df.iloc[:, [1, 2, 4, 5]].isna().apply(lambda x: all(x), axis=1)]
print("Below are the Missing values for each column from 1 to 5 excluding Genre:")
print(null_rows.head())

df = df[~df.iloc[:, [1, 2, 4, 5]].isna().apply(lambda x: all(x), axis=1)]
print("Shape after removing more rows with missing values:", df.shape)

# Removing rows where Year is 2022
df.drop(df.loc[df['Year'] == 2022].index, inplace=True)
print("Shape after removing rows with Year 2022:", df.shape)

# Number of Movies by Year
year_count = df.groupby('Year').size().reset_index(name='Count')
plt.figure(figsize=(15, 6))
plt.bar(year_count['Year'], year_count['Count'], color='purple')
plt.title('Number of Movies by Year of Launch')
plt.xlabel('Year of Movie Release')
plt.ylabel('Count of Movies Released')
plt.xticks(rotation=90, fontsize=8, ha='right')
plt.tight_layout()

manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 950), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
plt.show()

# Genre
dummies = df['Genre'].str.get_dummies(sep=', ')
df_genre = pd.concat([df, dummies], axis=1).replace(0, np.nan)
df_genre_count = df_genre.drop(['Name', 'Duration', 'Genre', 'Rating', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], axis=1)
df_genre_count = df_genre_count.groupby('Year').sum()
df_genre_count.reset_index(level=0, inplace=True)

# Plotting different genres across years
genres = df_genre_count.columns[1:]  # excluding 'Year'
plt.figure(figsize=(15, 6))
for genre in genres:
    plt.plot(df_genre_count['Year'], df_genre_count[genre], label=genre)
plt.title('Genre Through the Years')
plt.xticks(rotation=90, fontsize=8, ha='right')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()

manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 950), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
plt.show()

# Cleaning the 'Year' and 'Duration' columns
df.dropna(subset=['Year'], inplace=True)
df['Year'] = df['Year'].str.extract(r'(\d{4})')  # Extract numeric year
df.dropna(subset=['Year'], inplace=True)
df['Year'] = df['Year'].astype(int)
df['Duration'] = df['Duration'].str.replace(r' min', '')

# Top 20 Actors by Number of Movies
Actor1 = df[['Actor 1', 'Year']].rename(columns={'Actor 1': 'Actor'}, inplace=False)
Actor2 = df[['Actor 2', 'Year']].rename(columns={'Actor 2': 'Actor'}, inplace=False)
Actor3 = df[['Actor 3', 'Year']].rename(columns={'Actor 3': 'Actor'}, inplace=False)
Actor_Year = pd.concat([Actor1, Actor2, Actor3], ignore_index=True).dropna()
Actor_Year['Count'] = 1
Actor_Top = Actor_Year['Actor'].value_counts().rename_axis('Actor').reset_index(name='Count')

plt.figure(figsize=(10, 6))
sns.barplot(data=Actor_Top[0:20], x='Count', y='Actor', color='darkorange')
plt.title('Top 20 Actors by Number of Movies Made')
plt.tight_layout()

manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 600), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
plt.show()

# Top 20 Actors by Number of Movies Made Through the Years
Actor1 = df[['Actor 1', 'Year']].rename(columns={'Actor 1': 'Actor'}, inplace=False)
Actor2 = df[['Actor 2', 'Year']].rename(columns={'Actor 2': 'Actor'}, inplace=False)
Actor3 = df[['Actor 3', 'Year']].rename(columns={'Actor 3': 'Actor'}, inplace=False)
Actor_Year = pd.concat([Actor1, Actor2, Actor3], ignore_index=True).dropna()
Actor_Year['Count'] = 1
Actor_Top = Actor_Year.groupby(['Actor', 'Year']).size().reset_index(name='Count')

# Filtering top 20 actors overall
Top_20_Actors = Actor_Year['Actor'].value_counts().nlargest(20).index
Top_20_Actors_Data = Actor_Year[Actor_Year['Actor'].isin(Top_20_Actors)]

# Function to update plot based on year range for actors
def update(val):
    start_year = int(slider.val)
    filtered_data = Top_20_Actors_Data[Top_20_Actors_Data['Year'] == start_year]
    
    ax.clear()
    sns.countplot(data=filtered_data, x='Year', hue='Actor', ax=ax, edgecolor='white')
    ax.set_title('Top 20 Actors by Number of Movies Made in Year {}'.format(start_year))
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)  # Legend at top center
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    plt.draw()

# Initial plot setup for actors
fig, ax = plt.subplots(figsize=(15, 6))
sns.countplot(data=Top_20_Actors_Data, x='Year', hue='Actor', ax=ax)
plt.xticks(rotation=90, fontsize=8)
plt.title('Top 20 Actors by Number of Movies Made Through the Years')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)  # Legend at top center
plt.subplots_adjust(bottom=0.25)

# Define slider axis and create slider for actors
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Year', Top_20_Actors_Data['Year'].min(), Top_20_Actors_Data['Year'].max(), 
                valinit=Top_20_Actors_Data['Year'].min(), valstep=1)

# Update the plot when the slider value changes
slider.on_changed(update)

# Add keyboard event handling for actors
def on_key(event):
    if event.key == 'right':
        slider.set_val(min(slider.val + 1, slider.valmax))
    elif event.key == 'left':
        slider.set_val(max(slider.val - 1, slider.valmin))

# Connect the key event to the figure for actors
fig.canvas.mpl_connect('key_press_event', on_key)

# Display plot for actors
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 950), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
plt.show()

# Top 20 Directors by Number of Movies
Director_Year = df[['Director', 'Year']].dropna()
Director_Year['Count'] = 1
Director_Top = Director_Year.groupby('Director').size().reset_index(name='Count').sort_values(by='Count', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=Director_Top.head(20), x='Count', y='Director', color='green')
plt.title('Top 20 Directors by Number of Movies Made')
plt.tight_layout()

manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 600), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
plt.show()

# Top 20 Directors by Number of Movies Made Through the Years
Director_Year = df[['Director', 'Year']].dropna()
Director_Year['Count'] = 1

# Grouping by Director and Year
Director_Top_Year = Director_Year.groupby(['Director', 'Year']).size().reset_index(name='Count')

# Filtering top 20 directors overall
Top_20_Directors = Director_Top_Year.groupby('Director')['Count'].sum().nlargest(20).index
Top_20_Directors_Data = Director_Top_Year[Director_Top_Year['Director'].isin(Top_20_Directors)]

# Function to update plot based on year range for directors
def update_director(val):
    start_year = int(slider_director.val)
    filtered_data = Top_20_Directors_Data[Top_20_Directors_Data['Year'] == start_year]
    
    ax_director.clear()
    sns.countplot(data=filtered_data, x='Year', hue='Director', ax=ax_director, edgecolor='white')
    ax_director.set_title('Top 20 Directors by Number of Movies Made in Year {}'.format(start_year))
    
    ax_director.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)  # Legend at top center
    ax_director.tick_params(axis='x', rotation=90, labelsize=8)
    plt.draw()

# Initial plot setup for directors
fig_director, ax_director = plt.subplots(figsize=(15, 6))
sns.countplot(data=Top_20_Directors_Data, x='Year', hue='Director', ax=ax_director)
plt.xticks(rotation=90, fontsize=8)
plt.title('Top 20 Directors by Number of Movies Made Through the Years')

# Adjust hue (legend) position without changing graph elements
ax_director.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)  # Legend at top center

# Add space for the slider
plt.subplots_adjust(bottom=0.25)

# Define slider axis and create slider for directors
ax_slider_director = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_director = Slider(ax_slider_director, 'Year', Top_20_Directors_Data['Year'].min(), 
                         Top_20_Directors_Data['Year'].max(), valinit=Top_20_Directors_Data['Year'].min(), valstep=1)

# Update the plot when the slider value changes
slider_director.on_changed(update_director)

# Add keyboard event handling for directors
def on_key_director(event):
    if event.key == 'right':
        slider_director.set_val(min(slider_director.val + 1, slider_director.valmax))
    elif event.key == 'left':
        slider_director.set_val(max(slider_director.val - 1, slider_director.valmin))

# Connect the key event to the figure for directors
fig_director.canvas.mpl_connect('key_press_event', on_key_director)

# Display plot for directors
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 950), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
plt.show()

# 3D Scatter Plot of Duration, Rating, and Votes
dur_rat = df[['Duration', 'Rating', 'Votes']].dropna()
dur_rat['Duration'] = dur_rat['Duration'].astype('int64')
dur_rat['Votes'] = pd.to_numeric(dur_rat['Votes'], errors='coerce').fillna(0).astype(int)
dur_rat['Votes'] = dur_rat.Votes.replace(0, np.nan).dropna()
dur_rat = dur_rat.sort_values('Duration', ascending=True)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(dur_rat['Duration'], dur_rat['Rating'], dur_rat['Votes'], c=dur_rat['Rating'], cmap=plt.get_cmap('coolwarm'))
ax.set_xlabel('Duration')
ax.set_ylabel('Rating')
ax.set_zlabel('Votes')
plt.title('3D Plot of Duration, Rating and Votes')
plt.colorbar(sc)
plt.tight_layout()

manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 500), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
plt.show()

# Seaborn Pairplot
sns.pairplot(dur_rat)

manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 500), 
                                           int(manager.window.winfo_screenheight()/2 - 543)))  
plt.show()