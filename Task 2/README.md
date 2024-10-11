# IMDb Movie Rating Analysis (India)

This project involves analyzing a dataset of Indian movies from IMDb, with a focus on exploring different aspects of the data such as movie release trends, genre popularity, top actors and directors, and the relationship between movie duration, ratings, and votes. The project employs extensive data cleaning, visualization, and interactive features to present insights from the dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Steps Involved](#steps-involved)
- [Visualizations](#visualizations)
- [Dependencies](#dependencies)
- [License](#license)

## Project Overview
This project is a data analysis of Indian movies based on the IMDb dataset. It focuses on:
- Cleaning and preprocessing the data to remove missing and duplicate records.
- Exploring various insights such as the number of movies released by year, trends in genres, top actors and directors, and their popularity over time.
- Visualizing the relationship between key metrics such as movie duration, IMDb ratings, and the number of votes.
- Implementing interactive plots using sliders for exploring top actors and directors over the years.

## Dataset Information
- The dataset contains information on Indian movies, including fields like `Name`, `Year`, `Genre`, `Rating`, `Votes`, `Duration`, `Director`, and `Actors`.
- The dataset was sourced from [IMDb India Movies dataset](https://www.kaggle.com/datasets/).

## Steps Involved
1. **Data Cleaning**: Removing rows with missing values and duplicates, handling incorrect formats for the `Year` and `Duration` columns, and dropping irrelevant records.
2. **Data Exploration**: Analyzing movie releases by year, genre trends, and identifying top actors and directors.
3. **Interactive Visualizations**: Using sliders to visualize the top actors and directors through the years.
4. **3D Visualization**: A 3D scatter plot showcasing the relationship between movie `Duration`, `Rating`, and `Votes`.

## Visualizations
Some of the key visualizations include:
- **Movies Released by Year**: A bar chart showing the count of movies released each year.
- **Genre Popularity Over Time**: A line plot that shows the trend of different movie genres over the years.
- **Top 20 Actors and Directors**: Bar charts showing the most prolific actors and directors.
- **3D Scatter Plot**: A 3D plot that visualizes the relationship between movie `Duration`, `Rating`, and `Votes`.
- **Interactive Plots**: Sliders allow users to explore the top actors and directors by year.

## Dependencies
To run the project, you need to install the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `mpl_toolkits`
- `matplotlib.widgets`

## License
This project is licensed under the MIT License.