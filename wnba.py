#importing all the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



#read the csv file of the data and look at the first few data points
df = pd.read_csv("wnbadraft.csv")
df.head()

#Performing EDA

#Checking the shape of the data. The output is (1064, 14) meaning 1064 rows and 14 columns
df.shape

#Calling on .info to better understand data and corresponding data types. There are 7 floats, 3 integers, and 4 objects
print(df.info())

#Getting the statistical summary of the data through describe and transpose. After looking at the data, win_shares and win_shares_40 are problematic since the minimums are both negative
df.describe().T

#Converting the column names into a list to better access them later
columnNames = df.columns.tolist()

#Data Cleaning Processes 

#Checking for missing values. There is a lot of null values. 
(df.isnull().sum())

#Filling in the null values with appropriate fillers
df.fillna({"games":0}, inplace=True)
df.fillna({"win_shares":0}, inplace=True)
df.fillna({"win_shares_40":0}, inplace=True)
df.fillna({"minutes_played":0}, inplace=True)
df.fillna({"points":0}, inplace=True)
df.fillna({"total_rebounds":0}, inplace=True)
df.fillna({"assists":0}, inplace=True)
df.fillna({"former": "Unknown"}, inplace=True)
#print(df.isnull().sum())

#Finding missing data for colleges and seeing if I can find the players college
missingCollegePlayers = df.loc[df["college"].isna(), "player"].tolist()
print(missingCollegePlayers)

#Based on the research of the 86 players, it seems that a majority of players don't have colleges listed since they played overseas. 
df.fillna({"college": "International"}, inplace=True)
print(df.isnull().sum())

#Figuring out the missing players
missingPlayers= df.loc[df["player"].isna()]
print(missingPlayers)
df.loc[299, "player"] = "Tricia Liston"
df.loc[658, "player"] = "Jenni Dant"
print(df.loc[299, "player"], df.loc[658, "player"] )
print(df.isnull().sum())

#Checking for duplicate values
df.nunique()

#Performing univariate analysis
pointCounts = df['points'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(pointCounts.index, pointCounts, color='darkmagenta')
plt.title('Count Plot of Points')
plt.xlabel('Points')
plt.ylabel('Count')
plt.show()

totalReboundCounts = df['total_rebounds'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(totalReboundCounts.index, totalReboundCounts, color='darkred')
plt.title('Count Plot of Rebounds')
plt.xlabel('Rebounds')
plt.ylabel('Count')
plt.show()

assistCounts = df['assists'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(assistCounts.index, assistCounts, color='darkcyan')
plt.title('Count Plot of Assists')
plt.xlabel('Assists')
plt.ylabel('Count')
plt.show()

gameCounts = df['games'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(gameCounts.index, gameCounts, color='seagreen')
plt.title('Count Plot of Games')
plt.xlabel('Games')
plt.ylabel('Count')
plt.show()

winCounts = df['win_shares'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(winCounts.index, winCounts, color='lightsalmon')
plt.title('Count Plot of Wins')
plt.xlabel('Wins')
plt.ylabel('Count')
plt.show()

winShareCounts = df['win_shares_40'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(winShareCounts.index, winShareCounts, color='darkolivegreen')
plt.title('Count Plot of winShare')
plt.xlabel('WinShares')
plt.ylabel('Count')
plt.show()

sns.scatterplot(data=df, x="overall_pick", y="total_rebounds")
plt.title("Draft Pick vs Rebounds")
plt.show()
sns.scatterplot(data=df, x="overall_pick", y="games")
plt.title("Draft Pick vs Games Played")
plt.show()
sns.scatterplot(data=df, x="overall_pick", y="assists")
plt.title("Draft Pick vs Assists")
plt.show()
sns.scatterplot(data=df, x="overall_pick", y="minutes_played")
plt.title("Draft Pick vs Minutes Played")
plt.show()
sns.scatterplot(data=df, x="overall_pick", y="points")
plt.title("Draft Pick vs Points Scored")
plt.show()

#Kernel density plot

sns.set_style("darkgrid")

numericalColumns = df.select_dtypes(include=["int64", "float64"]).columns

plt.figure(figsize=(14, len(numericalColumns) * 3))
for index, feature in enumerate(numericalColumns, 1):
    plt.subplot(len(numericalColumns), 2, index)
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")

plt.tight_layout()
plt.show()

#Bivariate Analysis

sns.set_palette("Pastel1")
plt.figure(figsize=(10, 6))
sns.pairplot(df)
plt.suptitle('Pair Plot for DataFrame')
plt.show()


#Correlation Analysis 
success_metrics = ["games", "points", "total_rebounds", "assists", "minutes_played"]
print(df[["overall_pick"] + success_metrics].corr())

df["pick_group"] = pd.qcut(df["overall_pick"], 4, labels=["Top", "Upper Mid", "Lower Mid", "Late"])
df.groupby("pick_group", observed=False)[success_metrics].mean()

sns.boxplot(data=df, x="pick_group", y="points")

#Calculating the success score of the picks
scaler = MinMaxScaler()
df["successScore"] = scaler.fit_transform(df[success_metrics]).mean(axis=1)

#Looking at the success score outcomes
df['successScore'].describe()
df['successScore'].hist(bins=20)

#Finding the correlation between overall pick and success score
df[['overall_pick', 'successScore']].corr()

#Looking at the trends between the variables
cap = df["successScore"].quantile(0.99)
df_plot = df[df["successScore"] <= cap]

plt.figure(figsize=(10,6))
sns.regplot(data=df_plot, x="overall_pick", y="successScore", scatter_kws={'alpha':0.6})
plt.title("Draft Pick vs. Success Score")
plt.xlabel("Overall Pick Number")
plt.ylabel("Success Score")
plt.show()

#Creating a correlation heatmap

metrics = ["games", "points", "total_rebounds", "assists", "successScore"]
corr = df[metrics].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Heatmap of Success Metrics")
plt.show()

#Creating a dataframe with the necessary columns to use for PowerBI
finalDf = df[[
    "player", 
    "year", 
    "overall_pick", 
    "games", 
    "points", 
    "total_rebounds", 
    "assists", 
    "successScore"
]]

finalDf.to_csv("wnbaDraftSuccess.csv", index=False)
