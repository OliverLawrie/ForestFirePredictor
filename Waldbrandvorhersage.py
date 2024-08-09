#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Import necessary metrics
import seaborn as sns

#import dataset
df_fires = pd.read_csv("/Users/oliverlawrie/Desktop/ForestFirePredictor/WaldbrändeDE.csv")

# Initial exploration
print("Shape of the dataset:", df_fires.shape)
print("\nColumn names:")
print(df_fires.columns)
print("\nData types of each column:")
print(df_fires.dtypes)
print("\nFirst few rows of the dataset:")
print(df_fires.head())
print("\nSummary statistics:")
print(df_fires.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df_fires.isnull().sum())

# Visualise summary of each region to see point of interest for analysis
region_summary = df_fires.groupby('admlvl1').agg(
    number_of_fires=('id', 'count'),
    total_area_burnt=('area_ha', 'sum'),
    average_area_burnt=('area_ha', 'mean')
).reset_index()

# Display the summary
print(region_summary)

#Linear Reg Model to Predict Temp
#IMPORT TEXT FILE
file_path = '/Users/oliverlawrie/Desktop/ForestFirePredictor/regional_averages_tm_year.txt'
df = pd.read_csv(file_path, delimiter=';')  # Use the appropriate delimiter

# Define your features and target variable
X = df[['Jahr']]
y = df['Niedersachsen']  # Column that shows temperature

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Check split
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

#Fit and Train Model
regressor = LinearRegression() #Create Linear Regression model
regressor.fit(X_train, y_train) # fit the model with training data

# predict with test
y_predictions = regressor.predict(X_test)
print('Predictions:', y_predictions)

# get the coefficients and intercept
print("Coefficients:\n", regressor.coef_)
print('Intercept:\n', regressor.intercept_)

# COMPARING TEST DATA AND PREDICTED DATA
comparison_df = pd.DataFrame({"Actual":y_test,"Predicted":y_predictions})
print('Actual test data vs predicted: \n', comparison_df)

# EVALUATING MODEL METRICS
print('MAE:', mean_absolute_error(y_test,y_predictions))
print("MSE",mean_squared_error(y_test,y_predictions))
print("RMSE",np.sqrt(mean_squared_error(y_test,y_predictions)))
r2 = r2_score(y_test,y_predictions)
print('Model Score: ', r2)

# Plot LINEAR REGRESSION LINE
sns.regplot(x='Jahr', y='Niedersachsen', data=df, ci=None,
            scatter_kws={'s':100, 'facecolor':'red'})


# Predict future temperatures
future_years = pd.DataFrame({'Jahr': np.arange(2024, 2050)})
future_temperatures = regressor.predict(future_years)
future_years['Temperaturvorhersage'] = future_temperatures

print(future_years)

# Plot historical data and predictions
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Historical Data')
plt.plot(future_years['Jahr'], future_years['Temperaturvorhersage'], color='red', linestyle='--', label='Predictions')
plt.xlabel('Jahr')
plt.ylabel('Temperaturvorhersage')
plt.legend()
plt.title('Temperaturvorhersage in Niedersachsen bis 2050')
plt.show()

#With average future temperatures, create model to predict average area burnt in niedersachsen per year given the temperature.

# Evaluate future risk of Niedersachsen in coming years with forest fire data and predicted average temperatures

### Create model that uses temp and damage to see relationship between two, and then feed in pred temps from above to this model.

#Create df_total_damage with temperatures and total area burnt in Niedersachsen

#Create df with years and total area burned in niedersachsen
df_total_damage = df_fires[df_fires['admlvl1'] == 'Niedersachsen']
df_total_damage['Fire_Year'] = df_total_damage['initialdate'].str[:4]
df_total_damage=df_total_damage[['Fire_Year', 'area_ha']]
df_total_damage

#Calculate total area burned per year for use in model
df_total_damage = df_total_damage.groupby('Fire_Year')['area_ha'].sum().reset_index()

# Display the aggregated dataframe and visualise in barchart
print(df_total_damage)
df_total_damage.columns = ['Fire_Year', 'Total_Area_Burned'] #rename columns for clarity
df_total_damage[['Fire_Year', 'Total_Area_Burned']].plot(x='Fire_Year', y='Total_Area_Burned', kind='bar')

#Add the years between that had 0 hectares burnt

# Create a DataFrame with all years from 2013 to 2023
years = pd.DataFrame({'Fire_Year': [str(year) for year in range(2013, 2024)]})

# Merge with the aggregated data
df_total_damage = years.merge(df_total_damage, on='Fire_Year', how='left')

# Fill NaN values with 0
df_total_damage['Total_Area_Burned'] = df_total_damage['Total_Area_Burned'].fillna(0)

# Display the updated dataframe
print(df_total_damage)

# Plot the total area burned per year
df_total_damage.plot(x='Fire_Year', y='Total_Area_Burned', kind='bar')
plt.xlabel('Year')
plt.ylabel('Total Area Burned (ha)')
plt.title('Total Area Burned per Year in Niedersachsen')
plt.show()

#Create df to find temperatures between 2013-2023 in Niedersachsen
results_df = df[(df['Jahr'] >= 2013) & (df['Jahr'] <= 2023)][['Jahr', 'Niedersachsen']]

# Convert 'Jahr' in result_df to string if it's an integer
result_df['Jahr'] = result_df['Jahr'].astype(str)

# Merge the DataFrames on the 'Jahr' (Year) column
df_merged = pd.merge(result_df, df_total_damage, left_on='Jahr', right_on='Fire_Year')

# Select relevant columns for the model
df_model = df_merged[['Niedersachsen', 'Total_Area_Burned']]

# Display the final DataFrame
print(df_model)

# Define features (X) and target variable (y)
X = df_model[['Niedersachsen']]  # Feature: Temperature in Niedersachsen
y = df_model['Total_Area_Burned']  # Target: Total Area Burned

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Display the model's coefficients
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Assuming the future_years DataFrame already contains the predicted temperatures in 'Temperaturvorhersage'
# Rename the column to match the training data column name 'Niedersachsen'
future_years = future_years.rename(columns={'Temperaturvorhersage': 'Niedersachsen'})

# Use the trained model to predict the total area burned
future_years['Predicted_Total_Area_Burned'] = model.predict(future_years[['Niedersachsen']])

# Display the predictions
future_years[['Jahr', 'Niedersachsen', 'Predicted_Total_Area_Burned']]
print(future_years)


# Create a scatter plot where color indicates temperature
plt.figure(figsize=(12, 8))

# Normalize temperatures for better color mapping
norm = plt.Normalize(future_years['Niedersachsen'].min(), future_years['Niedersachsen'].max())

# Scatter plot for predicted total area burned with color representing temperature
sc = plt.scatter(future_years['Jahr'], future_years['Predicted_Total_Area_Burned'],
                 c=future_years['Niedersachsen'], cmap='viridis', norm=norm, s=100)

# Add color bar to indicate temperature scale
cbar = plt.colorbar(sc)
cbar.set_label('Temperature in Niedersachsen (°C)')

# Labels and title
plt.xlabel('Year')
plt.ylabel('Predicted Total Area Burned (ha)')
plt.title('Predicted Total Area Burned in Niedersachsen with Corresponding Temperatures')

# Show plot
plt.show()