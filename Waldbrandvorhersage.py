from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Import necessary metrics
import seaborn as sns

#IMPORT TEXT FILE
file_path = '/content/drive/MyDrive/regional_averages_tm_year.txt'
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

#Create df with years and total area burned in niedersachsen
df_niedersachsen = df_fires[df_fires['admlvl1'] == 'Niedersachsen']
# type(df.iloc[0].initialdate)
df_niedersachsen['Fire_Year'] = df_niedersachsen['initialdate'].str[:4]
df_niedersachsen=df_niedersachsen[['Fire_Year', 'area_ha', 'id']]
df_niedersachsen

#Plot for easier visualisations
df_niedersachsen[['Fire_Year', 'area_ha']].plot(x='Fire_Year', y='area_ha', kind='scatter')

#Calculate total area burned per year for use in model
df_total_area_burned = df_niedersachsen.groupby('Fire_Year')['area_ha'].sum().reset_index()

# Rename the columns for clarity

# Display the aggregated dataframe and visualise in barchart
print(df_total_area_burned)
df_total_area_burned.columns = ['Fire_Year', 'Total_Area_Burned'] #rename columns for clarity
df_total_area_burned[['Fire_Year', 'Total_Area_Burned']].plot(x='Fire_Year', y='Total_Area_Burned', kind='bar')


#Add the years between that had 0 hectares burnt

# Create a DataFrame with all years from 2013 to 2023
years = pd.DataFrame({'Fire_Year': [str(year) for year in range(2013, 2024)]})

# Merge with the aggregated data
df_total_area_burned = years.merge(df_total_area_burned, on='Fire_Year', how='left')

# Fill NaN values with 0
df_total_area_burned['Total_Area_Burned'] = df_total_area_burned['Total_Area_Burned'].fillna(0)

# Display the updated dataframe
print(df_total_area_burned)

# Plot the total area burned per year
df_total_area_burned.plot(x='Fire_Year', y='Total_Area_Burned', kind='bar')
plt.xlabel('Year')
plt.ylabel('Total Area Burned (ha)')
plt.title('Total Area Burned per Year in Niedersachsen')
plt.show()

#Filter temperature dataset to show temps between 2013-2023 to train model
# Filtering the DataFrame for the years between 2013 and 2023 inclusive
filtered_df = df[(df['Jahr'] >= 2013) & (df['Jahr'] <= 2023)]

# Selecting only the 'Niedersachsen' and 'Jahr' columns
result_df = filtered_df[['Jahr', 'Niedersachsen']]

# Display the result
print(result_df)

print(df_total_area_burned)

# Convert 'Fire_Year' in df_total_area_burned to numeric
df_total_area_burned['Fire_Year'] = pd.to_numeric(df_total_area_burned['Fire_Year'])

# Merge DataFrames on year
merged_df = pd.merge(df_total_area_burned, result_df, left_on='Fire_Year', right_on='Jahr')

# Prepare features and target variable with unique names
feature_data = merged_df[['Niedersachsen']]  # Unique feature variable
target_data = merged_df['Total_Area_Burned']  # Unique target variable

# Split data into training and testing sets with unique names
feature_train, feature_test, target_train, target_test = train_test_split(
    feature_data, target_data, test_size=0.2, random_state=42
)

# Initialize and train the new regression model with unique variable names
new_model = LinearRegression()
new_model.fit(feature_train, target_train)

# Make predictions with unique variable names
target_pred = new_model.predict(feature_test)

# Evaluate the new model
mse_new = mean_squared_error(target_test, target_pred)
r2_new = r2_score(target_test, target_pred)

print(f"Mean Squared Error (New Model): {mse_new}")
print(f"R-squared (New Model): {r2_new}")

# Print coefficients
print(f"Coefficient (New Model): {new_model.coef_[0]}")
print(f"Intercept (New Model): {new_model.intercept_}")

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

#Use this new model to predict total area burnt with predicted temp values until 2050.
# Predict future forest damage
# Convert 'Temperaturvorhersage' to match the training feature name

# Ensure 'Temperaturvorhersage' is renamed to match the training feature name
future_years = future_years.rename(columns={'Temperaturvorhersage': 'Total_Area_Burned'})

# Use the second regression model (assuming it is named 'regressor')
future_damage = regressor.predict(future_years[['Jahr', 'Temperaturvorhersage']])

# Add the predictions to the DataFrame
future_years['Predicted_Damage'] = future_damage

# Display the updated DataFrame
print(future_years)

# Add the predictions to the DataFrame
future_years['Predicted_Damage'] = future_damage

# Display the updated DataFrame
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