import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

datafile = pd.read_csv('data.csv')
datafile.info()

print(datafile.describe())
print(datafile.columns)
sb.pairplot(datafile)
plt.show()
# sb.heatmap(datafile.corr(), annot=True)
# plt.show()

plt.subplot(1, 3, 1)
sb.histplot(datafile['sqft_living'], kde=True)
plt.xlabel('Square Footage')
plt.title('Histogram of Square Footage')

plt.subplot(1, 3, 2)
sb.histplot(datafile['bedrooms'], kde=True)
plt.xlabel('Number of Bedrooms')
plt.title('Histogram of Bedrooms')

plt.subplot(1, 3, 3)
sb.histplot(datafile['bathrooms'], kde=True)
plt.xlabel('Number of Bathrooms')
plt.title('Histogram of Bathrooms')

plt.tight_layout()
plt.show()

X = datafile[['sqft_living', 'bedrooms', 'bathrooms']]
y = datafile['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_datafile = X_train.join(y_train)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

plt.scatter(y_test, predictions)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()


#statistical measures
print('\nStatistical Measures:')
print('Mean Square Footage:', datafile['sqft_living'].mean())
print('Mean Bedrooms:', datafile['bedrooms'].mean())
print('Mean Bathrooms:', datafile['bathrooms'].mean())
print('Mean Price:', datafile['price'].mean())
print('\nStandard Deviation Square Footage:', datafile['sqft_living'].std())
print('Standard Deviation Bedrooms:', datafile['bedrooms'].std())
print('Standard Deviation Bathrooms:', datafile['bathrooms'].std())
print('Standard Deviation Price:', datafile['price'].std())
print('\nCorrelation between features and price:')
print(datafile[['sqft_living', 'bedrooms', 'bathrooms', 'price']].corr())


def predict_house_price(square_footage, bedrooms, bathrooms, city):
    city_datafile = datafile[datafile['city'] == city]
    X_city = city_datafile[['sqft_living', 'bedrooms', 'bathrooms']]
    y_city = city_datafile['price']
    
    model_city = LinearRegression()
    model_city.fit(X_city, y_city)
    
    predicted_price = model_city.predict([[square_footage, bedrooms, bathrooms]])
    return predicted_price[0]


#cty_name=input("Enter city name to predict house price in a particular city(name as per dataset): ")
# predict house price in a particular city
predicted_price_city = predict_house_price(1930, 3.0, 2.0, "Kent")
print('Predicted Price in Kent:', predicted_price_city)
