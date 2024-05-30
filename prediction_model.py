import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the saved model from file
# with open("C:/Users/Kashaf/Desktop/random_forest_model.pkl", 'rb') as file:
#     loaded_model = pickle.load(file)

def loadModel_preprocessAllMaps():
    # Create a gender mapping dictionary
    gender_map = {'F': 0, 'M': 1}

    # Create a Sports mapping dictionary
    sports_list = ['Aeronautics', 'Alpinism', 'Archery', 'Art Competitions', 'Athletics', 'Badminton', 'Baseball', 'Basketball',
                   'Basque Pelota', 'Beach Volleyball', 'Boxing', 'Canoeing', 'Cricket', 'Croquet', 'Cycling', 'Diving',
                   'Equestrianism', 'Fencing', 'Figure Skating', 'Football', 'Golf', 'Gymnastics', 'Handball', 'Hockey',
                   'Ice Hockey', 'Jeu De Paume', 'Judo', 'Lacrosse', 'Modern Pentathlon', 'Motorboating', 'Polo', 'Racquets',
                   'Rhythmic Gymnastics', 'Roque', 'Rowing', 'Rugby', 'Rugby Sevens', 'Sailing', 'Shooting', 'Softball',
                   'Swimming', 'Synchronized Swimming', 'Table Tennis', 'Taekwondo', 'Tennis', 'Trampolining', 'Triathlon',
                   'Tug-Of-War', 'Volleyball', 'Water Polo', 'Weightlifting', 'Wrestling']

    sport_map = {sport: i for i, sport in enumerate(sports_list)}

    # Create a country mapping dictionary
    country_list = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Antigua', 'Argentina',
                    'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados',
                    'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Boliva', 'Bosnia and Herzegovina', 'Botswana',
                    'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde',
                    'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Cook Islands',
                    'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic', 'Democratic Republic of the Congo',
                    'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea',
                    'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana',
                    'Greece', 'Grenada', 'Guam', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary',
                    'Iceland', 'India', 'Individual Olympic Athletes', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy',
                    'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kosovo', 'Kuwait', 'Kyrgyzstan',
                    'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia',
                    'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico',
                    'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru',
                    'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Norway', 'Oman', 'Pakistan',
                    'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal',
                    'Puerto Rico', 'Qatar', 'Republic of Congo', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts', 'Saint Lucia',
                    'Saint Vincent', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles',
                    'Sierra Leone', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'South Sudan',
                    'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan',
                    'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad', 'Tunisia', 'Turkey', 'Turkmenistan', 'UK', 'USA',
                    'Uganda', 'Ukraine', 'United Arab Emirates', 'Unknown', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam',
                    'Virgin Islands, British', 'Virgin Islands, US', 'Yemen', 'Zambia', 'Zimbabwe']

    region_map = {region: i for i, region in enumerate(country_list)}

    maps = [gender_map, sport_map, region_map]
    return maps

def predict_athlete_will_win(gender, age, height, weight, sport, country):
    gender_map, sport_map, region_map = loadModel_preprocessAllMaps()

    # Convert string to numerical using map
    gender = gender_map[gender]
    sport = sport_map[sport]
    country = region_map[country]
    input_arr = [gender, age, height, weight, sport, country]
    res = loaded_model.predict([input_arr])

    return res

def predict_medal_counts(country_name, year):
    # Data Preparation
    country_dataset = pd.read_csv("Country_Medals_Formatted.csv")
    
    # Encode the 'Country_Name' column
    le = LabelEncoder()
    country_dataset['Country_Name'] = le.fit_transform(country_dataset['Country_Name'])

    # Split the data into features and target
    X = country_dataset[['Country_Name', 'Year']]
    y = country_dataset[['Gold', 'Silver', 'Bronze']]

    # Model Selection and Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Regressor
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Prediction
    X_pred = pd.DataFrame([[le.transform([country_name])[0], year]], columns=['Country_Name', 'Year'])
    prediction = model.predict(X_pred)

    # Extract Gold, Silver, Bronze values
    gold = round(prediction[0, 0])
    silver = round(prediction[0, 1])
    bronze = round(prediction[0, 2])

    return gold, silver, bronze