# openmeteo_downloader.py
import openmeteo_requests
import numpy as np
import requests_cache
import pandas as pd
from retry_requests import retry
import re
from datetime import datetime

class OpenMeteoWeatherDownloader:
    def __init__(self, location, latitude, longitude,
                 start_date="2024-12-01", end_date="2025-02-03",
                 cache_expire=3600, csv_filename="weather_data.csv",
                 ini_filename="location.ini"):
        """
        Initialize the downloader with location info, date range, cache expiry, and file names.

        :param location: Name of the location (string)
        :param latitude: Latitude coordinate (float)
        :param longitude: Longitude coordinate (float)
        :param start_date: Start date for the forecast (YYYY-MM-DD)
        :param end_date: End date for the forecast (YYYY-MM-DD)
        :param cache_expire: Cache expiry time in seconds (default 3600)
        :param csv_filename: Filename for the output CSV
        :param ini_filename: Filename for the output INI file
        """
        self.location = location
        self.latitude = latitude
        self.longitude = longitude
        self.start_date = start_date
        self.end_date = end_date
        self.csv_filename = csv_filename
        self.ini_filename = ini_filename

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=cache_expire)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

    def fetch_and_process(self):
        """
        Fetch the weather data from the Open-Meteo API, process it,
        export a CSV file, and create an INI file with the location information.
        """
        # Define API URL and parameters
        
        historical_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        forcast_url =  "https://api.open-meteo.com/v1/forecast"

        historical_params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "hourly": [
                "wind_speed_10m", "soil_temperature_0cm",
                "relative_humidity_2m", "vapour_pressure_deficit"
            ],
            "daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum", "shortwave_radiation_sum"]
        }

        forecast_params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            #"forecast_days": difference,
            "hourly": [
                "wind_speed_10m", "soil_temperature_0cm",
                "relative_humidity_2m", "vapour_pressure_deficit"
            ],
            "daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum", "shortwave_radiation_sum"]
        }

        # Check if the end date is a fuure date -> Then have to call the forecast api
        end_date = self.end_date
        start_date = self.start_date
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

        # Get the current date
        current_date = datetime.today().date()

        # Calculate the difference in days
        urls = []
        params = []
        
        # only historical
        if end_date <= current_date:
            # print("Only Historical Data")
            urls = [historical_url]
            params = [historical_params]
            historical_df = pd.DataFrame()
            dataframes_array = [historical_df]
        # only forecast
        elif start_date >= current_date:
            # print("Only Forecast")
            difference = (end_date - current_date).days
            if difference > 14:
                difference = 14
            forecast_params["forecast_days"] = difference + 1
            urls = [forcast_url]
            params = [forecast_params]
            forcast_df = pd.DataFrame()
            dataframes_array = [forcast_df]
        # historical + forecast
        else:
            difference = (end_date - current_date).days
            if difference > 14:
                difference = 14
            forecast_params["forecast_days"] = difference + 1

            self.end_date = datetime.now()
            self.end_date = self.end_date.strftime(("%Y-%m-%d"))
            # print(self.end_date)
            historical_params["end_date"] = self.end_date

            # print("HISTORICAL + FORECAST\n")
            # print(f"Days difference: {difference}")
            urls = [historical_url, forcast_url]
            params = [historical_params, forecast_params]

            historical_df = pd.DataFrame()
            forcast_df = pd.DataFrame()
            dataframes_array = [historical_df, forcast_df]


        for i in range(len(urls)):
            # Request weather data (assuming the API returns a list of responses)
            responses = self.openmeteo.weather_api(urls[i], params=params[i])
            response = responses[0]  # Process first location. Use a loop if needed.

            # --------------------
            # Process Hourly Data
            # --------------------
            hourly = response.Hourly()
            # The order of variables should match the order requested above.
            hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()
            hourly_soil_temperature_0cm = hourly.Variables(1).ValuesAsNumpy()
            #hourly_direct_radiation = hourly.Variables(2).ValuesAsNumpy()
            hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
            hourly_vapour_pressure_deficit = hourly.Variables(3).ValuesAsNumpy()
            

            num_values = len(hourly_relative_humidity_2m)
            hourly_dates = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                periods=num_values,  # Use the number of data points
                freq=pd.Timedelta(seconds=hourly.Interval())
            )

            hourly_data = {
                "date": list(hourly_dates),
                "rhmean": hourly_relative_humidity_2m,
                "windspeed": hourly_wind_speed_10m,
                "soilt": hourly_soil_temperature_0cm,
                "vp_deficit": hourly_vapour_pressure_deficit
            }
            hourly_dataframe = pd.DataFrame(data=hourly_data)

            # Resample Hourly Data to Daily Data
            hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date'])
            hourly_dataframe.set_index('date', inplace=True)

            daily_hourly_dataframe = hourly_dataframe.resample('D').agg({
                'rhmean': 'mean',
                'windspeed': 'mean',
                'soilt': 'mean',
                'vp_deficit': 'mean'
            }).reset_index()

            # --------------------
            # Process Daily Data
            # --------------------
            daily = response.Daily()
            daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
            daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
            daily_rain_sum = daily.Variables(2).ValuesAsNumpy()
            daily_mean_temperature = (daily_temperature_2m_max + daily_temperature_2m_min) / 2
            daily_mean_radiation = daily.Variables(3).ValuesAsNumpy() #* 0.0036

            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                ),
                "maxt": daily_temperature_2m_max,
                "mint": daily_temperature_2m_min,
                "rain": daily_rain_sum,
                "meant": daily_mean_temperature,
                "radn":daily_mean_radiation
            }
            daily_dataframe = pd.DataFrame(data=daily_data)

            # Merge daily data from hourly aggregation with the daily data
            merged_df = pd.merge(daily_dataframe, daily_hourly_dataframe, on="date", how="outer")
            merged_df['date'] = pd.to_datetime(merged_df['date'])

            # Convert the date into year and cumulative day-of-year
            merged_df['year'] = merged_df['date'].dt.year
            merged_df['day'] = merged_df['date'].dt.dayofyear
            merged_df.drop(columns=['date'], inplace=True)

            # Calculate saturation vapor pressure (vp_sat) in kPa
            merged_df['vp_sat_kPa'] = 0.61078 * np.exp((17.27 * merged_df['meant']) / (merged_df['meant'] + 237.3))
            # Convert vp_sat to hPa
            merged_df['vp_sat_hPa'] = merged_df['vp_sat_kPa'] * 10
            # Convert vp_deficit from kPa to hPa (if your vp_deficit is in kPa)
            merged_df['vp_deficit_hPa'] = merged_df['vp_deficit'] * 10
            # Calculate actual vapor pressure (vp) in hPa
            merged_df['vp'] = merged_df['vp_sat_hPa'] - merged_df['vp_deficit_hPa']

            merged_df['vpd'] = merged_df["vp_deficit"]
            # Drop intermediate columns (optional)
            merged_df.drop(columns=['vp_sat_kPa', 'vp_sat_hPa', 'vp_deficit_hPa'], inplace=True)

            # Reorder columns as desired
            new_order = ['year', 'day', 'rain', 'maxt', 'mint', 'meant', 'soilt', 'rhmean', 'vp', 'vpd','windspeed', 'radn']

            dataframes_array[i] = merged_df[new_order]

        # delete the last element of the historical df because this day already exists in the forcast df
        if(len(urls) == 2):
            #print("Deleting duplicate")
            dataframes_array[0].drop(dataframes_array[0].index[-1], inplace=True)

        # merge the 2 dataframes
        #print(dataframes_array)
        final_df = pd.concat(dataframes_array, ignore_index=True)

        # Save the final DataFrame to CSV
        final_df.to_csv(self.csv_filename, index=False)
        #print("CSV file saved:", self.csv_filename)
        #print(final_df)

        # Write location constants to an INI file
        with open(self.ini_filename, "w") as file:
            file.write(f"location = {self.location}\n")
            file.write(f"latitude = {self.latitude}\n")
            file.write(f"longitude = {self.longitude}\n")

        #return final_df
        # Return the total amount of rain
        # Filter the DataFrame to include only rows from the current date to the end date
        current_date = datetime.today().date()
        filtered_df = final_df.copy()
        filtered_df['date'] = pd.to_datetime(filtered_df['year'], format='%Y') + pd.to_timedelta(filtered_df['day'] - 1, unit='D')
        filtered_df = filtered_df[(filtered_df['date'] >= pd.Timestamp(current_date)) & (filtered_df['date'] <= pd.Timestamp(end_date))]
        
        # Extract year, month, day, and rain columns
        filtered_df['month'] = filtered_df['date'].dt.month
        filtered_df['day'] = filtered_df['date'].dt.day
        result_df = filtered_df[['year', 'month', 'day', 'rain']]

        #print(result_df.to_string(index=False))
        return final_df['rain'].sum(), result_df.to_string(index=False)


# If you want to test the class directly, you can add the following:
if __name__ == "__main__":
    downloader = OpenMeteoWeatherDownloader(
        location="Tylisos",
        latitude=35.341846,
        longitude=25.148254,
        start_date="2025-03-01",
        end_date="2025-04-15",
        csv_filename="Tylisos.csv",
        ini_filename="Tylisos.ini"
    )
    downloader.fetch_and_process()
    
    # Path to your file (change as needed)
    # file_path = "test_commands"
    
    # # Update the file with the new parameter values
    # downloader.update_command_params(file_path, updates)
    
    # print(f"Parameters in {file_path} have been updated.")
    
