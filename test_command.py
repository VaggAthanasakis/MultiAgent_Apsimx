import subprocess
import numpy as np
import requests_cache
import pandas as pd
from retry_requests import retry
import re
import sqlite3

def data_extraction_tool():
    """
    This tool is responsible for extracting data from a .db file.
    Can extract data like: total water applied.
    """
    # Path to your .db file
    db_path = r"C:\Users\vagga\Desktop\test_apsim_GUI\Python_Integration\APSIM_FILES\pears.db"
    #db_path = input_file
    # Connect to the database
    conn = sqlite3.connect(db_path)

    # Specify the table you want to read
    table_name = "Report"

    # Read the table into a Pandas DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # keep only the date and the ammount of waterApplied
    df = df[['Clock.Today', 'waterApplied']]

    # remove the time from the date
    df['Clock.Today'] = pd.to_datetime(df['Clock.Today']).dt.date

    total_water_applied = df['waterApplied'].sum()
    # Close the connection
    conn.close()

    print("\nTOTAL WATER: ",total_water_applied)
    return total_water_applied


apsim_exe = r"c:\Users\vagga\Desktop\test_apsim_GUI\program\APSIM2025.1.7644.0\bin\Models.exe"
commands_file = r"C:\Users\vagga\Desktop\test_apsim_GUI\Python_Integration\APSIM_FILES\pear_commands"

subprocess.run([apsim_exe, ' ','--apply', commands_file], check=True)


data_extraction_tool()