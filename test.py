import subprocess
import pandas as pd
import sqlite3
import warnings
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END , MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Literal
from typing_extensions import TypedDict
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from PIL import Image as PILImage
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime, time
import os
import re
from weather_data_retriever import OpenMeteoWeatherDownloader as openMeteoDataRetriever
import math
import logging
import configparser
import requests
import json
from pathlib import Path


def data_extraction_tool(crop: str):
    """
    This tool is responsible for extracting data given a .db file.
    Can extract data like total water applied.

    Args:
        crop: the crop that the simulation performed to. For example could be 'pear', 'olive', 'wheat', 'potato', 'corn', 'barley' etc.

    Returns:
        total_water_applied: The total amount of water applied
    """

    # Path to your .db file
    #logger.info("Inside Data Extraction Tool")
    print("\nInside Data Extraction Tool")
    print(f"\nEXTRACTION CROP: {crop}")

    crop = crop.capitalize()
    db_path = "/home/eathanasakis/Intership/MultiAgent_Apsimx/APSIM_FILES/Wheat.db"
    # Connect to the database
    conn = sqlite3.connect(db_path)

    # Specify the table you want to read
    table_name = "Report"

    # Read the table into a Pandas DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # keep only the date and the amount of waterApplied
    df = df[['Clock.Today', 'waterApplied']]
    
    # remove the time from the date
    df['Clock.Today'] = pd.to_datetime(df['Clock.Today']).dt.date

    # keep in a new dataphrame the rows that have a date greater than or equal to the current date
    current_date = datetime.now().date()
    df_curr_till_end = df[df['Clock.Today'] >= current_date]

    #print(f"\nDataFrame:\n{df_curr_till_end.to_string(index=False)}")

    total_water_applied = df['waterApplied'].sum()
    
    # Close the connection
    conn.close()

    #logger.info(f"Total Water Applied: {total_water_applied}")
    print(f"\nTotal Water Applied: {total_water_applied}")
    return total_water_applied, df_curr_till_end.to_string(index=False)

data_extraction_tool("wheat")