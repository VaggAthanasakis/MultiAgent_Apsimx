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


def get_api_data_tool(field_id: int):
    """
    This tool is used in order to retrieve data from an api endpoint.
    Retrieves data like: Field parameters (pH, NH4,...), Field location,
    type of crop, growth type, ...

    Args:
        field_id: the id of the Field that the cultivation is taking place
    Returns:
        crop: the crop name 
        growth_type:  perennial or annual
        sand: the sand percentage of the Soil Texture
        silt: the silt percentage of the Soil Texture
        clay: the clay percentage of the Soil Texture
    """
    #logger.info("Inside API Tool")
    print("\nInside API Tool")
    # The file where the data will be stored
    full_json_data = "full_json_data.json"
    clean_json_data = "clean_json_data.json"
    

    url = f"https://api.aigrow.gr/api/v1/se/seasons_operations/{str(field_id)}/"
    headers = {
        "X-API-Key": "N6mzKHWL1ycWygQn1WsnJs0vrTs0gYKq_x_K9pC0Fr8"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes (4xx, 5xx)
        data = response.json()
        with open(full_json_data, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        print(f"Data saved to {full_json_data}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

get_api_data_tool(62)