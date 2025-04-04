#from supervisor_v2 import command_file_format, get_api_data_tool
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
from datetime import datetime
import os
import re
from weather_data_retriever import OpenMeteoWeatherDownloader as openMeteoDataRetriever
import math
import logging
import configparser
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read("ubuntu_config.ini")

def command_file_format(start_date: str, end_date: str):
    """
    This function is used in order to modify the Command file.
    Modifies the Field Parameters for the simulation.
    MUST BE EXECUTED ONLY ONCE!!!
    
    Args:
        start_date: starting date of the period, FORMAT: YYYY-MM-DD.
        end_date: ending date of the period, FORMAT: YYYY-MM-DD.
        
    """
    logger.info("Inside Command File Format")
    data_json_path = config["Paths"]["api_data_file"].replace("{json_name}", "clean_data")
    
    with open(data_json_path, "r") as file:
        data_json = json.load(file)

    location = data_json.get("location")
    latitude = data_json.get("latitude")
    longitude = data_json.get("longitude")
    altitude = data_json.get("Altitude")
    slope = data_json.get("Slope")
    sand = data_json.get("Sand")
    silt = data_json.get("Silt")
    clay = data_json.get("Clay")
    BD = data_json.get("BD")
    LL15 = data_json.get("LL15")
    DUL = data_json.get("DUL")
    SAT = data_json.get("SAT")
    LL = data_json.get("LL")
    PH = data_json.get("pH")
    ESP = data_json.get("ESP")
    CEC = data_json.get("CEC")
    EC = data_json.get("EC")
    NO3 = data_json.get("NO3")
    carbon = data_json.get("Carbon")
    cn_ratio = data_json.get("SoilCNRatio")
    start_age = data_json.get("Start_Age")
    crop = data_json.get("crop_type")
    #crop = "Pear"


    weather_csv = location + ".csv"
    weather_ini = location + ".ini"

    # ApsimX validation Checks
    SAT_max = (1 - (BD/2.65))
    # round SAT down to two decimals 
    SAT_max = math.floor(SAT_max * 100) / 100.0

    if(SAT > SAT_max):
        SAT = SAT_max

    updates = {
        "load": crop,
        "[Clock].Start": start_date,
        "[Clock].End": end_date,
        "[Weather].FileName": weather_csv,
        "[Weather].ConstantsFile": weather_ini,
        "[Soil].Latitude": latitude,
        "[Soil].Longitude":longitude,
        "[Physical].ParticleSizeSand[1:6]": sand,
        "[Physical].ParticleSizeSilt[1:6]": silt,
        "[Physical].ParticleSizeClay[1:6]": clay,
        "[Physical].BD[1:6]": BD,
        "[Physical].LL15[1:6]": LL15,
        "[Physical].DUL[1:6]": DUL,
        "[Physical].SAT[1:6]": SAT,
        "[Chemical].PH[1:6]": PH,
        "[NO3].InitialValues[1:2]": ", ".join(map(str,NO3)),
        "[Organic].Carbon[1:2]": carbon,
        "[Organic].SoilCNRatio[1:6]":cn_ratio,
        "[SoilWaterUpdate].Script.FilePath": config["Paths"]["soil_moisture_data"].replace("{crop}", crop)

    }
    print(crop)

    growth_type = data_json.get("growth_type")
    # check if wenhave a perennial crop
    if(growth_type == "perennial"):
        # load the proper commands_file
        commands_file = config.get("Paths", "perennial_commands_file")
        # perform the updates
        updates.update({
            "[Row].Altitude":altitude,
            "[Row].Slope": slope,
            "[Alley].Altitude":altitude,
            "[Alley].Slope": slope,
            "[Row].Soil.Chemical.ESP[1:6]": ESP,
            "[Row].Soil.Chemical.CEC[1:6]": CEC,
            "[Row].Soil.Chemical.EC[1:6]": EC,
            f"[{crop}Soil].LL[1:6]": LL,
            "[SlurpSoil].LL[1:6]": LL,
            "[TreeInitialisation].Script.StartAge": start_age,
            "[TreeInitialisation].Script.SowingDate": start_date,
        })
    # Else we have an annual crop
    else:
        # load the proper commands_file
        commands_file = config.get("Paths", "annual_commands_file")
        # perform the updates
        updates.update({
            "[Soil].Chemical.ESP[1:6]": ESP,
            "[Soil].Chemical.CEC[1:6]": CEC,
            "[Soil].Chemical.EC[1:6]": EC,
            f"[{crop}Soil].LL[1:6]": LL,
        })
    
    with open(commands_file, "r") as file:
        lines = file.readlines()

    new_lines = []
    # Process each line
    for line in lines:
        updated_line = line
        # Loop through each parameter to update
        for param, new_value in updates.items():
        # Build a regex pattern that matches the parameter name at the beginning of a line
        # (allowing for optional whitespace) and an equal sign with optional whitespace.
            if param == "load":
                pattern = re.compile(r"^\s*" + re.escape(param) + r"\s+.*\.apsimx", re.IGNORECASE)
                replacement = f"{param} {new_value}.apsimx"
            else:
                if new_value is None: ## Checking if we have a new value for this param
                    continue
                pattern = re.compile(r"^\s*" + re.escape(param) + r"\s*=\s*.*", re.IGNORECASE)
                replacement = f"{param} = {new_value}"    
                
            
            #pattern = re.compile(r"^\s*" + re.escape(param) + r"\s*=\s*.*", re.IGNORECASE)
            if pattern.match(updated_line):
              # Replace the entire line with the new parameter setting.
                updated_line = replacement + "\n"
               # Once matched and updated, no need to check further parameters for this line.
                break
            
             # This regex captures any crop-specific LL parameter, e.g. [PearSoil].LL[1:6] = 0.16.
            crop_ll_pattern = re.compile(r"^\s*\[([A-Za-z]+)Soil\]\.LL\[1:6\]\s*=\s*(.*)", re.IGNORECASE)
            match_crop = crop_ll_pattern.match(updated_line)
            if match_crop:
                existing_crop = match_crop.group(1)  # Captured crop from the line.
                # If the line does not belong to "Slurp" and is not already the current crop,
                # update it to use the current crop.
                if existing_crop.lower() != "slurp" and existing_crop.lower() != crop.lower():
                    updated_line = f"[{crop}Soil].LL[1:6] = {LL}\n"
        new_lines.append(updated_line)

        # Write the updated lines back to the file (or to a new file if preferred)
        with open(commands_file, "w") as file:
            file.writelines(new_lines)    

    return "Command File Formatted"

command_file_format("2025-03-04","2025-03-19")