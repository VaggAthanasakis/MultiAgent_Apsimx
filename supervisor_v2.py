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


# Ignore all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tkinter")

# os.environ["LANGCHAIN_API_KEY"] = "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"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "langsmith-onboarding"

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read("ubuntu_config.ini")

def display_graph(graph):
    image_data = graph.get_graph().draw_mermaid_png()
    image = PILImage.open(BytesIO(image_data))
    
    # Convert the image to a format matplotlib can display
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')  # Hide axes if you prefer
    plt.show()


@tool
def apsim_tool(start_date: str, end_date: str, crop_type: str):
    """
    Creates a crop simulation about the development, the yield and the 
    irrigation demands of a spesific crop.
    Uses the Apsimx simulation model in order to perform the simulation.
    Requires the output weather file of the tool weather_data_retrieve_tool in order to run.

    Args:
        start_date: starting date of the period, FORMAT: YYYY-MM-DD.
        end_date: ending date of the period, FORMAT: YYYY-MM-DD.
        crop_type: either 'perennial' or 'annual'
        
    """

    # Command File Format moved here
    command_file_format(start_date, end_date)
    
    #logger.info("Inside Apsim Tool")
    print("\nInside Apsim Tool")

    apsim_exe = config.get("Paths", "apsim_exe")

    ###
    if(crop_type == "perennial"):
        commands_file = config.get("Paths", "perennial_commands_file")
    else:
        commands_file = config.get("Paths", "annual_commands_file")

    try:
        subprocess.run([apsim_exe, ' ','--apply', commands_file], check=True)
        return "SIMULATION PERFORMED"
    except subprocess.CalledProcessError as e:
        #logger.error(f"ApsimX Tool Failed: {e}")
        raise


@tool
def weather_data_retrieve_tool(start_date: str, end_date: str):
    """"
    Retrieve the weather data for a specific location
    in a specific period
    Returns the weather file that is used to the apsim tool.

    Args:
        start_date: starting date of the period, FORMAT: YYYY-MM-DD
        end_date: ending date of the period, FORMAT: YYYY-MM-DD
    Returns:
        total_rain: the total amount of rain for that period
    """
        #logger.info("Inside Weather Tool")
    print("\nInside Weather Tool")
    print(f"start: {start_date}, end: {end_date}")

    # Retrieve all the data needed from the api response
    data_json_path = config["Paths"]["api_data_file"].replace("{json_name}", "clean_data")
    
    # Read the data from the api response file
    with open(data_json_path, "r") as file:
        data_json = json.load(file)

    # get the information that is necessary 
    location = data_json.get("location")
    latitude = data_json.get("latitude")
    longitude = data_json.get("longitude")



    # file Paths for the weather files
    csv_file_path = config["Paths"]["weather_csv"].replace("{location}",location)
    ini_file_path = config["Paths"]["weather_ini"].replace("{location}",location)


    retriever = openMeteoDataRetriever(location=location,
                                      latitude=latitude,
                                      longitude=longitude,
                                      start_date=start_date,
                                      end_date=end_date,
                                      csv_filename=csv_file_path,
                                      ini_filename=ini_file_path)
    total_rain = retriever.fetch_and_process()
    print(f"Total Rain: {total_rain}")

    # Ensure that the weather files have been created before returnig
    while not (os.path.exists(csv_file_path) and os.path.exists(ini_file_path)):
        #logger.info("Sleeping")
        time.sleep(0.1)

    return total_rain


@tool
def get_sensor_data_tool(crop: str):
    """
    This tool is used in order to gather sensor data from the field, like
    soil moisture in various soil depths.
    It stores the results into a file.
    This tool must be executed everytime.

    The results of this tool have the form of: date, layer, soil moisture
    The date is in the format of YYYY-MM-DD, the layer is an integer
    and the soil moisture is a float number.
    The date is the date of the simulation.
    The layer is the soil depth in cm.
    The soil moisture is the soil moisture in mm
    
    Args:
        crop: the crop that the simulation performed to. For example could be 'pear', 'olive', 'wheat', 'potato', 'corn', 'barley' etc.
        crop_type: returns the crop type 'annual' or 'perennial' 
    """
    crop = crop.capitalize()
    print(f"\nCROP: {crop}")
    #logger.info("Inside sensor_data_tool")
    print("\nInside sensor_data_tool")
    sensor_data_file_path = config["Paths"]["soil_moisture_data"].replace("{crop}", crop)
    #print("\n",sensor_data_file_path)
    # Default sensor data if none provided

    sensor_data = [
        ("2025-04-1", 0, 20),
        ("2025-04-2", 0, 20),
        ("2025-04-3", 0, 12),
        ("2025-04-4", 0, 23),
        ("2025-04-5", 0, 32),
        ("2025-04-6", 0, 23),
        ("2025-04-7", 0, 23),
        ("2025-04-8", 0, 20),
        ("2025-04-9", 0, 20),
    ]

    # ​To convert volumetric water content (VWC) to millimeters (mm) of water within a specific soil depth, 
    # you can use the following formula:​
    # Depth of Water (mm) = VWC (%) × Soil Depth (mm) / 100


    with open(sensor_data_file_path, "w") as txtfile:
        # Write header
        txtfile.write("Date,Layer,SW\n")
        # Write each record
        for record in sensor_data:
            date_val, layer, sw = record
            # If the date is a datetime object, convert it to a string in YYYY-MM-DD format.
            if isinstance(date_val, datetime):
                date_val = date_val.strftime("%Y-%m-%d")
            txtfile.write(f"{date_val},{layer},{sw}\n")

    return sensor_data

@tool
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
    db_path = config["Paths"]["db_path"].replace("{crop}",crop)
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

    total_water_applied = df['waterApplied'].sum()
    
    # Close the connection
    conn.close()

    #logger.info(f"Total Water Applied: {total_water_applied}")
    print(f"\nTotal Water Applied: {total_water_applied}")
    return total_water_applied


@tool
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
    full_json_data = config["Paths"]["api_data_file"].replace("{json_name}","full_data")
    clean_json_data = config["Paths"]["api_data_file"].replace("{json_name}","clean_data")
    

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
    
    # soil_analysis = {}
    # for value in data["farmland"]["latest_soilanalysis"]["soilanalysis_values"]:
        
    #     prop_name = value["soilanalysis_property"]["name"].strip()  # Normalize names
        
    #     try:
    #         # Convert values to float (if possible)
    #         soil_analysis[prop_name] = float(value["value"])
    #     except (ValueError, TypeError):
    #         # Keep as string if conversion fails
    #         soil_analysis[prop_name] = value["value"]


    # get soil Analysis values
    soil_analysis = {}
    try:
        soil_values = data.get("farmland", {}).get("latest_soilanalysis", {}).get("soilanalysis_values", [])
        if not soil_values:
            raise ValueError("Missing or empty 'soilanalysis_values' in API response.")
        
        for value in soil_values:
            prop_name = value.get("soilanalysis_property", {}).get("name", "").strip()  # Normalize names
            if not prop_name:
                continue  # Skip if property name is missing
            
            try:
                # Convert values to float (if possible)
                soil_analysis[prop_name] = float(value["value"])
            except (ValueError, TypeError):
                # Keep as string if conversion fails
                soil_analysis[prop_name] = value["value"]
    except Exception as e:
        print(f"Error processing soil analysis data: {e}")


    # Safely get coordinates with try/except (to handle missing keys or indices)
    try:
        longitude = data["farmland"]["coordinates"]["coordinates"][0][0][0]
    except (KeyError, IndexError, TypeError):
        longitude = None
    
    try:
        latitude = data["farmland"]["coordinates"]["coordinates"][0][0][1]
    except (KeyError, IndexError, TypeError):
        latitude = None
    
    # get these values in order to return them
    crop =  data.get("croptype", {}).get("name")
    growth_type = data.get("croptype", {}).get("growth_type")

    clay = soil_analysis.get("clay") if soil_analysis.get("clay") is not None else 35
    sand = soil_analysis.get("sand") if soil_analysis.get("sand") is not None else 35
    silt = soil_analysis.get("silt") if soil_analysis.get("silt") is not None else 100-clay-sand

    planting_date = data.get("start_date") # may need to be data.get("sowing_date")


    extracted_data = {
        # get() without default returns None if keys are missing
        "crop_type": crop,
        "growth_type": growth_type,
        "longitude": longitude,
        "latitude": latitude,
        # Soil analysis values (will be None if keys don't exist)
        "Clay": clay,
        "Sand": sand,
        "Silt": silt,
        "pH": soil_analysis.get("ph") if soil_analysis.get("ph") is not None else 7.5,
        "BD": 1.16,                  ## We have to take the below values from the api
        "LL15": 0.28,
        "DUL": 0.36,
        "SAT": 0.56,
        "LL": 0.28,
        "ESP": 0.25,
        "CEC": 49.67,
        "EC": 0.304,
        "NO3": [3.1, 2.55],
        "Carbon": 4.53,
        "SoilCNRatio": 7.44,
        "InitialCNR": 7.44,
        "location": "Heraklion",
        "planting_date": planting_date,
    }

    # Check if we have a annual or a perennial crop
    if(growth_type == "perennial"):
        
        extracted_data.update({
            "Start_Age": 1,
            "Altitude": data.get("farmland", {}).get("elevation"),
            "Slope": data.get("farmland", {}).get("slope_lon"),
            "SowingDate": data.get("sowing_date"),
        })
    else:
        # may need some more values here
        pass


    print("\nSaving extracted data")
    with open(clean_json_data, "w", encoding="utf-8") as file:
        json.dump(extracted_data, file, indent=4, ensure_ascii=False)

    output = {
    "crop": crop,
    "growth_type": growth_type,
    "sand": sand,
    "silt": silt,
    "clay": clay
    }
    return output

@tool
def get_time():
    """
    This tool is responsible for returning the current 
    date in the format of yyyy-mm-dd

    """
    #logger.info("Inside get_time Tool")
    print("\nInside get_time Tool")
    current_date = datetime.now().strftime("%Y-%m-%d")
    print(current_date)
    return current_date

# helpfull functions
def command_file_format(start_date: str, end_date: str):
    """
    This function is used in order to modify the Command file.
    Modifies the Field Parameters for the simulation.
    MUST BE EXECUTED ONLY ONCE!!!
    
    Args:
        start_date: starting date of the period, FORMAT: YYYY-MM-DD.
        end_date: ending date of the period, FORMAT: YYYY-MM-DD.
        
    """
    #logger.info("Inside Command File Format")
    print("\nInside Command File Format")
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
    planting_date = data_json.get("planting_date")

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
            "[Sow].Script.PlantingDate": planting_date,
            # "[Sow].Script.PlantingDepth": planting_depth,
            # "[Sow].Script.RowSpacing": row_space,
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
                
            if pattern.match(updated_line):
              # Replace the entire line with the new parameter setting.
                updated_line = replacement + "\n"
               # Once matched and updated, no need to check further parameters for this line.
                break

            # --- Special Handling for Crop-Specific Soil LL Parameter ---
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


# Create a Supervisor Agent
#members = ["crop_simulator","simulation_analysis","advisor"]
members = ["crop_simulator","greek_translator"]
#members = ["advisor"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options] # type: ignore


#llm = ChatOllama(model="llama3.3:latest", temperature = 0)
# llm = ChatOllama(model="llama3.1:8b", temperature = 0)
llm = ChatOllama(model="qwen2.5:72b", temperature = 0)
greek_llm = ChatOllama(model="MHKetbi/ilsp-Llama-Krikri-8B-Instruct", temperature = 0)

class State(MessagesState):
    next: str
    progress: list

def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]: # type: ignore
    
    # Check for progress in the state
    progress = state.get("progress", [])
    progress_str = f"Completed tasks: {', '.join(progress)}." if progress else ""
    print(f"\n{progress_str}")
    # Update the system prompt to include progress information
    updated_prompt = (
        f"You are a supervisor managing a conversation between the following workers: {members}. "
        f"You have already completed the tasks: {progress_str} "
        "Given the conversation, output a JSON object with the key 'next' whose value is one of the following: "
        f"{members + ['FINISH']}. FINISH is outputed when all tasks have completed."
        "If the progress contains worker: 'crop_simulator' YOU CANNOT ROUTE TO 'crop_simulator' AGAIN"
        "If the progress contains worker: 'simulation_analysis' YOU CANNOT ROUTE TO 'simulation_analysis' AGAIN"
        "If you cannot route anywhere, RETURN key-word: 'FINISH' "        
)

    #print("\nPROGRESS: ",progress)
    
    messages = [{"role": "system", "content": updated_prompt}, ] + state["messages"]

    response = llm.with_structured_output(Router).invoke(messages)
    #print("STRUCTURED RESPONSE: ", response)
    if response is None:
        print("NONEeeeeeeee\n")
        goto = "FINISH"
        #raise ValueError("LLM did not return valid structured output. Response was None.")
    else:
        goto = response["next"]

    print(f"\nSUpervisor NExt: {response}\n")

    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})

# Agents
crop_simulator_agent = create_react_agent(
    llm,
    tools=[get_api_data_tool, weather_data_retrieve_tool, get_sensor_data_tool, apsim_tool, get_time, data_extraction_tool],
    messages_modifier="""
    You are an AI assistant designed to help with Crop activities by performing crop simulations and give a response to the farmer
    after reasoning about the data. Your responses must be in English NOT IN GREEK, thats another's agent work.

    Use ONE tool per response. Format: {"name": "<tool_name>", "parameters": {}}.
    The order of the tool execution MUST BE:
        1) get_api_data_tool: this tool is used in order to retrieve data  from an api endpoint (like sand(%), silt(%), clay(%)).
        2) weather_data_retrieve_tool: this tool is used in order to retrieve the weather data for a specific location
        3) get_sensor_data_tool: this tool is used in order to retrieve the soil humidity data (per layer of soil) from the field.
        4) apsim_tool: this tool is used in order to perform the crop simulation.
        5) data_extraction_tool: this tool is used in order to extract data from the simulation (like total water applied).
    If you there is not starting or ending date, before the call of 'weather_data_retrieve_tool' use the tool get_time to get the current date and set this current
    date as the starting date for the 'weather_data_retrieve_tool' and use as end date the current date +  days after that.
    Always call the tools with this order when you want to perform a simulation.
    Its important to keep all the values of the output of the get_api_data_tool (sand,silt,clay,crop,growth_type) in the state of the graph.

    
    AFTER THE TOOLS EXECUTION:
    -> You have to reason about the data and give a response to the farmer.
    -> Consider and include in the response the data that you have received from the tools like: water applied, weather forecast, percentages of sand-silt-clay of the Soil Texture, the crop and the growth type.
    -> If you think that the simulation gives a smaller amount of water than the optimal amount of water, you have to give a recommendation to the farmer about how much water he has to apply.
    -> In order to be more accurate, you have to take into consideration the soil texture (sand, silt, clay) and the weather forecast (rainfall).
    -> The response must be easily understandable by a farmer.
    -> Do not include polite closing remarks or encouragements to ask more questions. End responses directly with the relevant information.

    Example output format:
    'Irrigation Advisory Report

    Crop: Wheat
    Soil Texture:
    Sand: x%
    Silt: y%
    Clay: z%

    Weather Forecast 
    Total Expected Rainfall: 0.2 mm

    Soil Moisture (Upper Layer): 60 mm

    Irrigation Applied in the simulation: 12 mm

    → Usable water in the root zone is below optimal.

    Recommended Water Application:

    Apply 40 mm of irrigation water

    This will replenish the root zone to near field capacity.'
    
"""

)

greek_translator_agent = create_react_agent(
    greek_llm,
    tools = [],
    messages_modifier = """
    You are an AI assistant designed to help with Crop activities by translating the response of the simulation analysis to Greek.
    the response of the simulation analysis is in English and you have to translate it to Greek.
    The user has already performed a crop simulation and the response of the simulation analysis is in English.
    Your task is to translate the response to Greek.
    The reponse must be easily understandable by a farmer.

    Example output format: (Example Values, Do not use them)
    'Αναφορά Άρδευσης'
    Καλλιέργεια: Σιτάρι
    Σύνθεση Εδάφους:
    Άμμος: x%
    Ίλυς: y%
    Άργιλος: z%

    Πρόγνωση Καιρού:
    Συνολική Αναμενόμενη Βροχόπτωση: 0.2 mm
    Υγρασία Εδάφους (Άνω Στρώμα): 60 mm

    Άρδευση που εφαρμόστηκε στην προσομοίωση: 12 mm
    → Το διαθέσιμο νερό στη ρίζα είναι κάτω από το βέλτιστο.

    Συνιστώμενη Εφαρμογή Νερού:
    Εφαρμόστε 40 mm αρδευτικού νερού τις επόμενες ημέρες.

"""
)


# Create the nodes

# graph nodes
def crop_simulator_agent_node(state: State) -> Command[Literal["supervisor"]]:
    print("\nInside Crop Simulator Node")
    result = crop_simulator_agent.invoke(state)
    print(f"\n\nRESULT: {result["messages"][-1].content}\n")

    new_progress = state.get("progress", []) + ["crop_simulator"]
   
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="crop_simulator")
            ],
            "progress": new_progress,
        },
        goto="supervisor",
    )

def greek_translator_agent_node(state: State) -> Command[Literal["supervisor"]]:
    print("\nInside greek translator Node")
    result = greek_translator_agent.invoke(state)
    # Append that the weather retrieval is done
    new_progress = state.get("progress", []) + ["greek_translator"]
    print("New_Progress: ",new_progress)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="greek_translator")
            ],
            "progress": new_progress,
        },
        goto="supervisor",
    )





def multiagent_Apsimx_Simulator(prompt: str):

    # Build the Graph
    builder = StateGraph(State)
    builder.add_edge(START,"supervisor")
    builder.add_node("supervisor",supervisor_node)
    builder.add_node("crop_simulator",crop_simulator_agent_node)
    builder.add_node("greek_translator",greek_translator_agent_node)
    #builder.add_node("advisor",advisor_agent_node)
    graph = builder.compile()
    
    messages = [HumanMessage(content=prompt)]

    messages = graph.invoke({'messages':messages, 'progress':[]})

    for m in messages["messages"]:
        m.pretty_print()


    # remove the .temp files
    for temp_file in Path('.').glob('*.temp'):
        temp_file.unlink()



# This will be the prompt that the user will give to the system from the frontend
# user_prompt = """
#     1) Create a crop simulation in the field with id = 62 for the period starting from today and for the next 7 days.
#     2) Analyse the Data of the simulation in order to output the total water that the simulation is recommends to apply.
#     3) Give some advice for the crop to the farmer for the next days in greek.

# """

user_prompt = """
    1) Create a crop simulation in the field with id = 62 for the period starting from 10-10-2024 until 10-10-2025.
    2) Analyse the Data of the simulation in order to output the total water that the simulation is recommends to apply.
    3) Give some advice for the crop to the farmer for the next days in greek.

"""
# user_prompt = """
#     Give advice for the irrigation demands for a field of Wheat starting from today and for the next 7 days.
#     Do not perform any simulation.
# """


multiagent_Apsimx_Simulator(user_prompt)