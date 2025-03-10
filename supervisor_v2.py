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


# Ignore all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tkinter")

os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langsmith-onboarding"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
def apsim_tool(start_date: str, end_date: str):
    """
    Creates a crop simulation about the development, the yield and the 
    irrigation demands of a spesific crop.
    Uses the Apsimx simulation model in order to perform the simulation.
    Requires the output weather file of the tool weather_data_retrieve_tool in order to run.

    Args:
        start_date: starting date of the period, FORMAT: YYYY-MM-DD.
        end_date: ending date of the period, FORMAT: YYYY-MM-DD.
        
    """

    # Command File Format moved here
    command_file_format(start_date, end_date)
    
    logger.info("Inside Apsim Tool")

    apsim_exe = config.get("Paths", "apsim_exe")
    commands_file = config.get("Paths", "commands_file")

    try:
        subprocess.run([apsim_exe, ' ','--apply', commands_file], check=True)
        return "SIMULATION PERFORMED"
    except subprocess.CalledProcessError as e:
        logger.error(f"ApsimX Tool Failed: {e}")
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
    """

    # Retrieve all the data needed from the api response
    data_json_path = config["Paths"]["api_data_file"].replace("{json_name}", "clean_data")
    
    # Read the data from the api response file
    with open(data_json_path, "r") as file:
        data_json = json.load(file)

    # get the information that is necessary 
    location = data_json.get("location")
    latitude = data_json.get("latitude")
    longitude = data_json.get("longitude")

    logger.info("Inside Weather Tool")

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
    retriever.fetch_and_process()

    # Ensure that the weather files have been created before returnig
    while not (os.path.exists(csv_file_path) and os.path.exists(ini_file_path)):
        logger.info("Sleeping")
        time.sleep(0.1)

    return "Weather Data Acquired."


@tool
def get_sensor_data_tool(crop: str):
    """
    This tool is used in order to gather sensor data from the field, like
    soil moisture in various soil depths.
    It stores the results into a file.
    This tool must be executed everytime.
    
    Args:
        crop: the crop that the simulation performed to.
    """

    logger.info("Inside sensor_data_tool")
    sensor_data_file_path = config["Paths"]["soil_moisture_data"].replace("{crop}", crop)
    #print("\n",sensor_data_file_path)
    # Default sensor data if none provided

    sensor_data = [
            ("2006-01-23", 0, 0.1),
            ("2006-01-24", 1, 0.1),
            ("2006-01-25", 2, 0.1),
            ("2006-01-26", 0, 0.1),
            ("2006-01-27", 1, 0.1),
            ("2006-01-28", 2, 0.1),
            ("2006-01-29", 0, 0.1),
            ("2006-01-30", 1, 0.1),
            ("2006-02-01", 2, 0.1)
        ]
    
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


@tool
def data_extraction_tool(crop: str):
    """
    This tool is responsible for extracting data given a .db file.
    Can extract data like total water applied.

    Args:
        crop: the crop that the simulation performed to.
    """

    # Path to your .db file
    logger.info("Inside Data Extraction Tool")

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

    logger.info(f"Total Water Applied: {total_water_applied}")
    return total_water_applied


@tool
def get_api_data_tool(field_id: int):
    """
    This tool is used in order to retrieve data from an api endpoint.
    Retrieves data like: Field parameters (pH, NH4,...), Field location,
    type of crop, growth type, ...

    Args:
        field_id: the id of the Field that the cultivation is taking place
    """
    logger.info("Inside API Tool")
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

    soil_analysis = {}
    for value in data["farmland"]["latest_soilanalysis"]["soilanalysis_values"]:
        #print("\nVALUE: ",value)
        prop_name = value["soilanalysis_property"]["name"].strip().lower()  # Normalize names
        #print("\nPROP: ",prop_name)
        try:
            # Convert values to float (if possible)
            soil_analysis[prop_name] = float(value["value"])
        except (ValueError, TypeError):
            # Keep as string if conversion fails
            soil_analysis[prop_name] = value["value"]

    # Safely get coordinates with try/except (to handle missing keys or indices)
    try:
        longitude = data["farmland"]["coordinates"]["coordinates"][0][0][0]
    except (KeyError, IndexError, TypeError):
        longitude = None

    try:
        latitude = data["farmland"]["coordinates"]["coordinates"][0][0][1]
    except (KeyError, IndexError, TypeError):
        latitude = None

    extracted_data = {
        # get() without default returns None if keys are missing
        "crop_type": data.get("croptype", {}).get("name"),
        "growth_type": data.get("croptype", {}).get("growth_type"),
        "longitude": longitude,
        "latitude": latitude,
        "Altitude": data.get("farmland", {}).get("elevation"),
        "Slope": data.get("farmland", {}).get("slope_lon"),
        "SowingDate": data.get("sowing_date"),
        # Soil analysis values (will be None if keys don't exist)
        "Clay": soil_analysis.get("clay"),
        "Sand": soil_analysis.get("sand"),
        "Silt": soil_analysis.get("silt"),
        "pH": soil_analysis.get("ph"),
        "BD": 1.16,                  ## We have to take the below values from the api
        "LL15": 0.16,
        "DUL": 0.36,
        "SAT": 0.56,
        "LL": 0.16,
        "ESP": 0.25,
        "CEC": 49.67,
        "EC": 0.304,
        "NO3": [3.1, 2.55],
        "Carbon": 4.53,
        "SoilCNRatio": 7.44,
        "InitialCNR": 7.44,
        "Start_Age": 1,
        "location": "Heraklion",

    }
    with open(clean_json_data, "w", encoding="utf-8") as file:
        json.dump(extracted_data, file, indent=4, ensure_ascii=False)


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
    #crop = data_json.get("crop_type")
    crop = "pear"

    commands_file = config.get("Paths", "commands_file")

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
        "[Row].Altitude":altitude,
        "[Row].Slope": slope,
        "[Alley].Altitude":altitude,
        "[Alley].Slope": slope,
        "[Physical].ParticleSizeSand[1:6]": sand,
        "[Physical].ParticleSizeSilt[1:6]": silt,
        "[Physical].ParticleSizeClay[1:6]": clay,
        "[Physical].BD[1:6]": BD,
        "[Physical].LL15[1:6]": LL15,
        "[Physical].DUL[1:6]": DUL,
        "[Physical].SAT[1:6]": SAT,
        "[PearSoil].LL[1:6]": LL,
        "[SlurpSoil].LL[1:6]": LL,
        "[Chemical].PH[1:6]": PH,
        "[Row].Soil.Chemical.ESP[1:6]": ESP,
        "[Row].Soil.Chemical.CEC[1:6]": CEC,
        "[Row].Soil.Chemical.EC[1:6]": EC,
        "[NO3].InitialValues[1:2]": ", ".join(map(str,NO3)),
        "[Organic].Carbon[1:2]": carbon,
        "[Organic].SoilCNRatio[1:6]":cn_ratio,
        "[TreeInitialisation].Script.StartAge": start_age,
        "[TreeInitialisation].Script.SowingDate": start_date,
        "[SoilWaterUpdate].Script.FilePath": config["Paths"]["soil_moisture_data"].replace("{crop}", crop)

    }
    
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
        new_lines.append(updated_line)

        # Write the updated lines back to the file (or to a new file if preferred)
        with open(commands_file, "w") as file:
            file.writelines(new_lines)    

    return "Command File Formatted"


# Create a Supervisor Agent
members = ["crop_simulator","simulation_analysis"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options] # type: ignore


llm = ChatOllama(model="llama3.3:latest", temperature = 0)

class State(MessagesState):
    next: str
    progress: list

def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]: # type: ignore
    
    # Check for progress in the state
    progress = state.get("progress", [])
    progress_str = f"Completed tasks: {', '.join(progress)}." if progress else ""
    
    # Update the system prompt to include progress information
    updated_prompt = (
        f"You are a supervisor managing a conversation between the following workers: {members}. "
        f"You have already completed the tasks: {progress_str} "
        "Given the conversation, output a JSON object with the key 'next' whose value is one of the following: "
        f"{members + ['FINISH']}. FINISH is outputed when all tasks have completed."
        "If the progress contains worker: 'crop_simulator' YOU CANNOT ROUTE TO 'crop_simulator' AGAIN"
        "If the progress contains worker: 'simulation_analysis' YOU CANNOT ROUTE TO 'simulation_analysis' AGAIN"
        "If you cannot route anywhere, RETURN FINISH"        
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
    tools=[get_api_data_tool, weather_data_retrieve_tool, get_sensor_data_tool, apsim_tool],
    messages_modifier="""
    You are an AI assistant designed to help with Crop activities. 

    Use ONE tool per response. Format: {"name": "<tool_name>", "parameters": {}}.
    apsim_tool MUST BE EXECUTED LAST
    The order of the tool execution MUST BE:
        1) get_api_data_tool
        2) weather_data_retrieve_tool
        3) get_sensor_data_tool
        4) apsim_tool
    Always call the tools with this order.
    get_sensor_data_tool MUST BE EXECUTED EVERY TIME
    If the user prompt requires a crop simulation, you must call the apsim_tool.
    DO not analyze the data of the simulation.

"""

)

simulation_analysis_agent = create_react_agent(
    llm,
    tools = [data_extraction_tool],
    messages_modifier = """
    You are an AI assistant designed to analyse the output of a crop simulation.
    You can provide information about the Total amount of water applied.
    In order to extract data from files, you can use the tool: 'data_extraction_tool'

"""
)
# graph nodes
def crop_simulator_agent_node(state: State) -> Command[Literal["supervisor"]]:
    result = crop_simulator_agent.invoke(state)
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

def simulation_analysis_node(state: State) -> Command[Literal["supervisor"]]:
    result = simulation_analysis_agent.invoke(state)
    # Append that the weather retrieval is done
    new_progress = state.get("progress", []) + ["simulation_analysis"]
   
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="simulation_analysis")
            ],
            "progress": new_progress,
        },
        goto="supervisor",
    )

# water = data_extraction_tool("pear")

# Build the Graph
builder = StateGraph(State)
builder.add_edge(START,"supervisor")
builder.add_node("supervisor",supervisor_node)
builder.add_node("crop_simulator",crop_simulator_agent_node)
builder.add_node("simulation_analysis",simulation_analysis_node)
graph = builder.compile()


#display_graph(graph)

# prompt = """
#     Perform each one of the following tasks:
#     1) Collect weather data  for the location of Heraklion with Latitude 35.513828, Longitude 24.018038
#        for the period starting from 2024-01-01 until 2025-01-01.
#        The Field parameters are:
#             Sand= 5.29, Silt= 20.78, Clay= 73.92, BD= 1.16, LL15= 0.16,
#             DUL= 0.36, SAT= 0.8, LL= 0.16, PH= 7.5, ESP= 0.25, 
#             CEC= 49.67, EC= 0.304, NO3 = [3.1,2.55], Carbon= 4.53,
#             cn_ratio= 7.44, StartAge= 1
#     2) Create a simulation for the Crop "Pear" with these data.
#     3) Analyse the Data of the simulation in order to output the total applied water.
#     4) Then Finish.

# """

# This will be the prompt that the user will give to the system from the frontend
user_prompt = """
    1) Create a simulation for the Crop "pear" in the field with id = 62 for the period starting from 2024-01-01 until 2025-01-01
    2) Analyse the Data of the simulation in order to output the total applied water.

"""

messages = [HumanMessage(content=user_prompt)]

messages = graph.invoke({'messages':messages, 'progress':[]})

for m in messages["messages"]:
    m.pretty_print()

