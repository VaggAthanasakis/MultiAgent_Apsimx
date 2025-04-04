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
        logger.info("Sleeping")
        time.sleep(0.1)

    return total_rain


@tool
def get_sensor_data_tool(crop: str):
    """
    This tool is used in order to gather sensor data from the field, like
    soil moisture in various soil depths.
    It stores the results into a file.
    This tool must be executed everytime.
    
    Args:
        crop: the crop that the simulation performed to.
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
        ("2025-03-5", 0, 0.1),
        ("2025-03-6", 0, 0.1),
        ("2025-03-7", 0, 0.1),
        ("2025-03-8", 0, 0.1),
        ("2025-03-9", 0, 0.1),
        ("2025-03-10", 0, 0.1),
        ("2025-03-11", 0, 0.1),
        ("2025-03-12", 0, 0.1),
        ("2025-03-13", 0, 0.1),
        ("2025-03-14", 0, 0.1),
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

    Returns:
        total_water_applied: The total amount of water applied
    """

    # Path to your .db file
    #logger.info("Inside Data Extraction Tool")
    print("\nInside Data Extraction Tool")

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
    
    soil_analysis = {}
    for value in data["farmland"]["latest_soilanalysis"]["soilanalysis_values"]:
        
        prop_name = value["soilanalysis_property"]["name"].strip()  # Normalize names
        
        try:
            # Convert values to float (if possible)
            soil_analysis[prop_name] = float(value["value"])
        except (ValueError, TypeError):
            # Keep as string if conversion fails
            soil_analysis[prop_name] = value["value"]

    # soil_analysis_values = data.get("farmland", {}).get("latest_soilanalysis", {}).get("soilanalysis_values", [])
    # soil_analysis = {}
    # for value in soil_analysis_values:
    #     try:
    #         # Safely get property name with fallbacks
    #         prop_entry = value.get("soilanalysis_property", {})
    #         prop_name = prop_entry.get("name", "").strip().lower()
            
    #         if not prop_name:
    #             continue  # Skip entries without property name
                
    #         raw_value = value.get("value")
            
    #         # Try converting to float, fallback to original value
    #         soil_analysis[prop_name] = float(raw_value) if raw_value is not None else None
    #     except (ValueError, TypeError):
    #         soil_analysis[prop_name] = raw_value

    # # Set default values if no analysis found
    # if not soil_analysis:
    #     soil_analysis = {
    #         "ph": 7.0,
    #         "organic_matter": 2.5,
    #         "phosphorus": 15.0,
    #         "potassium": 150.0
    #     }
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
        # Soil analysis values (will be None if keys don't exist)
        "Clay": soil_analysis.get("clay"),
        "Sand": soil_analysis.get("sand"),
        "Silt": soil_analysis.get("silt"),
        "pH": soil_analysis.get("ph"),
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
    }

    # Check if we have a annual or a perennial crop

    crop_type = data.get("croptype", {}).get("growth_type")
    if(crop_type == "perennial"):
        
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

    crop =  data.get("croptype", {}).get("name")
    return crop, crop_type

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
members = ["crop_simulator","simulation_analysis","advisor"]
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
    You are an AI assistant designed to help with Crop activities by performing crop simulations. 

    Use ONE tool per response. Format: {"name": "<tool_name>", "parameters": {}}.
    The order of the tool execution MUST BE:
        1) get_api_data_tool
        2) weather_data_retrieve_tool
        3) get_sensor_data_tool
        4) apsim_tool
    Always call the tools with this order when you want to perform a simulation.
    DO not analyze the data of the simulation.
    It's not your job to communicate with the user, just perform the tool executions without outputting 
    any message to the user.

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

# advisor_agent = create_react_agent(
#     llm,
#     tools = [weather_data_retrieve_tool, get_sensor_data_tool, get_time],
#     messages_modifier = """
#     You are a professinal AI assistant designed to give advice for a specific 
#     crop. You can give advice for the water irrigation demands and for the weather.
#     In order to gain data for the soil humidity for the specific crop, you can gain soil
#     humidity values from the tool 'get_sensor_data_tool'.
#     In order to gain weather forecast for the next days, you can call the tool 'weather_data_retrieve_tool'.
#     Just call the weather_data_retrieve_tool with starting day: current_date and end_day: 7 days after that.
#     In order to find the current_date, use the tool 'get_time'
#     Use ONE tool per response. Format: {"name": "<tool_name>", "parameters": {}}.
    
    
#     Instructions: in order to give advice for a field you have to call the tools with this order ALWAYS:
#         1) get_time
#         2) weather_data_retrieve_tool (args: start_date=current_date end_date: current_date + 7)
#         3) get_sensor_data_tool

#     IT IS HIGHLY IMPORTANT TO CALL THE TOOL weather_data_retrieve_tool. IT IS MANDATORY to use the
#     output of the get_time tool (current_date) as the start_date argument of the weather_data_retrieve tool

#     Aftef that you have to reason for this data.
#     + Check the humidity values of the soil and if you think that there is not enough water in the soil as an advice
#       to the farmer for how much water he has to apply.
#     + Check the weather forecast, starting from the current date, for the next 7 days in order to find out the amount 
#       of the rain (mm) that is expected to be dropped.
#     + Combine the soil humidity values with the weather forecast and synthesize a response like a mini advice 
#       to the farmer about the amount of water (in mm) that he has to apply to the field, if needed.

# """
# )
advisor_agent = create_react_agent(
    llm,
    tools=[weather_data_retrieve_tool, get_sensor_data_tool, get_time],
    messages_modifier="""
You are a professional AI assistant designed to give advice for a specific crop.
You provide irrigation and weather recommendations based on soil humidity and weather forecast.

You have access to the following tools:
- get_time: Returns the current date in YYYY-MM-DD format.
- weather_data_retrieve_tool: Retrieves the total rain (in mm) forecasted. It requires a starting date and an ending date. If the ending date is not provided, it computes the date 7 days after the starting date.
- get_sensor_data_tool: Returns the soil humidity value.

Instructions:
1) Always start by calling get_time to get the current date (store this as "current_date").
2) Next, call weather_data_retrieve_tool using current_date as the start_date and end_date start_date + 7 days.
3) Finally, call get_sensor_data_tool to get the soil humidity.

After obtaining the data, analyze:
- If the soil humidity is low and little rain is forecasted, advise the farmer on applying additional water (in mm).
- Otherwise, advise that no extra irrigation is needed.

Remember: Use ONE tool call per response. Format your tool call as: {"name": "<tool_name>", "parameters": {}}.
Also, it is HIGHLY IMPORTANT that when calling weather_data_retrieve_tool, you use the output of get_time (current_date) as its start_date.
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

def advisor_agent_node(state: State) -> Command[Literal["supervisor"]]:
    result = advisor_agent.invoke(state)
    # Append that the weather retrieval is done
    new_progress = state.get("progress", []) + ["advisor"]
   
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="advisor")
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
builder.add_node("advisor",advisor_agent_node)
graph = builder.compile()


#display_graph(graph)



# This will be the prompt that the user will give to the system from the frontend
# user_prompt = """
#     1) Create a crop simulation in the field with id = 62 for the period starting from 2025-03-04 until 2025-03-20
#     2) Analyse the Data of the simulation in order to output the total applied water.
#     3) Give some advice for the crop.

# """
user_prompt = """
    give advice for a field starting from today and for the next 7 days
"""


messages = [HumanMessage(content=user_prompt)]

messages = graph.invoke({'messages':messages, 'progress':[]})

for m in messages["messages"]:
    m.pretty_print()

