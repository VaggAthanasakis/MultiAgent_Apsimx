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
import os
import re
from weather_data_retriever import OpenMeteoWeatherDownloader as openMeteoDataRetriever
import math
import logging
import configparser

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
def apsim_tool():
    """
    Creates a crop simulation about the development, the yield and the 
    irrigation demands of a spesific crop.
    Uses the Apsimx simulation model in order to perform the simulation.
    Requires the output weather file of the tool weather_data_retrieve_tool in order to run.
    """
    
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
def weather_data_retrieve_tool(location: str, latitude: float, longitude: float, start_date: str, end_date: str):
    """"
    Retrieve the weather data for a specific location
    in a spesific period
     Returns the weather file that is used to the apsim tool.


    Args:
        location: The location for which the weather data will be retrieved
        latitude: The latitude of this location
        longitude: The longtitude of this location
        start_date: starting date of the period, FORMAT: YYYY-MM-DD
        end_date: ending date of the period, FORMAT: YYYY-MM-DD
    """
    logger.info("Inside Weather Tool")

    # file Paths
    csv_file_path = config["Paths"]["weather_csv"].replace("{location}",location)
    ini_file_path = config["Paths"]["weather_ini"].replace("{location}",location)

    print("\n",csv_file_path)
    print("\n",ini_file_path)

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
def command_file_format_tool(
    start_date: str, end_date: str, latitude: float, longitude: float,
    sand: float, silt: float, clay: float, BD: float, LL15: float,
    DUL: float, SAT: float, LL: float, PH: float, ESP: float, CEC: float,
    EC: float, NO3: list, carbon: float, cn_ratio: float, start_age: int,
    location: str, crop: str                       
    ):
    """
    This tool is used in order to modify the Command file.
    Modifies the Field Parameters for the simulation.
    MUST BE EXECUTED ONLY ONCE!!!
    
    Args:
        start_date: starting date of the period, FORMAT: YYYY-MM-DD.
        end_date: ending date of the period, FORMAT: YYYY-MM-DD.
        latitude: The latitude of this location.
        longitude: The longtitude of this location.
        sand: The soil sand content (expressed  as a percentage).
        silt: The soil silt content (expressed either as a percentage).
        clay: The soil clay content (expressed either as a percentage).
        BD: The soil bulk density (typically in g/cm³).
        LL15: Moisture level below which water is essentially unavailable to plants.
        DUL: Drained upper limit
        SAT: The saturated water content. 
        LL: The plant lower limit for water extraction
        PH: The soil pH value.
        ESP: The Exchangeable Sodium Percentage (ESP) of the soil.
        CEC: The soil Cation Exchange Capacity (CEC), for example in cmol(+)/kg.
        EC: The soil electrical conductivity (EC), for example in dS/m.
        NO3: A list of nitrate (NO₃⁻) concentration values (units as defined by the model requirements).
        carbon: The soil organic carbon content (expressed as a percentage or fraction).
        cn_ratio: The soil carbon-to-nitrogen (C:N) ratio.
        start_age: The starting age of the crop from which the simulation begins.
        location: The location of the simulation.
        crop: The crop that will be simulated
        
    """
    logger.info("Inside Command Tool")

    # command_file = r"APSIM_FILES/pear_commands"
    
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
        "[Physical].ParticleSizeSand[1:6]": sand,
        "[Physical].ParticleSizeSilt[1:6]": silt,
        "[Physical].ParticleSizeClay[1:6]": clay,
        "[Physical].BD[1:6]": BD,
        "[Physical].LL15[1:6]": LL15,
        "[Physical].DUL[1:6]": DUL,
        "[Physical].SAT[1:6]": SAT,
        "[GrapevineSoil].LL[1:6]": LL,
        "[SlurpSoil].LL[1:6]": LL,
        "[Chemical].PH[1:6]": PH,
        "[Chemical].ESP[1:6]": ESP,
        "[Chemical].CEC[1:6]": CEC,
        "[Chemical].EC[1:6]": EC,
        "[NO3].InitialValues[1:2]": ", ".join(map(str,NO3)),
        "[Organic].Carbon[1:2]": carbon,
        "[Organic].SoilCNRatio[1:6]":cn_ratio,
        "[TreeInitialisation].Script.StartAge": start_age,
        "[TreeInitialisation].Script.SowingDate": start_date
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

@tool
def data_extraction_tool(crop: str):
    """
    This tool is responsible for extracting data given a .db file.
    Can extract data like total water applied.

    Args:
        crop: the crop that the simulation performed to.
    """

    # Path to your .db file

    db_path = config["Paths"]["db_path"].replace("{crop}",crop)
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

    logger.info(f"Total Water Applied: {total_water_applied}")
    return total_water_applied



# Create a Supervisor Agent
members = ["crop_simulator","simulation_analysis"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options] # type: ignore


llm = ChatOllama(model="llama3.1:70b", temperature = 0)

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

    print("\nPROGRESS: ",progress)
    
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
    tools=[command_file_format_tool, weather_data_retrieve_tool, apsim_tool],
    messages_modifier="""
    You are an AI assistant designed to help with Crop activities. 

    Use ONE tool per response. Format: {"name": "<tool_name>", "parameters": {}}.
    The order of the tool execution MUST BE:
        1) weather_data_retrieve_tool
        2) command_file_format_tool (MUST BE EXECUTED ONLY ONCE)
        3) apsim_tool
    Always call the tools with this order.
    command_file_format_tool MUST BE EXECUTED 
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


# Build the Graph
builder = StateGraph(State)
builder.add_edge(START,"supervisor")
builder.add_node("supervisor",supervisor_node)
builder.add_node("crop_simulator",crop_simulator_agent_node)
builder.add_node("simulation_analysis",simulation_analysis_node)
graph = builder.compile()


#display_graph(graph)

prompt = """
    Perform each one of the following tasks:
    1) Collect weather data  for the location of Heraklion with Latitude 35.513828, Longitude 24.018038
       for the period starting from 2022-01-01 until 2025-01-01.
       The Field parameters are:
            Sand= 5.29, Silt= 20.78, Clay= 73.92, BD= 1.16, LL15= 0.16,
            DUL= 0.36, SAT= 0.8, LL= 0.16, PH= 7.5, ESP= 0.25, 
            CEC= 49.67, EC= 0.304, NO3 = [3.1,2.55], Carbon= 4.53,
            cn_ratio= 7.44, StartAge= 1
    2) Create a simulation for the Crop "avocado" with these data.
    3) Analyse the Data of the simulation in order to output the total applied water.
    4) Then Finish.

"""

messages = [HumanMessage(content=prompt)]

messages = graph.invoke({'messages':messages, 'progress':[]})

for m in messages["messages"]:
    m.pretty_print()

