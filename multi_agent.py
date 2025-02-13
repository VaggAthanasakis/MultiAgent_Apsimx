import subprocess
import pandas as pd
import sqlite3
import warnings
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END , MessagesState
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Literal
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from weather_data_retriever import OpenMeteoWeatherDownloader as openMeteoDataRetriever
import re
import math

# Ignore all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tkinter")

def display_graph(graph):
    image_data = graph.get_graph().draw_mermaid_png()
    image = PILImage.open(BytesIO(image_data))
    
    # Convert the image to a format matplotlib can display
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')  # Hide axes if you prefer
    plt.show()
    # image_data = graph.get_graph().draw_mermaid_png()
    # image = PILImage.open(BytesIO(image_data))
    # image.show()


@tool
def apsim_tool(crop: str):
    """
    Creates a crop simulation about the development, the yield and the 
    irrigation demands of a spesific crop.
    Uses the Apsimx simulation model in order to perform the simulation.
    """
    print(f"\nAPSIM TOOL {crop}\n")
    

    apsim_exe = r"c:\Users\vagga\Desktop\test_apsim_GUI\program\APSIM2025.1.7644.0\bin\Models.exe"
    commands_file = r"C:\Users\vagga\Desktop\test_apsim_GUI\Python_Integration\APSIM_FILES\pear_commands"

    subprocess.run([apsim_exe, ' ','--apply', commands_file], check=True)
    print("Simulation Executed.")

    return "Simulation Executed."
    

@tool
def weather_data_retrieve_tool(location: str, latitude: float, longitude: float, start_date: str, end_date: str):
    """"
    Retrieve the weather data for a specific location
    in a spesific period

    Args:
        location: The location for which the weather data will be retrieved
        latitude: The latitude of this location
        longitude: The longtitude of this location
        start_date: starting date of the period, FORMAT: YYYY-MM-DD
        end_date: ending date of the period, FORMAT: YYYY-MM-DD
    """
    print("\nWEATHER TOOL\n")

    # file Paths
    csv_file_path = f"APSIM_FILES\{location}.csv"
    ini_file_path = f"APSIM_FILES\{location}.ini"


    # files
    csv_file = location + ".csv"
    ini_file = location + ".ini"

    retriever = openMeteoDataRetriever(location=location,
                                      latitude=latitude,
                                      longitude=longitude,
                                      start_date=start_date,
                                      end_date=end_date,
                                      csv_filename=csv_file_path,
                                      ini_filename=ini_file_path)
    retriever.fetch_and_process()

    return "Weather Data Acquired."


@tool
def command_file_format_tool(
    start_date: str, end_date: str, latitude: float, longitude: float,
    sand: float, silt: float, clay: float, BD: float, LL15: float,
    DUL: float, SAT: float, LL: float, PH: float, ESP: float, CEC: float,
    EC: float, NO3: list, carbon: float, cn_ratio: float, start_age: int,
    location: str                       
    ):
    """
    This tool is used in order to modify the Command file.
    Must be the 1st tool that will be executed.
    MUST BE EXECUTED ONLY ONCE!!!
    Modifies the Field Parameters for the simulation.

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
        
    """
    print("\nCommand TOOL\n")

    command_file = r"APSIM_FILES/pear_commands"

    csv_file = location + ".csv"
    ini_file = location + ".ini"

    # ApsimX validation Checks
    SAT_max = (1 - (BD/2.65))
    # round SAT down to two decimals 
    SAT_max = math.floor(SAT_max * 100) / 100.0

    if(SAT > SAT_max):
        SAT = SAT_max

    updates = {
        "[Clock].Start": start_date,
        "[Clock].End": end_date,
        "[Weather].FileName": csv_file,
        "[Weather].ConstantsFile": ini_file,
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
    


    with open(command_file, "r") as file:
        lines = file.readlines()

    new_lines = []
    # Process each line
    for line in lines:
        updated_line = line
        # Loop through each parameter to update
        for param, new_value in updates.items():
        # Build a regex pattern that matches the parameter name at the beginning of a line
        # (allowing for optional whitespace) and an equal sign with optional whitespace.
            pattern = re.compile(r"^\s*" + re.escape(param) + r"\s*=\s*.*", re.IGNORECASE)
            if pattern.match(updated_line):
              # Replace the entire line with the new parameter setting.
                updated_line = f"{param} = {new_value}\n"
               # Once matched and updated, no need to check further parameters for this line.
                break
        new_lines.append(updated_line)

        # Write the updated lines back to the file (or to a new file if preferred)
        with open(command_file, "w") as file:
            file.writelines(new_lines)    

    return "Command File Formatted"

@tool
def data_extraction_tool(input_file: str):
    """
    This tool is responsible for extracting data given a .db file.
    Can extract data like total water applied, total yield etc.

    Args: 
        input_file: The file from which the data will be extracted

    """
    # Path to your .db file
    # db_path = r"C:\Users\vagga\Desktop\test_apsim_GUI\Python_Integration\APSIM_FILES\pears.db"
    db_path = input_file
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

    # plt.plot(df['Clock.Today'], df['waterApplied'], marker='o', linestyle='-')

    # plt.xlabel('Date')
    # plt.ylabel('Water Applied (mm)')
    # plt.title('Water Applied Over Time')

    # #plt.xticks(rotation=45)  # Rotate for better readability
    # plt.show()
    # plt.close()
    return total_water_applied

llm = ChatOllama(model="llama3.1:8b", temperature = 0)


crop_simulator_agent_prompt = """
You are an AI assistant designed to help with Crop activities. 

Use ONE tool per response. Format: {"name": "<tool_name>", "parameters": {}}.
The order of the tool execution MUST BE:
    1) command_file_format_tool (MUST BE EXECUTED ONLY ONCE)
    2) weather_data_retrieve_tool
    3) apsim_tool
Always call the tools with this order.
command_file_format_tool MUST BE EXECUTED 

If you have the final answer or deliverable,"
prefix your response with FINAL ANSWER so the team knows to stop."
"""

simulation_analysis_agent_prompt = """
You are an AI assistant designed to analyse the output of a crop simulation.
Given the simulation output file, you can provide information about the Total yield of a crop,
the Total amount of water applied etc.
In order to extract data from files, you can use the tool: 'data_extraction_tool'

If you have the final answer or deliverable,"
prefix your response with FINAL ANSWER so the team knows to stop."
"""


def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        print("\nFINAL ANSWEAR\n")
        return END
    return goto

# Crop Simulator Agent
crop_simulator_agent = create_react_agent(
    llm,
    tools = [command_file_format_tool, weather_data_retrieve_tool, apsim_tool],
    messages_modifier = crop_simulator_agent_prompt
)

simulation_analysis_agent = create_react_agent(
    llm,
    tools = [],
    messages_modifier = simulation_analysis_agent_prompt
)

# Node
def crop_simulator_node(state: MessagesState) -> Command[Literal[END]]:
    result = crop_simulator_agent.invoke(state)
    goto = get_next_node(result["messages"][-1],END)

    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="crop_simulator"
    )
    return Command(
        update={
            "messages": result["messages"],
        },
        goto=goto
    )

builder = StateGraph(MessagesState)
builder.add_node("crop_simulator",crop_simulator_node)
builder.add_edge(START,"crop_simulator")
graph = builder.compile()

#display_graph(graph)

prompt = """
    1) Collect weather data  for the location of Tylisos with Latitude 35.513828, Longitude 24.018038
       for the period starting from 2022-01-01 until 2025-01-01.
       The Field parameters are:
            Sand= 5.29, Silt= 20.78, Clay= 73.92, BD= 1.16, LL15= 0.16,
            DUL= 0.36, SAT= 0.8, LL= 0.16, PH= 7.5, ESP= 0.25, 
            CEC= 49.67, EC= 0.304, NO3 = [3.1,2.55], Carbon= 4.53,
            cn_ratio= 7.44, StartAge= 1
    2) Create a simulation for the Crop "Avocado" with these data.
    
    Then Finish.

"""


for s in graph.stream(
    {"messages": [("user", prompt)]}, subgraphs=True
):
    print(s)
    print("\n==========================================================================")

