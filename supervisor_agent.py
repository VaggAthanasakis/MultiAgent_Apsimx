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
from weather_data_retriever import OpenMeteoWeatherDownloader as openMeteoDataRetriever

# Ignore all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tkinter")

os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langsmith-onboarding"

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
    print("Inside Tool")
    apsim_exe = r"c:\Users\vagga\Desktop\test_apsim_GUI\program\APSIM2025.1.7644.0\bin\Models.exe"
    commands_file = r"C:\Users\vagga\Desktop\test_apsim_GUI\Python_Integration\commands"
    #output_dir = r"C:\Users\vagga\Desktop\test_apsim_GUI\Python_Integration\output" 

    subprocess.run([apsim_exe, ' ','--apply', commands_file], check=True)

    # Path to your .db file
    db_path = r"C:\Users\vagga\Desktop\test_apsim_GUI\Python_Integration\barley_irrigation_tests.db"

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

    #print(df.head())
    # Close the connection
    conn.close()

    plt.plot(df['Clock.Today'], df['waterApplied'], marker='o', linestyle='-')

    plt.xlabel('Date')
    plt.ylabel('Water Applied (mm)')
    plt.title('Water Applied Over Time')

    #plt.xticks(rotation=45)  # Rotate for better readability
    plt.show()
    plt.close()
    

@tool
def weather_data_retrieve_tool(location: str, latitude, longtitude: int, start_date: str, end_date: str):
    """"
    Retrieve the weather data for a specific location
    in a spesific period

    Args:
        location: The location for which the weather data will be retrieved
        latitude: The latitude of this location
        longtitute: The longtitude of this location
        start_date: starting date of the period, FORMAT: YYYY-MM-DD
        end_date: ending date of the period, FORMAT: YYYY-MM-DD
    """
    csv_file = location + ".csv"
    ini_file = location + ".ini"
    retriver = openMeteoDataRetriever(location=location,
                                      latitude=latitude,
                                      longitude=longtitude,
                                      start_date=start_date,
                                      end_date=end_date,
                                      csv_filename=csv_file,
                                      ini_filename=ini_file)
    retriver.fetch_and_process()

    # update the commands file
    updates = {
        "[Clock].Start": start_date,
        "[Clock].End": end_date,
        "[Weather].FileName": csv_file,
        "[Weather].ConstantsFile": ini_file
    }
    retriver.update_command_params(file_path="commands", updates=updates)




# Create a Supervisor Agent
members = ["crop_simulator","weather_data_retriever"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


# system_prompt = (
#     """
#     You are a supervisor tasked with managing a conversation between the "
#     f" following workers: {members}. Given the following user request,"
#     " respond with the worker to act next. Each worker will perform a"
#     " task and respond with their results and status. When finished,"
#     " respond with FINISH."
#     """
# )

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options] # type: ignore

llm = ChatOllama(model="llama3.1:8b", temperature = 0.1)

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
        f"Your progress so far: {progress_str} (If a task has completed, you dont need to do it again)"
        "If an agents works is included in the progress, DO NOT CALL HIM AGAIN"
        "For example, if weather_data_retriever is in the progress, Do not Call weather_data_retriever_agent again."
        "Given the conversation, output a JSON object with the key 'next' whose value is one of the following: "
        f"{members + ['FINISH']}. "
        "When finished, output FINISH."
)

    print("\nPROGRESS: ",progress)
    
    messages = [{"role": "system", "content": updated_prompt}, ] + state["messages"]

    response = llm.with_structured_output(Router).invoke(messages)
    if response is None:
        raise ValueError("LLM did not return valid structured output. Response was None.")
    
    goto = response["next"]
    print(f"\nSUpervisor NExt: {response}\n")

    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})

# Agents
crop_simulator_agent = create_react_agent(
    llm,
    tools=[apsim_tool],
    messages_modifier="You are a crop simulator agent tasked with performing crop \
                       simulations about the development, the yield and the irrigation \
                       demands of a spesific crop."
)

#general_agent = llm
weather_agent = create_react_agent(
    llm,
    tools=[weather_data_retrieve_tool],
    messages_modifier="You are an agent responsible for providing weather data \
                       for spedific locations and spedific time periods. \
                       You only have to retrieve data once."
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

def weather_retriever_node(state: State) -> Command[Literal["supervisor"]]:
    result = weather_agent.invoke(state)
    # Append that the weather retrieval is done
    new_progress = state.get("progress", []) + ["weather_data_retriever"]
   
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="weather_data_retriever")
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
builder.add_node("weather_data_retriever",weather_retriever_node)
graph = builder.compile()


display_graph(graph)

prompt = """Retrieve, just once, weather data for the city of Chania with Latitude 35.513828, Longitude	24.018038,
            for the period starting from 2024-06-01 until 2024-10-20. DO THIS ONLY ONE TIME.
            After that, create a simulation about the crop Barley.
            Then finish"""
messages = [HumanMessage(content=prompt)]

messages = graph.invoke({'messages':messages, 'progress':[]})

for m in messages["messages"]:
    m.pretty_print()

# for s in graph.stream(
#     {"messages": [("user", "Create me a simulation about the barley crop. After that, terminate")]}, subgraphs=True
# ):
#     print(s)
#     print("==========================================================================")

