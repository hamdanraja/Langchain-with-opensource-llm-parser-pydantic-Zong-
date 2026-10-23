import requests
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage

model="qwen3.5:0.8b"
weather_Url="https://api.open-meteo.com/v1/forecast"
cities={
    "islamabad": (33.6844, 73.0479),
    "rawalpindi": (33.5651, 73.0169),
    "lahore": (31.5204, 74.3587),
    "sialkot": (32.4945, 74.5229),
    "multan": (30.1575, 71.5249),
    "karachi": (24.8607, 67.0011),
}
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}
@tool
def get_weather(city: str) -> str:
       """Get the current weather for Islamabad, Rawalpindi, Lahore, Sialkot, Multan, or Karachi, Pakistan. Use this tool when the user asks about current weather, temperature, humidity, rain, or wind in one of these cities. The city argument must be a supported city name."""
       city=city.strip().lower()
       if city not in cities:
           return f"Sorry, I don't have weather information for {city}. Please choose from Islamabad, Rawalpindi, Lahore, Sialkot, Multan, or Karachi."
       latitude, longitude = cities[city]
       params = {
           "latitude": latitude,
           "longitude": longitude,
           "current": (
               "temperature_2m,relative_humidity_2m,"
               "apparent_temperature,wind_speed_10m,"
               "precipitation,weather_code"
           ),
           "timezone": "Asia/Karachi",
       }
       try:
           response = requests.get(weather_Url, params=params, timeout=20)
           response.raise_for_status()
           current = response.json().get("current")
           if not isinstance(current, dict):
               return f"No current weather data for {city.title()}."
           code = current.get("weather_code")
           return {
               "city": city.title(),
               "time": current.get("time"),
               "temperature_c": current.get("temperature_2m"),
               "feels_like_c": current.get("apparent_temperature"),
               "humidity_percent": current.get("relative_humidity_2m"),
               "wind_speed_kmh": current.get("wind_speed_10m"),
               "precipitation_mm": current.get("precipitation"),
               "condition": WEATHER_CODES.get(code, "Unknown condition"),
           }
       except (requests.RequestException, ValueError) as error:
           return f"Unable to retrieve weather: {error}"
llm= ChatOllama(
     model=model,
     temperature=0,
)
tools=[get_weather]
tool_map={weather_tool.name: weather_tool for weather_tool in tools}
llm_with_tools=llm.bind_tools(tools)
SYSTEM_PROMPT = """
You are a helpful assistant.

Use get_weather for current weather questions about supported cities.
Use the tool's result to answer weather questions accurately.provide answer in list formatted text including bullets no' * ','-', including city name, time, temperature, feels like temperature, humidity, wind speed, precipitation, and weather condition.
For general questions, answer directly without calling a tool.
dont introduce your self or ask that you are here to answer about weather question,if user ask for weather question, use the tool to get the answer,otherwise answer the general question correclty.
Never invent weather data.
"""
def run_agent(question: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    max_iterations = 5

    for step in range(max_iterations):
        print(f"\nAgent loop iteration: {step + 1}")

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            if isinstance(response.content, str):
                return response.content

            return str(response.content)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name not in tool_map:
                observation = {"error": f"Unknown tool: {tool_name}"}
            else:
                observation = tool_map[tool_name].invoke(tool_args)

            print(f"Tool called: {tool_name}")
            print(f"Tool result: {observation}")

            messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call_id,
                )
            )

    return "The agent reached its maximum number of iterations."


if __name__ == "__main__":
    print("LangChain Weather Agent")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            answer = run_agent(question)
            print(f"\nAssistant: {answer}")

        except Exception as error:
            print(f"\nError: {error}")



