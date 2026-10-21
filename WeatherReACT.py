import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

MODEL_NAME = "qwen3.5:0.8b"

CITIES = {
    "islamabad": {
        "latitude": 33.6844,
        "longitude": 73.0479,
    },
    "rawalpindi": {
        "latitude": 33.5651,
        "longitude": 73.0169,
    },
        "lahore": {
        "latitude": 31.5204	,
		"longitude": 74.3587,
    },
        "sialkot": {
        "latitude": 32.4945,
        "longitude": 74.5229,
    },
        "multan": {
        "latitude": 30.1575,
        "longitude": 71.5249,
    },
        "karachi": {
        "latitude": 24.8607,
        "longitude": 67.0011,
    },
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
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@tool
def get_current_weather(city: str) -> str:
    """Get the current weather for Islamabad, Rawalpindi, Lahore, Sialkot, Multan, or Karachi, Pakistan. Use this tool when the user asks about current weather, temperature, humidity, rain, or wind in one of these cities. The city argument must be a supported city name."""
    city = city.strip().lower()

    if city not in CITIES:
        return (
          "This weather tool supports only Islamabad, Rawalpindi"
           " Lahore, Multan, Sialkot, and Karachi, Pakistan."
        )

    coordinates = CITIES[city]

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "current": (
            "temperature_2m,relative_humidity_2m,"
            "apparent_temperature,is_day,precipitation,"
            "weather_code,wind_speed_10m"
        ),
        "timezone": "Asia/Karachi",
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        current = data["current"]
        weather_code = current["weather_code"]
        condition = WEATHER_CODES.get(
            weather_code,
            f"Weather code {weather_code}",
        )

        return (
            f"Current weather in {city.title()}, Pakistan:\n"
            f"Time: {current['time']} (Pakistan time)\n"
            f"Condition: {condition}\n"
            f"Temperature: {current['temperature_2m']}°C\n"
            f"Feels like: {current['apparent_temperature']}°C\n"
            f"Humidity: {current['relative_humidity_2m']}%\n"
            f"Wind speed: {current['wind_speed_10m']} km/h\n"
            f"Precipitation: {current['precipitation']} mm"
        )

    except (requests.RequestException, KeyError, ValueError) as error:
        return (
            "I could not retrieve current weather data. "
            f"Please try again later. Details: {error}"
        )


llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=[get_current_weather],
    system_prompt=(
        "You are a helpful AI assistant. "
        "Answer general questions using your knowledge. "
        "For current weather, temperature, rain, or weather "
        "conditions in Islamabad, Rawalpindi, Lahore, Multan, Sialkot, or Karachi, call the "
        "get_current_weather tool. Never invent live weather data. "
        "Use the tool result to answer weather questions. "
        "If the user asks about weather in another city, explain "
        "that this weather tool currently supports Islamabad "
        "Rawalpindi, Lahore, Multan, Sialkot, and Karachi. For unrelated questions, answer normally "
        "without calling the weather tool. "
        "Keep answers clear and easy to understand."
    ),
)


def chat_with_agent(question: str) -> str:
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    return str(result["messages"][-1].content)


if __name__ == "__main__":
    print("ReAct Weather Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            answer = chat_with_agent(question)
            print(f"\nAssistant: {answer}\n")

        except Exception as error:
            print(f"\nError: {error}")
            print(
                "Check that Ollama is running, the model exists, "
                "and your installed LangChain packages are compatible.\n"
            )
