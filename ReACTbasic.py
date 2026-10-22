import re
import requests

MODEL_NAME = "qwen3.5:0.8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

CITIES = {
    "islamabad": {"latitude": 33.6844, "longitude": 73.0479},
    "rawalpindi": {"latitude": 33.5651, "longitude": 73.0169},
    "lahore": {"latitude": 31.5204, "longitude": 74.3587},
    "sialkot": {"latitude": 32.4945, "longitude": 74.5229},
    "multan": {"latitude": 30.1575, "longitude": 71.5249},
    "karachi": {"latitude": 24.8607, "longitude": 67.0011},
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
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

REACT_PROMPT = """
You are a helpful assistant.

For current weather questions, use the weather tool.

Use this exact format:
Thought: I need current weather data for the requested city.
Action: get_current_weather
Action Input: CityName

Only use supported cities.
Do not provide a final answer before receiving an Observation.

For general questions unrelated to current weather, answer the
actual question directly without using a tool.
Do not describe yourself unless asked.

General answer format:
Final Answer: Your direct answer

Supported cities:
Islamabad, Rawalpindi, Lahore, Sialkot, Multan, Karachi.

Return one Action or one Final Answer at a time.
Do not use Markdown code fences.
""".strip()


def get_current_weather(city: str) -> dict:
    city_key = city.strip().lower()

    if city_key not in CITIES:
        return {"error": f"Unsupported city: {city}"}

    params = {
        "latitude": CITIES[city_key]["latitude"],
        "longitude": CITIES[city_key]["longitude"],
        "current": (
            "temperature_2m,relative_humidity_2m,"
            "apparent_temperature,precipitation,"
            "weather_code,wind_speed_10m"
        ),
        "timezone": "Asia/Karachi",
    }

    try:
        response = requests.get(WEATHER_URL, params=params, timeout=20)
        response.raise_for_status()
        current = response.json().get("current")

        if not isinstance(current, dict):
            return {"error": f"No current weather data for {city.title()}."}

        weather_code = current.get("weather_code")

        return {
            "city": city_key.title(),
            "time": current.get("time", "Not available"),
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "precipitation": current.get("precipitation"),
            "condition": WEATHER_CODES.get(
                weather_code, "Unknown weather condition"
            ),
        }

    except (requests.RequestException, ValueError) as error:
        return {"error": f"Unable to retrieve weather: {error}"}


def format_weather(data: dict) -> str:
    if "error" in data:
        return data["error"]

    def show(value, unit=""):
        if value is None:
            return "Not available"
        return f"{value}{unit}"

    return (
        f"City: {data['city']}\n"
        f"Observation time: {data['time']}\n"
        f"Temperature: {show(data['temperature'], ' °C')}\n"
        f"Feels like: {show(data['feels_like'], ' °C')}\n"
        f"Humidity: {show(data['humidity'], '%')}\n"
        f"Wind speed: {show(data['wind_speed'], ' km/h')}\n"
        f"Precipitation: {show(data['precipitation'], ' mm')}\n"
        f"Condition: {data['condition']}"
    )


def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 512
            },
        },
        timeout=300,
    )

    response.raise_for_status()
    output = response.json().get("response", "").strip()

    if not output:
        raise RuntimeError("Ollama returned an empty response.")

    print("\nModel response:", output)
    return output


def find_requested_cities(question: str) -> list:
    question = question.lower()
    return [
        city for city in CITIES
        if re.search(rf"\b{re.escape(city)}\b", question)
    ]


def is_weather_question(question: str) -> bool:
    terms = (
        "weather", "temperature", "humidity", "wind",
        "rain", "precipitation", "forecast", "feels like",
    )
    return any(term in question.lower() for term in terms)


def chat_with_agent(question: str, history: list) -> str:
    cities = find_requested_cities(question)

    if is_weather_question(question):
        if not cities:
            answer = (
                "Please specify a supported city: "
                + ", ".join(city.title() for city in CITIES)
                + "."
            )
        else:
            observations = []

            for city in cities:
                data = get_current_weather(city)
                observations.append(data)

            answer = "\n\n".join(
                format_weather(data) for data in observations
            )

            valid = [
                data for data in observations
                if "error" not in data
            ]

            temperatures = [
                (data["city"], data["temperature"])
                for data in valid
                if isinstance(data["temperature"], (int, float))
            ]

            if len(temperatures) >= 2:
                warmest = max(temperatures, key=lambda item: item[1])
                coolest = min(temperatures, key=lambda item: item[1])

                answer += (
                    "\n\nComparison:\n"
                    f"Warmest: {warmest[0]} ({warmest[1]} °C)\n"
                    f"Coolest: {coolest[0]} ({coolest[1]} °C)"
                )

        history.append(f"User: {question}\nAssistant: {answer}")
        return answer

    trace = ""

    for _ in range(5):
        prompt = (
            REACT_PROMPT
            + "\n\nConversation history:\n"
            + ("\n".join(history[-6:]) if history else "No previous conversation.")
            + "\n\nCurrent user question:\n"
            + question
            + "\n\nReAct trace:\n"
            + (trace if trace else "No actions taken yet.")
            + "\n\nProvide the next Action or Final Answer."
        )

        model_output = call_ollama(prompt)

        action_match = re.search(
            r"Action:\s*get_current_weather\b",
            model_output,
            re.IGNORECASE,
        )
        input_match = re.search(
            r"Action Input:\s*([^\r\n]+)",
            model_output,
            re.IGNORECASE,
        )
        final_match = re.search(
            r"Final Answer:\s*(.*)",
            model_output,
            re.IGNORECASE | re.DOTALL,
        )

        if action_match:
            if not input_match:
                trace += "\nAction Input is missing. Provide a valid city.\n"
                continue

            city = input_match.group(1).strip().strip("\"'[] ").lower()

            if city not in CITIES:
                trace += (
                    f"\nUnsupported city: {city}. Supported cities: "
                    f"{', '.join(CITIES)}.\n"
                )
                continue

            observation = get_current_weather(city)  # ← fresh API call every time

            trace += (
                f"\n{model_output}\n"
                f"Observation: {format_weather(observation)}\n"
            )
            continue

        if final_match:
            answer = final_match.group(1).strip()

            if not answer:
                trace += "\nThe final answer was empty. Try again.\n"
                continue

            history.append(f"User: {question}\nAssistant: {answer}")
            return answer

        trace += (
            f"\nInvalid response format:\n{model_output}\n"
            "Return a valid Action and Action Input, or a Final Answer.\n"
        )

    answer = "I could not complete the request. Please try again."
    history.append(f"User: {question}\nAssistant: {answer}")
    return answer


if __name__ == "__main__":
    conversation_history = []

    print("ReAct Weather Assistant")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            answer = chat_with_agent(
                question,
                conversation_history,
            )
            print(f"\nAssistant: {answer}\n")

        except requests.Timeout:
            print("\nA request timed out. Check Ollama and your internet connection.\n")

        except requests.RequestException as error:
            print(f"\nConnection error: {error}\n")

        except RuntimeError as error:
            print(f"\nModel error: {error}\n")