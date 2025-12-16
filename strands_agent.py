import streamlit as st
import requests
from datetime import datetime
from strands import Agent
from strands.models import BedrockModel
from strands.tools.decorator import tool

# =========================
# CONFIG
# =========================
OPENWEATHER_API_KEY = "2c3bb2d8515973d551473828e22d620c"

# =========================
# TOOL CALL TRACKER
# =========================
tool_calls_log = []

# =========================
# TOOL 1: WEATHER TOOL
# =========================
@tool
def get_weather(city: str) -> str:
    """
    Get current weather details for a given city.
    """
    tool_calls_log.append({
        "tool": "get_weather",
        "params": {"city": city}
    })

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }

        res = requests.get(url, params=params, timeout=10)
        print("Status:", res.status_code)
        print("Response:", res.text)

        res.raise_for_status()
        data = res.json()

        temp = data["main"]["temp"]
        condition = data["weather"][0]["description"]
        rain = data.get("rain", {}).get("1h", 0)

        return (
            f"Weather in {city}:\n"
            f"- Temperature: {temp}°C\n"
            f"- Condition: {condition}\n"
            f"- Rain (last 1h): {rain} mm"
        )

    except Exception as e:
        return f"Error fetching weather data: {e}"


# =========================
# TOOL 2: TIME TOOL
# =========================
@tool
def get_current_time() -> str:
    """
    Get the current local time.
    """
    tool_calls_log.append({
        "tool": "get_current_time",
        "params": {}
    })

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =========================
# AGENT PROMPT
# =========================
SYSTEM_PROMPT = """
You are a helpful AI agent.

You can use tools like get_weather and get_current_time
to answer user questions using live data.

Be concise, practical, and conversational.
"""

# Configure Bedrock Claude model
model = BedrockModel(
    model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1"
)

# =========================
# INITIALIZE STRANDS AGENT
# =========================
agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather, get_current_time],
    model=model
)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Strands Agent Demo",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Strands Agent — Weather & Time Demo")
st.markdown(
    "Ask things like:\n"
    "- *What's the weather in Chennai?*\n"
    "- *Should I carry an umbrella today?*\n"
    "- *What time is it now?*"
)

# -------------------------
# CHAT STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# USER INPUT
# -------------------------
if prompt := st.chat_input("Ask me something..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    tool_calls_log.clear()

    try:
        response = agent(prompt)
    except Exception as e:
        response = f"⚠️ Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(response)

        if tool_calls_log:
            with st.expander("🔧 Tool Calls"):
                for i, call in enumerate(tool_calls_log, 1):
                    st.write(f"**Call {i}: {call['tool']}**")
                    st.json(call["params"])

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
