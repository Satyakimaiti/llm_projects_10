import streamlit as st
from game_engine import Game

st.set_page_config(layout="wide")

# Initialize game
if "game" not in st.session_state:
    st.session_state.game = Game()

game = st.session_state.game

st.title("🔥 LLM Arena (Gemini vs Groq)")

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Run Turn"):
        game.run_turn()

with col2:
    if st.button("🔄 Reset"):
        st.session_state.game = Game()

st.subheader(f"Turn: {game.turn}")

# Player cards
cols = st.columns(4)

for i, agent in enumerate(game.agents):
    with cols[i]:
        st.markdown(f"### {agent.name}")
        st.write(f"Model: {agent.model}")
        st.write(f"Coins: {agent.coins}")

# Show decisions (DEBUG VIEW)
st.subheader("🧠 Decisions")

if game.last_actions:
    for agent, decision in game.last_actions:
        st.write(f"**{agent.name} → {decision}**")

# Messages
st.subheader("💬 Messages")

for msg in game.messages[-10:]:
    st.write(f"**{msg['from']}**: {msg['text']}")