from agent import Agent


class Game:
    def __init__(self):
        self.agents = [
            Agent("Alex", "gemini", "gemini-1.5-flash-latest"),
            Agent("Blake", "gemini", "gemini-1.5-pro-latest"),
            Agent("Charlie", "groq", "llama-3.1-70b-versatile"),
            Agent("Drew", "groq", "llama-3.1-8b-instant"),
        ]

        self.turn = 0
        self.messages = []
        self.last_actions = []

    def get_state(self):
        return {
            "players": [
                {"name": a.name, "coins": a.coins}
                for a in self.agents
            ],
            "recent_messages": self.messages[-5:]
        }

    def run_turn(self):
        self.turn += 1
        state = self.get_state()
        actions = []

        # Step 1: collect decisions
        for agent in self.agents:
            decision = agent.decide(state)
            actions.append((agent, decision))

        self.last_actions = actions

        # Step 2: apply actions
        for agent, decision in actions:
            action = decision.get("action")
            target_name = decision.get("target")
            message = decision.get("message")

            # Store message
            if message and message != "...":
                self.messages.append({
                    "from": agent.name,
                    "text": message
                })

            # Handle coin transfer
            if action == "give" and target_name:
                target = next(
                    (a for a in self.agents if a.name == target_name),
                    None
                )

                if target and target != agent and agent.coins > 0:
                    agent.coins -= 1
                    target.coins += 1