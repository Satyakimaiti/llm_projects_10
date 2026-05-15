import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq
import os

# Load environment variables
load_dotenv(override=True)

print("GROQ API KEY LOADED:",
      os.getenv("GROQ_API_KEY") is not None)


class TokenPredictor:

    def __init__(self, model_name: str):

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.model_name = model_name

    def predict_tokens(
        self,
        prompt: str,
        max_tokens: int = 100
    ) -> List[Dict]:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0,
            stream=True
        )

        predictions = []

        for chunk in response:

            try:

                # Skip empty chunks
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Skip chunks without content
                if delta.content is None:
                    continue

                # Split streamed chunk into words
                tokens = delta.content.split()

                for token in tokens:

                    token = token.strip()

                    if token == "":
                        continue

                    predictions.append({
                        "token": token,
                        "probability": 1.0,
                        "alternatives": []
                    })

            except Exception as e:
                print("Chunk Error:", e)

        return predictions


def create_token_graph(
    model_name: str,
    predictions: List[Dict]
) -> nx.DiGraph:

    G = nx.DiGraph()

    # START node
    G.add_node(
        "START",
        token="START",
        color="lightgreen",
        size=3000
    )

    previous_node = "START"

    # Token nodes
    for i, pred in enumerate(predictions):

        token_id = f"t{i}"

        G.add_node(
            token_id,
            token=pred["token"],
            color="skyblue",
            size=2500
        )

        G.add_edge(previous_node, token_id)

        previous_node = token_id

    # END node
    G.add_node(
        "END",
        token="END",
        color="salmon",
        size=3000
    )

    G.add_edge(previous_node, "END")

    return G

def visualize_predictions(
    G: nx.DiGraph,
    figsize=(20, 12)
):

    plt.figure(figsize=figsize)

    pos = {}

    # Layout settings
    max_per_row = 10
    x_spacing = 3
    y_spacing = 3

    nodes = list(G.nodes())

    # Create wrapped layout
    for i, node in enumerate(nodes):

        row = i // max_per_row
        col = i % max_per_row

        pos[node] = (
            col * x_spacing,
            -row * y_spacing
        )

    # Colors
    node_colors = [
        G.nodes[node]["color"]
        for node in G.nodes()
    ]

    # Dynamic sizes
    node_sizes = []

    for node in G.nodes():

        token = G.nodes[node]["token"]

        size = 1800 + len(token) * 80

        node_sizes.append(size)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9
    )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=12,
        width=1.5,
        alpha=0.7
    )

    # Labels
    labels = {}

    for node in G.nodes():

        token = G.nodes[node]["token"]

        # Shorten long words
        if len(token) > 15:
            token = token[:12] + "..."

        labels[node] = token

    nx.draw_networkx_labels(
        G,
        pos,
        labels,
        font_size=9,
        font_weight="bold"
    )

    plt.title(
        "Groq Token Flow Visualization",
        fontsize=18,
        fontweight="bold"
    )

    plt.axis("off")

    plt.tight_layout()

    return plt