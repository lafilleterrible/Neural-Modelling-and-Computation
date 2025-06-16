import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

# Streamlit page config
st.set_page_config(page_title="Alzheimer Brain Simulation", layout="centered")
st.title("🧠 Alzheimer's Disease Neural Network Simulation")

# Background brain image
bg_image_path = "brain_background.png"
if os.path.exists(bg_image_path):
    brain_img = mpimg.imread(bg_image_path)
else:
    brain_img = None

# Generate the brain network
def generate_brain(state):
    G = nx.erdos_renyi_graph(n=10, p=0.5)

    if state == "Healthy":
        for (u, v) in G.edges():
            G[u][v]['weight'] = round(random.uniform(0.7, 1.0), 2)

    elif state == "Early Alzheimer":
        to_remove = random.sample(list(G.edges()), k=int(0.3 * G.number_of_edges()))
        G.remove_edges_from(to_remove)
        for (u, v) in G.edges():
            G[u][v]['weight'] = round(random.uniform(0.3, 0.6), 2)

    elif state == "Advanced Alzheimer":
        to_remove = random.sample(list(G.edges()), k=int(0.6 * G.number_of_edges()))
        G.remove_edges_from(to_remove)
        for (u, v) in G.edges():
            G[u][v]['weight'] = round(random.uniform(0.1, 0.3), 2)

    G.remove_nodes_from(list(nx.isolates(G)))
    return G

# Select brain state
state = st.radio("Select Brain State:", ["Healthy", "Early Alzheimer", "Advanced Alzheimer"])

# Button to regenerate the graph
if st.button("🔁 Regenerate Network"):
    st.rerun()

# Generate and position the network
G = generate_brain(state)
pos = nx.spring_layout(G, seed=42, center=(0.1, 0.0), scale=0.9)

# Select neuron and show its connections
if len(G.nodes()) > 0:
    selected_node = st.selectbox("🔎 Select a neuron", sorted(G.nodes()))
    neighbors = list(G.neighbors(selected_node))
    st.info(f"Neuron {selected_node} is connected to: {neighbors}")

    # Propagate activation: selected node + neighbors
    activated = set([selected_node]) | set(neighbors)
    colors = ['blue' if node in activated else 'lightgray' for node in G.nodes()]
else:
    colors = ['lightgray' for _ in G.nodes()]

# Plot
fig, ax = plt.subplots(figsize=(6, 4))

# Adjusted background image overlay
if brain_img is not None:
    ax.imshow(brain_img, extent=[-1.1, 1.1, -1.0, 1.0], alpha=0.3, aspect='auto')

weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, ax=ax)
nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.viridis,
                       width=2, ax=ax)
nx.draw_networkx_labels(G, pos, font_color='black', ax=ax)
nx.draw_networkx_edge_labels(G, pos,
    edge_labels={(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)},
    font_size=8, ax=ax)

ax.axis('off')
st.pyplot(fig)

# Explanation per condition
if state == "Healthy":
    st.success("Healthy brain: strong and dense connections support efficient signal propagation.")
elif state == "Early Alzheimer":
    st.warning("Early Alzheimer: some neurons lose connections; signal propagation is less effective.")
else:
    st.error("Advanced Alzheimer: disconnected neurons disappear, simulating severe network breakdown.")

# Display network metrics
st.markdown("### 🧠 Network Stats")
st.metric("Number of Neurons", len(G.nodes()))
st.metric("Number of Connections", len(G.edges()))
st.metric("Average Weight", f"{sum(weights)/len(weights):.2f}" if weights else "N/A")
