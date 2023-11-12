import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os.path

# Read the CSV file
output_dir_unique = os.path.join(os.path.dirname(os.path.abspath(__file__ or '.')), "Semantic Inconsistencies in Dataset")
file =os.path.join(output_dir_unique, 'inconsistenciesWeFixed.txt')
df = pd.read_csv(file )

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges based on the DataFrame
for index, row in df.iterrows():
    G.add_node(row['Artist'])
    G.add_node(row['Album'])
    G.add_edge(row['Artist'], row['Album'], label=row['Similar_Incorect_Albums'])

# Plot the graph

pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, 'label')
nx.draw(G, pos, with_labels=True, font_size=7, node_size=400, node_color="c")
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='black', font_size=4)

plt.show()