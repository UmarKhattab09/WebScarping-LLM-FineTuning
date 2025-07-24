from pyvis.network import Network
import pandas as pd
import networkx as nx

class NewsCategoriesImpact:

    def __init__(self,df):
        self.df=df
    def newscategoriesimpact(self):
        df =self.df.groupby('Impact')['News'].count().reset_index()
        print(df)

    #     # Create graph
        G = nx.Graph()

    #     # Add central node
        G.add_node('News', size=20)  # Adjust central size as needed

    #     # Add edges from 'News' to each impact level with size based on count
        for index, row in df.iterrows():
            impact = row['Impact']
            count = row['News']
            G.add_node(impact, size=count * 5)  # Multiply to make size more visible
            G.add_edge('News', impact)

    #     # Visualize using Pyvis
        net = Network(notebook=True, height="450px", width="100%", bgcolor="#222222", font_color="white",cdn_resources='remote')

        net.from_nx(G)

    #     # Set node sizes in pyvis
        for node in net.nodes:
            node['value'] = G.nodes[node['id']]['size']

    #     # html=net.show("test.html")
        html = net.generate_html()
        html = html.replace("'","\"")
        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms
    allow-scripts allow-same-origin allow-popups
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
        
        return output_html







df = pd.read_csv("C:/Users/khatt/Documents/WebScarping-LLM-FineTuning/outputs/5_iMPACT.csv")



test=NewsCategoriesImpact(df)

test.newscategoriesimpact()