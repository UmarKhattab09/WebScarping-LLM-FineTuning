

from pyvis.network import Network
import pandas as pd
import networkx as nx

class NewsCategories:

    def __init__(self,df):
        self.df=df
    def newscategories(self):
        self.df= self.df.groupby('NewsType')['News'].count().reset_index()
        G = nx.from_pandas_edgelist(
        self.df,
        source='News',
        target='NewsType',
        # edge_attr='News',
        create_using=nx.Graph()

    )
        net=Network(notebook=True,width="1000px",height="700px",font_color="black",cdn_resources='remote')
        node_degree =dict(G.degree)
        net.from_nx(G)
        html = net.generate_html()
        html = html.replace("'","\"")
        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms
    allow-scripts allow-same-origin allow-popups
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
        
        return output_html







# df = pd.read_csv("C:/Users/khatt/Documents/WebScarping-LLM-FineTuning/outputs/6.csv")


# test=NewsCategories(df)
# test.newscategories()