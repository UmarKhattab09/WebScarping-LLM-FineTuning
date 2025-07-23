import gradio as gr
from webscraping import WebScraping
import pandas as pd
from transformers import pipeline
from newscategory import NewsCategories


def outputdf(count):
    savepath = f"outputs/{count}.csv"
    try:
        df = pd.read_csv(savepath)
        return df
    except FileNotFoundError:
        web = WebScraping(count)
        df = web.dataframe()
        df.to_csv(savepath,index=False)
        return df

def trainingllm(text):
    classifier=pipeline("text-classification",model="C:/Users/khatt/Documents/WebScarping-LLM-FineTuning/models",tokenizer="C:/Users/khatt/Documents/WebScarping-LLM-FineTuning/models")
    result = classifier.predict(text)
    PredictedNews=result[0]['label']
    score=result[0]['score']
    return PredictedNews,score


def characternetwork(df):
    Network = NewsCategories(df)
    html = Network.newscategories() 
    return html   


def load_df_and_graph(count_range):
    df = outputdf(count_range)
    html = characternetwork(df)
    return df,html

def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.HTML('<h1> WEB SCRAPING OF www.npr.org. LLM WITH FINE TUNING.')

        with gr.Row():
            with gr.Column():
                dataframe = gr.Dataframe(headers=["NewsType","News"],datatype=["str","str"])
            
            with gr.Column():  
                Counts = gr.Textbox(label="Range To Train LLM ON")  
                range = gr.Button("Load DataFrame")
                range.click(outputdf,inputs=[Counts],outputs=[dataframe])

        with gr.Row():
            with gr.Column():
                gr.HTML('<h1> Predicting a News Category.</h1>')
                gr.HTML('<p> This Model is trained on very small dataset, around 5 per category and 5 epochs. THere is a trainingllm folder where you can train it. The class is not wokring however you can train it on trainingllm.ipynb. Training takes a lot of time.</p>')

        with gr.Row():
            with gr.Column():
                News = gr.Textbox(label="Enter a News ")
                Getcategory = gr.Button("Get Category")       
            
        with gr.Column():  
                NewsCategory = gr.Textbox()
                Confidence = gr.Textbox()
        Getcategory.click(trainingllm,inputs=[News],outputs=[NewsCategory,Confidence])




        with gr.Row():
            with gr.Column():
                gr.HTML('<h1>Categories Implemented</h1>')
                gr.HTML('<p> Also, I am gonna add weights basically the LLM can learn from the urgency/importance of the News and add weights which will be displayed as a graph. Will Work Soon.</p>')

         

        with gr.Row():
            networkplot=gr.HTML()
        range.click(fn=load_df_and_graph, inputs=[Counts], outputs=[dataframe, networkplot])
            
          
  
    demo.launch()

if __name__ =="__main__":
    main()

                


