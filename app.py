import gradio as gr
from webscraping import WebScraping
import pandas as pd
from transformers import pipeline
from newscategory import NewsCategoriesImpact
from newscategory import NewsCategories
import os
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), "./trainingllm/.env")
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv(dotenv_path=env_path)


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
    
def outputdfimpact(count):
    savepath = f"outputs/{count}_iMPACT.csv"
    try:
        df = pd.read_csv(savepath)
        return df
    except FileNotFoundError:
        web = WebScraping(count)
        df = web.slowdataframe()
        df.to_csv(savepath,index=False)
        return df


def trainingllm(text):
    hf_token=os.getenv("huggingfacetoken")

    token = hf_token
    model_id = "UmarKhattab09/llmfinetuning"
    model_name = "UmarKhattab09/llmfinetuning"
    model = AutoModelForSequenceClassification.from_pretrained(model_id, use_auth_token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
    classifier = pipeline(
    "text-classification",
        model=model_name,
        tokenizer=model_name,

    )    
    result = classifier.predict(text)
    PredictedNews=result[0]['label']
    score=result[0]['score']
    return PredictedNews,score


def characternetwork(df):
    Network = NewsCategories(df)
    html = Network.newscategories() 
    return html   

def characternetworkimpact(df):
    Network = NewsCategoriesImpact(df)
    html = Network.newscategoriesimpact() 
    return html   

def load_df_and_graph1(count_range):
    df = outputdf(count_range)
    html = characternetwork(df)
    return df,html


def load_df_and_graph2(count_range):
    df = outputdfimpact(count_range)
    html = characternetworkimpact(df)
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
                count = gr.Button("LoadDataFrame with NewsImpact (Slow)")
                count.click(outputdfimpact,inputs=[Counts],outputs=[dataframe])
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
                gr.HTML('<p> Will Take Time because of the ')
         

        with gr.Row():
            networkplot=gr.HTML()
        range.click(fn=load_df_and_graph1, inputs=[Counts], outputs=[dataframe, networkplot])
        count.click(fn=load_df_and_graph2,inputs=[Counts],outputs=[dataframe,networkplot])
            
          
  
    demo.launch()

if __name__ =="__main__":
    main()

                


