import sys
import os
from huggingface_hub import login
import huggingface_hub
import pandas as pd
from sklearn import preprocessing
from transformers import pipeline
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments,Trainer
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split

"""
The training.py is not working properly. It works till TokenizedDataset, will fix if given chance
You can train it in trainingllm.ipynb. Increase EPOCHS, DATASET. OTHER HYPER PARAMETERS !!

"""










class LLM:
    model_name="meta-llama/Llama-3.2-1B"

    def __init__(self,df):
        self.model = self.load_model()
        self.df = df
        #TOKEN
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # REQUIRED
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.save_pretrained("tokenizer")


 
    
    def load_model(self):
        hf_token = "TOKEN"
        huggingface_hub.login(hf_token)
        
    
    def datasetmodiications(self):
        le = preprocessing.LabelEncoder()
        le.fit(df['NewsType'].tolist())
        self.df['label']=le.transform(self.df['NewsType'].tolist())
        train_df,test_df=train_test_split(self.df,test_size=0.2,random_state=42)
        df_train=train_df[['label','News']]
        df_test=test_df[['label','News']]
        return df_train,df_test
    
    def compute_metrics(self,eval_pred):
        logits,labels=eval_pred
        predictions=np.argmax(logits,axis=-1)
        return metric.compute(predictions=predictions,references=labels)
    
    def preprocessing(self,examples):
        return self.tokenizer(examples['News'],truncation=True)
    
    def llmstuff(self,df):
        df_train,df_test = self.datasetmodiications()
        dataset_train=Dataset.from_pandas(df_train)
        dataset_test=Dataset.from_pandas(df_test)
        print("Completed Dataset")
        print("_______________________________")
        #Token
        #tokenizer=AutoTokenizer.from_pretrained(self.model_name) #imp
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
        # self.tokenizer.save_pretrained("tokenizer")
        print("Completed Token")
        print("_______________________________")
        
        #Tokenized Dataset
        tokenized_train=dataset_train.map(self.preprocessing,batched=True)
        tokenized_test=dataset_test.map(self.preprocessing,batched=True)
        print("Tokenized Dataset")
        print("_______________________________")
        
        #INITIALIZING 
        self.model=AutoModelForSequenceClassification.from_pretrained(self.model_name,num_labels=5) #NUMLABELS IS THE LABELS WE NEED TO DETECT
        self.model.config.pad_token_id=model.config.eos_token_id
        numberoflayers=0
        for param in model.base_model.parameters():
            numberoflayers+=1
        layersnumber=0
        for param in model.base_model.parameters():
            if layersnumber>numberoflayers-25:
                break
            Layersnumber+=1
            param.requires_grad=False
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        metric=evaluate.load("accuracy")
        
        training_args= TrainingArguments(
            output_dir="../outputs",
            learning_rate=2e-5,
            num_train_epochs=5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            weight_decay=0.01,
            fp16=True,
            report_to="none",
            save_steps=2000)
        print("Initialized Completed")
        print("_______________________________")
        
        
        trainer=Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics)






        trainer.train()
        trainer.save_model("/models/news_classifier_model/")
        tokenizer.save_pretrained("/models/news_classifer_model/")

        
df = pd.read_csv("C:/Users/khatt/Documents/WebScarping-LLM-FineTuning/outputs/2.csv",index_col=0)

test = LLM(df)
# x,y= test.datasetmodiications()
z = test.llmstuff(df)
print()
