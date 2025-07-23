

import pandas as pd
from bs4 import BeautifulSoup
import requests
class WebScraping:
    def __init__(self,count):
        self.count = count    
        self.url_dict = {
            'politics':f"https://www.npr.org/get/1014/render/partial/next?start={1}&count={self.count}",
            'science' : f"https://www.npr.org/get/1007/render/partial/next?start={1}&count={self.count}",
            'health'  : f"https://www.npr.org/get/1128/render/partial/next?start={1}&count={self.count}",
            'business': f"https://www.npr.org/get/1006/render/partial/next?start={1}&count={self.count}",
            'climate' : f"https://www.npr.org/get/1167/render/partial/next?start={1}&count={self.count}" #
        }


    def url_extractor(self,url):
        response=requests.get(url)
        status = response.status_code
        html_content = response.text
        soup = BeautifulSoup(html_content,'html.parser')
        return soup
    
    def extractinglinks(self,url):
        data=[]
        for article in url.find_all('article'):
            anchortag=article.find('a')
            try:
                articlelink=anchortag['href']
                if articlelink.startswith("https://www.npr.org"):
                    data.append(articlelink)
            except:
                f"no href"
        return data
    
    def storyextractor(self,links):
        separator=' '
        data =[]
        for i in range(len(links)):
            article = self.url_extractor(links[i])
            storydiv=article.find("div",id="storytext")
            try:
                text=storydiv.find_all('p')
            except:
                f"no p"
            for i in range(len(text)):
                try:
                    temp= text[i].get_text(separator= ' ', strip = True)
                except:
                    f"no temp"
                text[i] = temp
            finaltxt= separator.join(text)
            data.append(finaltxt)
        return data
    
    def createDataFrame(self,data):
        rows = []
        for news_type,news_list in data.items():
            for news_item in news_list:
                rows.append((news_type,news_item))

        df = pd.DataFrame(rows,columns=["NewsType","News"])
        df.to_csv()
        return df
             
    def dataframe(self):
        links = {}
        for k,v in self.url_dict.items():
            test = self.url_extractor(v)
            links[k] = test
        extract = {}
        for k,v in links.items():
            extract[k] = self.extractinglinks(v)
        dataframedata = {}
        for k,v in extract.items():
            dataframedata[k] = self.storyextractor(v)

        df = self.createDataFrame(dataframedata)
        df.to_csv(f"outputs/{self.count}.csv")
        return df
    

        
            
# test = WebScraping(count=2)

# x = test.dataframe()
# print(x)
# save = test.outputdf()







