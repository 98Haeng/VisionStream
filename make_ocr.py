import autogen
import os
import json
import pickle
import openai
import pandas as pd
import smtplib
from a2oclient.aiapi import AISvcConnInfo
from langchain_community.chat_models import ChatOpenAI
from PIL import Image

def extractor(filename):
    # model = 'yolonas-service'
    with open(filename, 'rb') as f:
        img = f.read()
        
        coninfo = AISvcConnInfo("http://aiio.gridone.net:18080", "yolonas-service", apiKey="")
        input = {
            # 'ocr_svc':model,
            # 'ocr-type':'WORD',
            'items': [ {"id": filename, 
                      "image": img}
            ]        
        }
    p_input = pickle.dumps(input)
    output = coninfo.PostPickle("predict",p_input)
    output = pickle.loads(output)
    # output = output[0][3]
    # output = output[0]
    # print(output)
    # return output
    return output

output = extractor("newData/test/img3.jpg")
print(output)
