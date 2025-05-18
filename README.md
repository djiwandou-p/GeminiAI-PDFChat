# Chat with PDF 
This Repo implements chat with your PDF via a GUI. This Code utilized Gemini AI'ss LLM and Embedding models for information retreival from your documents. 


![image](https://github.com/user-attachments/assets/ccb29357-5cf9-4f3f-84cb-29bc0e4354aa)
(tested on Windows 11, using wsl, run on 19-May-2025)


## Clone the Repo:
Clone the repository. 
```shell
git clone https://github.com/djiwandou/GeminiAI-PDFChat.git
```

## Environment Setup
In order to set your environment up to run the code here, first install all requirements:
tested in Windows 11 using WSL
* init venv
```shell
python3 -m venv venv
```

* activate venv
```shell
source venv/bin/activate
```

* install requirements 
```shell
pip install -r requirements.txt
```

## GeminiAI API Key 

You will need the PaLM API key to run this, get your Gemini AI API key from [here](https://aistudio.google.com/apikey) (Free!)
In the `.env` set your API key. 

```shell
GOOGLE_API_KEY=
```

## Run the WebApp:

```shell
streamlit run PaLM_ChatPDF.py
```
