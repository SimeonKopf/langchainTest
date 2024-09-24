from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
import warnings
import pandas as pd
import time


warnings.filterwarnings("ignore", message=".*pydantic.error_wrappers.*")

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")






vectorstore = DocArrayInMemorySearch.from_texts(
    ["dogs love to play fetch", "pizza is a popular Italian dish",
    "Elon Musk founded SpaceX", "flowers need sunlight to grow",
    "Python is a programming language"],
    embedding=OllamaEmbeddings(model="llama3.1"),
)
retriever = vectorstore.as_retriever()


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)


output_parser = StrOutputParser()


questions = [
    "What do dogs love to play?",
    "What type of dish is pizza, and which country is it associated with?",
    "Who founded SpaceX?",
    "What do flowers need to grow?",
    "What is Python?",

]

headers = ['question', 'ground_truth', 'answer', 'contexts']
df = pd.DataFrame(columns=headers)

temperatur = [0,0.2,0.4,0.6,0.8,1]
tokens = [100,200,300,400,500]
retries = [1,2,3,4,5]
i=0;

for q in questions:
    print("Iteration" + q )

    for i in range(0,5):
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=temperatur [i],
            max_tokens=tokens[i],
            timeout=None,
            max_retries=retries[i],
        )
        time.sleep(5)
        chain = setup_and_retrieval | prompt | llm | output_parser
        answer = chain.invoke(q)
        df.loc[len(df)] = [q,None, str(answer), str(setup_and_retrieval)]
        time.sleep(3)

df.to_csv("geminiTest.csv", na_rep="Placeholder", index=False)
