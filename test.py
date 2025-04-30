from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceHub(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     model_kwargs={
#         "temperature": 0.1, 
#         "return_full_text": False,
#         "max_new_tokens": 100
#     },
# )

system_prompt = "You are a helpful assistant"
user_prompt = "{input}"

token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>user<|end_header_id|>assistant<|end_header_id|>"

prompt = ChatPromptTemplate.from_messages([
    ("system", token_s + system_prompt),
    ("user", user_prompt + token_e),
])

# chain = prompt | llm

input = "What is AI?"

# response = chain.invoke({"input": input})

# print(response)
# print("--------------------------------")

llm = ChatOllama(
    model="phi3",
    temperature=0.1,
)

chain3 = prompt | llm
response = chain3.invoke({"input": input})
print(response.content)
