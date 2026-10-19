from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = ChatOllama(
    model="qwen3.5:0.8b",
    temperature=0.2,
    num_predict=-1,
    reasoning=False
)

parser = StrOutputParser()

summary_prompt = ChatPromptTemplate.from_template(
    "Analyze the topic {topic} and write a short, simple summary."
)

recommendation_prompt = ChatPromptTemplate.from_template(
    "Based on this analysis, suggest 3 practical recommendations.\n\n"
    "Topic: {topic}\n"
    "Analysis: {analysis}"
)

summary_chain = summary_prompt | model | parser 
recommendation_chain = recommendation_prompt | model | parser

sequential_chain = (
    RunnablePassthrough.assign(analysis = summary_chain)  
    | RunnablePassthrough.assign(recommendation = recommendation_chain)  
)

topic = input("Enter a topic: ")

result = sequential_chain.invoke({"topic": topic})


print("\n SUMMARY (Step 1)")

print(result["analysis"])

print("\n RECOMMENDATIONS (Step 2)")
print(result["recommendation"])