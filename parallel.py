
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model = ChatOllama(
    model="qwen3.5:0.8b",
    temperature=0.2,
    num_predict=-1,
    reasoning=False
)

parser = StrOutputParser()

summary_prompt = ChatPromptTemplate.from_template(
    "Explain the topic {topic} in simple English in 3 sentences."
)

benefits_prompt = ChatPromptTemplate.from_template(
    "List 3 important benefits of {topic} in simple English."
)

risks_prompt = ChatPromptTemplate.from_template(
    "List 3 important risks or challenges of {topic} in simple English."
)

summary_chain = summary_prompt | model | parser 
benefits_chain = benefits_prompt | model | parser
risks_chain = risks_prompt | model | parser

parallel_chain = RunnableParallel(
    summary=summary_chain,
    benefits=benefits_chain,
    risks=risks_chain
)

topic = input("Enter a topic: ")

result = parallel_chain.invoke({"topic": topic})

for section, answer in result.items():
    print(f"\n{section.upper()}")
    print(answer)
