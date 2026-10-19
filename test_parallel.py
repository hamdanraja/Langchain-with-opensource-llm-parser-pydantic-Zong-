
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser
)
from langchain_core.runnables import RunnableParallel

llm = ChatOllama(
    model="qwen3.5:0.8b",
    temperature=0.2,
    num_predict=300,
    reasoning= False
)

str_parser = StrOutputParser()
list_parser = CommaSeparatedListOutputParser()
json_parser = JsonOutputParser()

str_prompt = ChatPromptTemplate.from_template(
    "Give a one-paragraph summary about {country}."
)

str_chain = str_prompt | llm | str_parser

list_prompt = ChatPromptTemplate.from_template(
    "List exactly 4 famous things about {country}. "
    "Return only the items as comma-separated values. "
    "Do not use a numbered list or add explanations."
)

list_chain = list_prompt | llm | list_parser

json_prompt = ChatPromptTemplate.from_template(
    """Give information about {country}.
Return only a valid JSON object with these three string fields:
{{"capital": "...", "language": "...", "population": "..."}}
Do not include Markdown code fences or text outside the JSON."""
)

json_chain = json_prompt | llm | json_parser

parallel_chain = RunnableParallel(
    summary=str_chain,
    facts=list_chain,
    info=json_chain
)

if __name__ == "__main__":
    country = input("Enter a country: ").strip()

    if not country:
        country = "Japan"

    result = parallel_chain.invoke({"country": country})

    print("\nSUMMARY (StrOutputParser)")
    print(result["summary"])

    print("\nFACTS (CommaSeparatedListOutputParser)")
    print(result["facts"])

    print("\nSTRUCTURED INFO (JsonOutputParser)")
    print(result["info"])
