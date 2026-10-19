
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOllama(
    model="qwen3.5:0.8b",
    temperature=0.2,   
    num_predict=400,
    reasoning=False
)

class CountryInfo(BaseModel):
    capital: str = Field(description="Capital city of the country")
    language: str = Field(description="Main language of the country")
    population: str = Field(description="Estimated population of the country")

str_parser = StrOutputParser()

summary_prompt = ChatPromptTemplate.from_template(
    "Give a concise, factual paragraph about {country}. "
    "Include its location, capital, and population if known."
)

summary_chain = summary_prompt | llm | str_parser

facts_prompt = ChatPromptTemplate.from_template(
    "Give exactly 4 short facts about {country}. "
    "Separate the facts using the | character. "
    "Do not use the | character inside a fact. "
    "Do not add numbering or introductory text."
)

facts_chain = facts_prompt | llm | str_parser

info_prompt = ChatPromptTemplate.from_template(
    """Provide information about {country}.

Return the capital city, main language, and estimated population.
Use the country named in the question.
"""
)

structured_llm = llm.with_structured_output(
    CountryInfo,
    method="json_schema"
)

info_chain = info_prompt | structured_llm

parallel_chain = RunnableParallel(
    summary=summary_chain,
    facts=facts_chain,
    info=info_chain
)

def run_parallel(country):
    print("\nRunning all three chains in parallel...")

    result = parallel_chain.invoke({
        "country": country
    })

    print("\nSUMMARY")
    print(result["summary"])

    facts = [
        item.strip()
        for item in result["facts"].split("|")
        if item.strip()
    ]

    print("\nFACTS")
    for index, fact in enumerate(facts, start=1):
        print(f"{index}. {fact}")

    print("\nPYDANTIC OUTPUT")
    print(result["info"])

    print("\nCAPITAL")
    print(result["info"].capital)

    print("\nLANGUAGE")
    print(result["info"].language)

    print("\nPOPULATION")
    print(result["info"].population)

    print("\nAS DICTIONARY")
    print(result["info"].model_dump())

if __name__ == "__main__":
    country = input("Enter a country: ").strip()

    if not country:
        country = "Japan"

    try:
        run_parallel(country)
    except Exception as error:
        print(f"\nError: {error}")
