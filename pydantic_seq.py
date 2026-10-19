
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    "Include its location, capital, and population if known. "
    "Avoid unsupported historical claims."
)

summary_chain = summary_prompt | llm | str_parser

facts_prompt = ChatPromptTemplate.from_template(
    "Give exactly 4 short facts about {country}, using this summary:\n"
    "{summary}\n\n"
    "Separate the four facts using the | character. "
    "Do not use the | character inside a fact. "
    "Do not add numbering or introductory text."
)

facts_chain = facts_prompt | llm | str_parser

info_prompt = ChatPromptTemplate.from_template(
    """Provide information about {country}.

Summary:
{summary}

Facts:
{facts}

Return the capital city, main language, and estimated population.
Use the country named in the question, not a different country.
"""
)

structured_llm = llm.with_structured_output(
    CountryInfo,
    method="json_schema"
)

info_chain = info_prompt | structured_llm

def run_sequential(country):
    print("\nRunning Step 1: Summary")

    summary = summary_chain.invoke({
        "country": country
    })

    print("\nSUMMARY")
    print(summary)

    print("\nRunning Step 2: Facts")

    facts_text = facts_chain.invoke({
        "country": country,
        "summary": summary
    })

    facts = [
        item.strip()
        for item in facts_text.split("|")
        if item.strip()
    ]

    print("\nFACTS")
    for index, fact in enumerate(facts, start=1):
        print(f"{index}. {fact}")

    print("\nRunning Step 3: Pydantic Structured Output")

    info = info_chain.invoke({
        "country": country,
        "summary": summary,
        "facts": "\n".join(facts)
    })

    print("\nPYDANTIC OUTPUT")
    print(info)

    print("\nCAPITAL")
    print(info.capital)

    print("\nLANGUAGE")
    print(info.language)

    print("\nPOPULATION")
    print(info.population)

    print("\nAS DICTIONARY")
    print(info.model_dump())

if __name__ == "__main__":
    country = input("Enter a country: ").strip()

    if not country:
        country = "Japan"

    try:
        run_sequential(country)
    except Exception as error:
        print(f"\nError: {error}")
