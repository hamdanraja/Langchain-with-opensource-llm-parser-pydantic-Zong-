
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser
)

llm = ChatOllama(
    model="qwen3.5:0.8b",
    temperature=0.2,
    num_predict=300,
    reasoning=False
)

str_parser = StrOutputParser()
list_parser = CommaSeparatedListOutputParser()
json_parser = JsonOutputParser()

summary_prompt = ChatPromptTemplate.from_template(
    "Give a one-paragraph summary about {country}."
)

summary_chain = summary_prompt | llm | str_parser

facts_prompt = ChatPromptTemplate.from_template(
    "Using this summary:\n{summary}\n\n"
    "List exactly 4 important facts about {country}. "
    "Return only comma-separated values without numbering or explanations."
)

facts_chain = facts_prompt | llm | list_parser

json_prompt = ChatPromptTemplate.from_template(
    """Using the following information:

Summary: {summary}
Facts: {facts}

Return only a valid JSON object with these three string fields:
{{"capital": "...", "language": "...", "population": "..."}}
Do not include Markdown code fences or text outside the JSON."""
)

json_chain = json_prompt | llm | json_parser

def run_sequential(country):
    summary = summary_chain.invoke({"country": country})
    print("\nSUMMARY")
    print(summary)

    facts = facts_chain.invoke({
        "country": country,
        "summary": summary
    })
    print("\nFACTS")
    print(facts)

    facts_text = ", ".join(facts)

    info = json_chain.invoke({
        "country": country,
        "summary": summary,
        "facts": facts_text
    })
    print("\nSTRUCTURED JSON DATA")
    print(info)

if __name__ == "__main__":
    country = input("Enter a country: ").strip()

    if not country:
        country = "Japan"

    run_sequential(country)
