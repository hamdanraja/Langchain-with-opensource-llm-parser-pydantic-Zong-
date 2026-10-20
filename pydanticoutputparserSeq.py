
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class TopicInfo(BaseModel):
    topic: str = Field(description="Name of the topic")
    category: str = Field(description="Category of the topic")
    key_facts: list[str] = Field(description="Important facts about the topic")


class TopicSummary(BaseModel):
    summary: str = Field(description="A short summary based on the extracted information")
    keywords: list[str] = Field(description="Important keywords")
    conclusion: str = Field(description="A short conclusion")


llm = ChatOllama(model="qwen3.5:0.8b", temperature=0, reasoning=False)

info_parser = PydanticOutputParser(pydantic_object=TopicInfo)
summary_parser = PydanticOutputParser(pydantic_object=TopicSummary)


info_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract information about the topic.\n{format_instructions}"),
    ("human", "{topic}")
]).partial(format_instructions=info_parser.get_format_instructions())


summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the supplied information. Do not ignore the supplied facts.\n{format_instructions}"),
    ("human", "Extracted information:\n{information}")
]).partial(format_instructions=summary_parser.get_format_instructions())


info_chain = info_prompt | llm | info_parser
summary_chain = summary_prompt | llm | summary_parser

sequential_chain = (
    info_chain
    | (lambda info: {
        "information": info.model_dump_json(),
        "topic_info": info
    })
    | {
        "topic_info": lambda data: data["topic_info"],
        "summary": summary_chain
    }
)

topic = input("Enter any topic: ").strip()

if topic:
    result = sequential_chain.invoke({"topic": topic})

    print("\nTopic Information:")
    print(result["topic_info"].model_dump_json(indent=2))

    print("\nTopic Summary:")
    print(result["summary"].model_dump_json(indent=2))
else:
    print("Please enter a topic.")