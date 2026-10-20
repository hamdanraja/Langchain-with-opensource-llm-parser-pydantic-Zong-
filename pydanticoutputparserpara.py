
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel


class TopicInfo(BaseModel):
    topic: str = Field(description="Name of the topic")
    category: str = Field(description="Category of the topic")
    key_facts: list[str] = Field(description="Important facts about the topic")


class TopicSummary(BaseModel):
    summary: str = Field(description="A short summary of the topic")
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
    ("system", "Generate a short summary of the user's topic.\n{format_instructions}"),
    ("human", "{topic}")
]).partial(format_instructions=summary_parser.get_format_instructions())


info_chain = info_prompt | llm | info_parser
summary_chain = summary_prompt | llm | summary_parser


parallel_chain = RunnableParallel(
    info=info_chain,
    summary=summary_chain
)


topic = input("Enter any topic: ").strip()

if topic:
    result = parallel_chain.invoke({"topic": topic})

    print("\nTopic Information:")
    print(result["info"].model_dump_json(indent=2))

    print("\nTopic Summary:")
    print(result["summary"].model_dump_json(indent=2))
else:
    print("Please enter a topic.")
