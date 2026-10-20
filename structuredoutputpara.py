
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel


class TopicInfo(BaseModel):
    topic_name: str = Field(description="Name of the topic")
    category: str = Field(description="Category of the topic")
    key_fact: str = Field(description="Most important fact")
    origin: str = Field(description="Origin or background")
    notable_for: list[str] = Field(description="Notable features")


class TopicSummary(BaseModel):
    one_liner: str = Field(description="One-sentence summary")
    keywords: list[str] = Field(description="Important keywords")
    fun_fact: str = Field(description="An interesting fact")
    verdict: str = Field(description="Overall impression in 10 words or fewer")




llm = ChatOllama(model="qwen3.5:0.8b", temperature=0, reasoning=False)

info_llm = llm.with_structured_output(TopicInfo)
summary_llm = llm.with_structured_output(TopicSummary)


info_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract structured information about the user's topic."),
    ("human", "{text}")
])

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Create a concise summary of the user's topic."),
    ("human", "{text}")
])


info_chain = info_prompt | info_llm
summary_chain = summary_prompt | summary_llm


parallel_chain = RunnableParallel(
    info=info_chain,
    summary=summary_chain
)


topic = input("Enter a topic: ").strip()

if topic:
    result = parallel_chain.invoke({"text": topic})

    print("\nTopic Information:")
    print(result["info"].model_dump_json(indent=2))

    print("\nTopic Summary:")
    print(result["summary"].model_dump_json(indent=2))
else:
    print("Please enter a topic.")
