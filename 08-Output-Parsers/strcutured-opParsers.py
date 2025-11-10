import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
# Corrected Imports: ResponseSchema and StructuredOutputParser are from 'langchain'
from langchain_core.output_parsers import StructuredOutputParser

load_dotenv()

google_api_key = os.getenv("google_api_key")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", # Corrected model name
    temperature=0.7,
    google_api_key=google_api_key
)

# Response schema defines the structure of the output we want from the model
response_schemas = [  # Renamed to avoid conflict with the class name
    ResponseSchema(name='fact-1', description='fact 1 about the topic'),
    ResponseSchema(name='fact-2', description='fact 2 about the topic'),
    ResponseSchema(name='fact-3', description='fact 3 about the topic')
]

# The parser is now correctly imported
parser = StructuredOutputParser.from_response_schemas(response_schemas)

template = PromptTemplate(
  template="Give me 3 interesting facts about the topic of {topic}.\n{format_instructions}",
  input_variables=['topic'],
  partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({'topic': 'space exploration'})

# Chaining is more idiomatic in LangChain now, but invoke works fine
result = model.invoke(prompt)

# Parsing the model output using the structured output parser
final_result = parser.parse(result.content)

print(final_result)


# Structured output parsers are useful when we want the model to return data in a specific structured format
# But its disadvantage is that it can be rigid and may fail if the model output deviates from the expected format
# So we need to ensure that the model is well-instructed to follow the format instructions