#from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint  
# we are using hugging face's opensource models as they usually give unstructured text outputs than the google models so that we can demonstrate output parsers better 
from html import parser
from langchain_google_genai import ChatGoogleGenerativeAI # there is some issue with langchain_huggingface package currently
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JSOOutputParser
import os
load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Corrected model name to "gemini-2.0-flash"
    temperature=0, # It's good practice to set temperature for consistent output
    google_api_key=os.getenv("google_api_key")
)

template=PromptTemplate(
  template="Give me a name, place and age of a fictional person \n{format_instructions}",  # format instructions is a variable that will be replaced by the output parser instructions
  input_variables=[],        # this is for user inputs to add to the prompt template, we don't have any user inputs in this case
  partial_variables={"format_instructions": parser.get_format_instructions()}  # this is for variables that we want to fill in the prompt template partially
  
  # why partial variables?
  # because it is used to fill in variables in the prompt template that are not user inputs but are needed for the prompt to work
  # in this case, we are filling in the format_instructions variable in the which is used to tell the model how to format the output
  
  
  # we are passing the format instructions from the output parser to the prompt template
  # input variables is empty as we don't have any user input in this case
  # partial variables is used to pass the format instructions to the prompt template i.e. we are filling in the format_instructions variable in the template with the output parser instructions
)

chain= template | model | parser  # chaining the prompt template, model and output parser together
result=chain.invoke({})  # invoking the chain with empty input as we don't have any user input
print(result)  # printing the final parsed output

# JSON output parser is useful when we want to extract structured data from the model's output
# But it doenst allow for much flexibility in the output format
# So if the model output is not exactly in the expected format, it will throw an error # we can build custom parsers to handle such cases