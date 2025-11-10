from langchain_google_genai import ChatGoogleGenerativeAI # there is some issue with langchain_huggingface package currently
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os

from pydantic import Field, BaseModel
load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Corrected model name to "gemini-2.0-flash"
    temperature=0, # It's good practice to set temperature for consistent output
    google_api_key=os.getenv("google_api_key")
)

class Person(BaseModel):
    name: str = Field(description="The name of the fictional person")
    place: str = Field(description="The place where the fictional person lives")
    age: int = Field(gt=18,description="The age of the fictional person")
    
parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
  template='Give me a name, place and age of a fictional {place} person \n{format_instructions}',  # format instructions is a variable that will be replaced by the output parser instructions
  input_variables=["place"],        # this is for user inputs to add to the prompt template, we have 'place' as user input in this case
  partial_variables={"format_instructions": parser.get_format_instructions()}  # this is for variables that we want to fill in the prompt template partially  
)

prompt=template.invoke({"place":"Indian"})  # invoking the prompt template with user input 'place' as 'Indian'
result=model.invoke(prompt)  # invoking the model with the generated prompt
final_result=parser.parse(result.content)  # parsing the model output using the pydantic output parser
print(final_result)  # printing the final parsed output 