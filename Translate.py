from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()


llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

def generate_translation(language, text):
    translation_prompt = PromptTemplate(
        input_variables=['language', 'text'],
        template="Translate {text} to {language}"
    )

    chain = LLMChain(llm=llm, prompt=translation_prompt, output_key="translatedText")
    response = chain({'language': language, 'text': text})
    return response

if __name__ == "__main__":
    print(generate_translation("French","Hello"))

