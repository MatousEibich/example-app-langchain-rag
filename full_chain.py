import os

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from basic_chain import get_model
from filter import ensemble_retriever_from_docs
from local_loader import load_txt_files
from memory import create_memory_chain
from rag_chain import make_rag_chain


def create_full_chain(retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
    model = get_model("ChatGPT", openai_api_key=openai_api_key)
    system_prompt = """
    Jsi asistent pro odpovídání na otázky nad technickým manuálem pro HR systém OKBase.
    Předpokládej, že všechny dotazy se týkají tohoto systému. Pokud je jasné, že se systému netýkají, odmítni odpovědět.
    Níže jsou části dokumentace získané pomocí Retrieval metod.
    Tvým prvním úkolem je rozhodnout, které části dokumentace jsou relevantní.
    Ty, které vyhodnotíš jako irelevantní dále nezohledňujeme.
    Při vytváření odpovědi musíš spojit relevantní části dohromady, pokud je to možné, a nevytvářet žádné nové formulace – zachovej původní text.
    Pokud relevantní části obsahují odkazy na screenshoty, přidej tyto odkazy na konec odpovědi.
    Je ale důležité, aby se jednalo pouze o screenshoty z relevantních částí, irelevantní screenshoty nás samozřejmě nezajímají.
    Pokud je dotaz nejednoznačný nebo moc obecný, polož objasňující otázky pro upřesnění dotazu, než odpovíš.
    Příkladem může být například dotaz 'Jak zadám adresu?' - tento dotaz je příliš obecný, protože adresa se do HR systému OKBase zadává na sposutě různých míst a je tedy potřeba ujasnit, co přesně uživatel myslí.
    
    Získané části dokumentace:
    {context}
    
    VŽDY ODPOVÍDEJ V ČESKÉM JAZYCE.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
    chain = create_memory_chain(model, rag_chain, chat_memory)
    return chain


def ask_question(chain, query):
    response = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )
    return response


def main():
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()

    docs = load_txt_files()
    ensemble_retriever = ensemble_retriever_from_docs(docs)
    chain = create_full_chain(ensemble_retriever)

    queries = [
        "Generate a grocery list for my family meal plan for the next week(following 7 days). Prefer local, in-season ingredients."
        "Create a list of estimated calorie counts and grams of carbohydrates for each meal."
    ]

    for query in queries:
        response = ask_question(chain, query)
        console.print(Markdown(response.content))


if __name__ == '__main__':
    # this is to quiet parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
