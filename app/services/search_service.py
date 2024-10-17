# app/services/search-services.py
from langchain_openai import OpenAI
from langchain_pinecone import PineconeVectorStore 
from langchain_openai import OpenAIEmbeddings
from app.services.vectorization_service import embedder, index
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from app.core.config import settings
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
import logging
from app.services.product_service import get_product_by_slug
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# Crea el VectorStore para Pinecone con LangChain
pinecone_vector_store = PineconeVectorStore(index, embedder, text_key='product_slug')

# Crear el retriever utilizando el adaptador de LangChain
history_aware_retriever = pinecone_vector_store.as_retriever()

# Inicializa el modelo OpenAI
llm = ChatOpenAI(openai_api_key=settings.openai_api_key)

# Create standalone question prompt templates
standalone_question_template = """
Your task is to reformulate the given user question into a standalone question that sounds natural, as if the user is directly asking it. 

Ensure that the standalone question makes sense on its own, without requiring any previous context. Preserve the user's original intent, details, and nuances, but make it clear and self-contained, as if it's the first time the question is being asked.

If the original question is already self-explanatory, refine it slightly to make it sound more natural and conversational.

**Original User Question:** {question}

**Standalone Reformulated Question:** 
"""
standalone_question_prompt = PromptTemplate.from_template(
    standalone_question_template)




answer_template = """You are a helpful sales assistant for Pandorifit, specializing in guiding customers to find the best sports products. Answer the customer's question in a friendly and concise manner, highlighting why one of the products from the list is the best choice.

Consider all the products listed below, and choose the one that best fits the customer's needs based on their question. Explain your reasoning for why this product is the most suitable.

If the customer's question is not about sports products, training, or health advice, politely inform them that you can only assist with Pandorifit's sports products.

Keep your answer brief, friendly, and conversational. If it makes sense, you can suggest another product from the list as an additional recommendation.

**Products List:** 
{products_list}

**Customer question:** {question}  
**Chat History:**  
{chat_history}
**Context:**  
{context}
**Answer:**
"""
# answer_prompt = PromptTemplate.from_template(answer_template)

# Crea el ChatPromptTemplate para manejar el historial de chat
# chat_history_placeholder = MessagesPlaceholder(variable_name="chat_history")

# Store para el historial de la sesión
session_store = {}
def add_message_to_history(chat_history_instance, message_type, message_content):
    """Agrega un mensaje al historial de chat, asegurándose de que no esté duplicado."""
    existing_messages = [msg.content for msg in chat_history_instance.messages if msg.type == message_type]
    if message_content not in existing_messages:
        if message_type == "human":
            chat_history_instance.add_user_message(message_content)
        elif message_type == "ai":
            chat_history_instance.add_ai_message(message_content)

def get_or_create_session_id(session_id: str = None) -> str:
    """
    Obtiene un session_id existente o crea uno nuevo si no se proporciona.

    Args:
        session_id (str): El identificador de la sesión, si se proporciona.

    Returns:
        str: El identificador de la sesión.
    """
    if session_id is None or session_id not in session_store:
        session_id = session_id or "default_session"
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
    return session_id

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]




# Crea el ChatPromptTemplate con historial
contextual_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_template),
        MessagesPlaceholder("chat_history"),  # Agregamos historial de chat como placeholder
        ("human", "{question}"),
    ]
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)





class SearchService:
    
    chat_history: List[Dict[str, str]] = [] 
    
    # print(chat_history)
    
    @staticmethod
    async def generate_standalone_question(question: str) -> str:
        """Generate a standalone question from a user query"""
        prompt_input = {"question": question}
        try:
            chain = LLMChain(llm=llm, prompt=standalone_question_prompt)
            standalone_question = await chain.apredict(**prompt_input)
        except Exception as e:
            logging.error(f"Error generating standalone question: {e}")
            raise
        return standalone_question

    @staticmethod
    async def search_products_by_query(user_query: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Search for products using a standalone question and Pinecone."""
        # Obtener o crear el session_id
        session_id = get_or_create_session_id(session_id)
        chat_history_instance = get_session_history(session_id)
        add_message_to_history(chat_history_instance, "human", user_query)

        # Generar una pregunta autónoma
        standalone_question = await SearchService.generate_standalone_question(user_query)
        print(f"Here is the standalone question: >>> {standalone_question}")

        # Generar embedding para la pregunta autónoma
        embedding = embedder.embed_query(standalone_question)
        results = index.query(vector=embedding, top_k=10)

        # Extraer todos los productos recomendados
        recommended_products = []
        seen_product_slugs = set() 
        for match in results['matches']:
            product_slug = match['id'].split("_")[0]
            product_lang = match['id'].split("_")[1]
                    # Verificamos si el producto ya ha sido procesado
            if product_slug in seen_product_slugs:
                continue  # Saltar si ya hemos visto este producto

            # Añadimos el producto al conjunto para evitar futuros duplicados
            seen_product_slugs.add(product_slug)
            product_details = await get_product_by_slug(product_slug)

            # Acceder a la traducción del producto basada en el idioma
            product_translations = product_details.translations
            selected_translation = product_translations.get(product_lang) or product_translations.get('en', {})

            product_name = selected_translation.name if selected_translation else "Nombre no disponible"
            product_description = selected_translation.description if selected_translation else "Descripción no disponible"

            recommended_products.append({
                "slug": product_slug,
                "lang": product_lang,
                "name": product_name,
                "description": product_description,
                "image_url": product_details.images
            })
            # Salimos del bucle si ya tenemos 5 productos únicos
            if len(recommended_products) >= 5:
                break

        return recommended_products

    @staticmethod
    async def generate_answer(products_list: List[Dict[str, Any]], question: str, session_id: str = None) -> str:
        """Generate a final answer to the user's question based on the list of recommended products."""
    
        # Obtener o crear el session_id
        session_id = get_or_create_session_id(session_id)
        chat_history_instance = get_session_history(session_id)
        

        # Preparar la lista de productos en formato de texto para el prompt
        formatted_products_list = "\n".join([f"- {product['name']}: {product['description']}" for product in products_list])

        combined_prompt_input = {
            "products_list": formatted_products_list,
            "question": question,
            "chat_history": chat_history_instance.messages,
            "context": "Información relevante del contexto, si la tienes."
        }

        try:
            # Step 1: Crear un retriever basado en el historial de chat para contexto adicional
            history_aware_retriever = create_history_aware_retriever(
                llm=llm,
                retriever=pinecone_vector_store.as_retriever(), 
                prompt=contextualize_q_prompt
            )

            # Step 2: Crear una cadena de documentos para combinar respuestas
            document_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=contextual_answer_prompt
            )

            # Step 3: Crear una cadena de recuperación y generación de respuestas
            rag_chain = create_retrieval_chain(
                history_aware_retriever,
                document_chain
            )

            # Step 4: Ejecutar la cadena conversacional con el historial de mensajes
            conversational_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # Ejecutar el flujo con la entrada más reciente
            response = conversational_chain.invoke(
                {
                    "chat_history": chat_history_instance.messages,
                    "products_list": formatted_products_list,
                    "question": question,
                    "context": "Información relevante del contexto, si la tienes.",
                    "input": question
                },
                config={"configurable": {"session_id": session_id}},
            )

            response_text = response.get("answer", "")
            print(response_text)
            if response_text:
                add_message_to_history(chat_history_instance, "ai", response_text)

        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            raise

        return response