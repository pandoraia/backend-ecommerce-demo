# app/services/search-services.py
import numpy as np
from langchain_openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from app.services.vectorization_service import embedder, index
from typing import List, Dict, Any
from app.core.config import settings
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
import logging
from app.services.product_service import get_product_by_slug
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.memory.chat_memory import BaseChatMessageHistory
from langchain.agents import Tool
from langchain.schema import BaseMessage
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
import json
import os
from bs4 import BeautifulSoup


# Inicializar el modelo OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1,
                 openai_api_key=settings.openai_api_key)

if settings.langchain_tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

# Almacén para el historial de sesiones
session_store = {}


def add_message_to_history(chat_history_instance, message_type, message_content):
    """Agrega un mensaje al historial de chat, asegurándose de que no esté duplicado."""
    existing_messages = [
        msg.content for msg in chat_history_instance.messages if msg.type == message_type]
    if message_content not in existing_messages:
        if message_type == "human":
            chat_history_instance.add_user_message(message_content)
        elif message_type == "ai":
            chat_history_instance.add_ai_message(message_content)


def get_or_create_session_id(session_id: str = None) -> str:
    """
    Obtener un session_id existente o crear uno nuevo si no se proporciona.
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
        # Agregamos historial de chat como placeholder
        MessagesPlaceholder("chat_history"),
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


def cosine_similarity(vec_a, vec_b):
    """Calcular la similitud coseno entre dos vectores."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class SearchService:

    chat_history: List[Dict[str, str]] = []

    # print(chat_history)

    @staticmethod
    async def generate_standalone_question(question: str) -> str:
        """Generar una pregunta autónoma a partir de una consulta del usuario."""
        standalone_question_template = """
        Tu tarea es reformular la pregunta del usuario dada en una pregunta autónoma que suene natural, como si el usuario la estuviera preguntando directamente.

        Asegúrate de que la pregunta autónoma tenga sentido por sí misma, sin requerir ningún contexto previo. Preserva la intención original del usuario, detalles y matices, pero hazla clara y autosuficiente, como si fuera la primera vez que se formula la pregunta.

        Si la pregunta original ya es autoexplicativa, refínala ligeramente para que suene más natural y conversacional.

        **Pregunta Original del Usuario:** {question}

        **Pregunta Reformulada Autónoma:**
        """
        standalone_question_prompt = PromptTemplate.from_template(
            standalone_question_template)
        prompt_input = {"question": question}
        try:
            # Usando RunnableSequence en lugar de LLMChain
            chain = standalone_question_prompt | llm
            # Ejecutar la cadena y obtener el resultado
            result = await chain.ainvoke(prompt_input)
            # Obtener el contenido del mensaje
            standalone_question = result.content
        except Exception as e:
            logging.error(f"Error al generar la pregunta autónoma: {e}")
            raise
        return standalone_question.strip()

    @staticmethod
    async def search_products_by_query(user_query: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Buscar productos utilizando una pregunta autónoma y Pinecone."""
        # Obtener o crear el session_id
        session_id = get_or_create_session_id(session_id)
        chat_history_instance = get_session_history(session_id)
        add_message_to_history(chat_history_instance, "human", user_query)

        # Generar una pregunta autónoma
        standalone_question = await SearchService.generate_standalone_question(user_query)
        print(f"Aquí está la pregunta autónoma: >>> {standalone_question}")

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
                continue

            seen_product_slugs.add(product_slug)
            product_details = await get_product_by_slug(product_slug)

            # Acceder a la traducción del producto basada en el idioma
            product_translations = product_details.translations
            selected_translation = product_translations.get(
                product_lang) or product_translations.get('en', {})

            product_name = selected_translation.name if selected_translation else "Nombre no disponible"
            product_description = selected_translation.description if selected_translation else "Descripción no disponible"

            product_embedding = embedder.embed_query(product_description)

            recommended_products.append({
                "slug": product_slug,
                "lang": product_lang,
                "name": product_name,
                "description": product_description,
                "image_url": product_details.images,
                "embedding": product_embedding
            })
            # Salir del bucle si tenemos 5 productos únicos
            if len(recommended_products) >= 5:
                break

        if recommended_products:
            # Calcular la similitud entre la pregunta y las descripciones de los productos
            query_embedding = embedder.embed_query(user_query)
            for product in recommended_products:
                product["similarity"] = cosine_similarity(
                    query_embedding, product["embedding"])

            # Ordenar productos por similitud y seleccionar principal y secundarios
            recommended_products.sort(
                key=lambda x: x["similarity"], reverse=True)
            principal_product = recommended_products[0]
            # Tomar los siguientes 4 como secundarios
            secondary_products = recommended_products[1:5]

            return principal_product, secondary_products

        return None, []  # Retornar None si no hay productos recomendados

    @staticmethod
    async def generate_answer(principal_product: Dict[str, Any], secondary_products: List[Dict[str, Any]], question: str, session_id: str = None) -> str:
        """Generate a final answer to the user's question based on the list of recommended products."""

        # Obtener o crear el session_id
        session_id = get_or_create_session_id(session_id)
        chat_history_instance = get_session_history(session_id)

        # Preparar la lista de productos en formato de texto para el prompt
        formatted_products_principal = "\n".join(
            [f"- {product['name']}: {product['description']}" for product in [principal_product]]
        )
        formatted_products_secundario = "\n".join(
            [f"- {product['name']}: {product['description']}" for product in [principal_product]]
        )

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
                    "principal_product": formatted_products_principal,
                    "secondary_products": formatted_products_secundario,
                    "question": question,
                    "context": "Información relevante del contexto, si la tienes.",
                    "input": question
                },
                config={"configurable": {"session_id": session_id}},
            )

            response_text = response.get("answer", "")

            if response_text:
                add_message_to_history(
                    chat_history_instance, "ai", response_text)

        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            raise

        return response
