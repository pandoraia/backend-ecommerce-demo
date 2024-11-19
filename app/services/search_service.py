# app/services/search_service.py

import numpy as np
import asyncio
from langchain.chat_models import ChatOpenAI
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

# Crear el VectorStore para Pinecone con LangChain
# (Asumiendo que ya está configurado en 'vectorization_service.py')

# Inicializar el modelo OpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0.1,
                 openai_api_key=settings.openai_api_key)

# Almacén para el historial de sesiones
session_store = {}


def add_message_to_history(chat_history_instance, message_type, message_content):
    """Agregar un mensaje al historial de chat, asegurando que no esté duplicado."""
    existing_messages = [
        msg.content for msg in chat_history_instance.messages if isinstance(msg, BaseMessage)]
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


def cosine_similarity(vec_a, vec_b):
    """Calcular la similitud coseno entre dos vectores."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class SearchService:

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
            # Saltar si ya hemos procesado este producto
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

            # Eliminar 'embedding' para evitar problemas de serialización
            # product_embedding = embedder.embed_query(product_description)

            recommended_products.append({
                "slug": product_slug,
                "lang": product_lang,
                "name": product_name,
                "description": product_description,
                "image_url": product_details.images,
                # "embedding": product_embedding
            })
            # Salir del bucle si tenemos 5 productos únicos
            if len(recommended_products) >= 5:
                break

        if recommended_products:
            # Calcular la similitud entre la pregunta y las descripciones de los productos
            query_embedding = embedder.embed_query(user_query)
            for product in recommended_products:
                product_embedding = embedder.embed_query(
                    product['description'])
                product["similarity"] = cosine_similarity(
                    query_embedding, product_embedding)

            # Ordenar productos por similitud y seleccionar principal y secundarios
            recommended_products.sort(
                key=lambda x: x["similarity"], reverse=True)
            principal_product = recommended_products[0]
            # Tomar los siguientes 4 como productos secundarios
            secondary_products = recommended_products[1:5]

            return principal_product, secondary_products

        return None, []  # Retornar None si no hay productos recomendados

    @staticmethod
    async def generate_answer(question: str, session_id: str = None) -> Dict[str, Any]:
        """Generar una respuesta final a la pregunta del usuario utilizando un agente con herramientas."""

        # Obtener o crear session_id
        session_id = get_or_create_session_id(session_id)
        chat_history_instance = get_session_history(session_id)

        # Agregar la pregunta al historial
        add_message_to_history(chat_history_instance, "human", question)

        # Ejecutar el agente de manera asíncrona
        result = await agent_executor.ainvoke({"input": question})

        # Verificar si el resultado es un diccionario con las claves esperadas
        if isinstance(result, dict):
            response = result.get('output', '')
            intermediate_steps = result.get('intermediate_steps', [])
        else:
            response = result
            intermediate_steps = []

        # Asegurarse de que 'response' sea una cadena de texto
        if isinstance(response, BaseMessage):
            response_text = response.content
        else:
            response_text = response

        # Inicializar diccionario de productos
        products = {
            "principal": None,
            "secondary": []
        }

        # Extraer productos de las salidas de las herramientas
        for action, observation in intermediate_steps:
            if action.tool in ["get_principal_product", "get_secondary_products", "get_all_products"]:
                # Parsear la observación como JSON
                try:
                    product_data = json.loads(observation)
                    if action.tool == "get_principal_product":
                        products["principal"] = product_data.get("principal")
                    elif action.tool == "get_secondary_products":
                        products["secondary"] = product_data.get(
                            "secondary", [])
                    elif action.tool == "get_all_products":
                        products["principal"] = product_data.get("principal")
                        products["secondary"] = product_data.get(
                            "secondary", [])
                except json.JSONDecodeError:
                    pass  # Manejar error si la observación no es JSON

        if response_text:
            add_message_to_history(chat_history_instance, "ai", response_text)

        return {
            "response": response_text,
            "products": products
        }

# Definir las herramientas


async def get_principal_product(question: str) -> str:
    principal_product, _ = await SearchService.search_products_by_query(question)
    if principal_product:
        # Preparar la salida como JSON
        output = json.dumps({
            "principal": principal_product
        })
        return output
    else:
        return json.dumps({})


async def get_secondary_products(question: str) -> str:
    _, secondary_products = await SearchService.search_products_by_query(question)
    if secondary_products:
        # Preparar la salida como JSON
        output = json.dumps({
            "secondary": secondary_products
        })
        return output
    else:
        return json.dumps({})


async def get_all_products(question: str) -> str:
    principal_product, secondary_products = await SearchService.search_products_by_query(question)
    # Preparar la salida como JSON
    output = json.dumps({
        "principal": principal_product,
        "secondary": secondary_products
    })
    return output

# Definir las herramientas como Herramientas de LangChain
tools = [
    Tool(
        name="get_principal_product",
        func=get_principal_product,
        coroutine=get_principal_product,
        description="Usa esta herramienta para obtener la recomendación del producto principal basándote en la pregunta del usuario."
    ),
    Tool(
        name="get_secondary_products",
        func=get_secondary_products,
        coroutine=get_secondary_products,
        description="Usa esta herramienta para obtener recomendaciones de productos secundarios basándote en la pregunta del usuario."
    ),
    Tool(
        name="get_all_products",
        func=get_all_products,
        coroutine=get_all_products,
        description="Usa esta herramienta para obtener tanto las recomendaciones de productos principales como secundarios basándote en la pregunta del usuario."
    )
]

# Definir el prompt del agente
system_prompt = """
Eres un entrenador deportivo profesional y experto en vender productos deportivos.

Tu papel es proporcionar consejos personalizados basados en las necesidades del usuario, ofreciendo productos de Pandorafit que puedan ayudarle a alcanzar sus objetivos de fitness.

Si el usuario comparte detalles personales, como su nivel de condición física u objetivos (por ejemplo, perder peso o ganar músculo), ajusta tus recomendaciones en consecuencia, sugiriendo productos y consejos relevantes y coherentes que se relacionen con lo que el usuario te está preguntando.

Por ejemplo, ofrece equipos de cardio para perder peso, equipos de entrenamiento de fuerza para ganar músculo o ayudas de flexibilidad para la recuperación, dependiendo de la conversación.

Si el usuario hace preguntas que no están relacionadas con los productos de Pandorafit, infórmale amablemente que solo puedes proporcionar orientación sobre los productos deportivos en tu inventario.

Mantén siempre tu respuesta profesional, ética, empática y enfocada en soluciones.

Haz preguntas de seguimiento si es necesario para aclarar sus necesidades y hacer tus recomendaciones más personalizadas.

Responde de manera concisa, utilizando un máximo de tres oraciones, y solo recomienda productos que tengas disponibles.

Es importante responder siempre en el idioma en el que la persona te escribe.

Analiza cuidadosamente la pregunta:

- Si la pregunta es altamente relevante para un producto o objetivo específico, considera usar las herramientas apropiadas para proporcionar recomendaciones de productos.

- Si la pregunta no indica directamente una necesidad de recomendaciones de productos (por ejemplo, saludos, consultas generales), evita sugerir productos y responde solo con información textual relevante.

Utiliza las herramientas cuando sea apropiado para obtener recomendaciones de productos.
-No incluir urls en la respuesta escrita del texto.
"""

human_prompt = "{input}"

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template(human_prompt)
])

# Crear la memoria de conversación
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# Crear el agente utilizando la nueva forma recomendada
agent = ConversationalChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    system_prompt=system_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    return_intermediate_steps=True,
    output_key="output",  # Especificar la clave de salida
)
