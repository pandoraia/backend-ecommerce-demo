from langchain.agents import initialize_agent, AgentType
import numpy as np
from langchain_openai import ChatOpenAI
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
from langchain.memory import ConversationBufferMemory
import json
import os
from app.services.mongo_chat_history import MongoDBChatMessageHistory
from langchain.schema import BaseMessage, AIMessage, HumanMessage
import re

def convert_markdown_to_html(text: str) -> str:
    """
    Convierte patrones sencillos de Markdown en HTML:
      - **texto** => <strong>texto</strong>
      - _texto_ => <em>texto</em>
      - Elimina triple backticks ``` (si los hubiera).
      - Puedes añadir más reglas según tus necesidades.
    """
    # Reemplazar **texto** por <strong>texto</strong>
    bold_pattern = r"\*\*(.*?)\*\*"
    text = re.sub(bold_pattern, r"<strong>\1</strong>", text)

    # Reemplazar _texto_ por <em>texto</em>
    italic_pattern = r"_(.*?)_"
    text = re.sub(italic_pattern, r"<em>\1</em>", text)

    # Eliminar bloques de triple backticks
    triple_backticks_pattern = r"```(.*?)```"
    text = re.sub(triple_backticks_pattern, r"\1", text, flags=re.DOTALL)

    return text

def ensure_html_format(response: str) -> str:
    """
    Verifica si 'response' contiene etiquetas HTML básicas.
    Si no las encuentra, la envuelve todo el contenido en <p></p>.
    Además, hace una limpieza de Markdown usando 'convert_markdown_to_html'.
    """
    # 1) Convertir markdown a HTML básico
    response = convert_markdown_to_html(response)

    # 2) Ver si existen etiquetas HTML
    html_tags = ["<p>", "<ul>", "<ol>", "<li>",
                 "<div>", "<span>", "<strong>", "<em>"]
    if not any(tag in response.lower() for tag in html_tags):
        # Si no encontró ninguna etiqueta, entonces envuelve en <p>...</p>
        return f"<p>{response}</p>"

    return response


# Inicializar el modelo OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=settings.openai_api_key
)

if settings.langchain_tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

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
        Your task is to rephrase the user's question into a standalone question that sounds natural, as if the user is asking it directly.
        Ensure that the standalone question makes sense on its own, without requiring any prior context. Preserve the user's original intent, details, and nuances, but make it clear and self-sufficient, as if it is being asked for the first time.
        If the original question is already self-explanatory, refine it slightly to make it sound more natural and conversational.
        The rephrased question must always remain in the same language as the original question provided by the user.
        **Original User Question:** {question}
        **Rephrased Standalone Question:**
        """
        standalone_question_prompt = PromptTemplate.from_template(
            standalone_question_template)
        prompt_input = {"question": question}
        try:
            chain = standalone_question_prompt | llm
            result = await chain.ainvoke(prompt_input)
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

            recommended_products.append({
                "slug": product_slug,
                "lang": product_lang,
                "name": product_name,
                "description": product_description,
                "image_url": product_details.images,
                "price": product_details.price,
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
            secondary_products = recommended_products[1:5]

            return principal_product, secondary_products

        return None, []  # Retornar None si no hay productos recomendados

    @staticmethod
    async def generate_answer(question: str, session_id: str ) -> Dict[str, Any]:
        """Genera una respuesta final considerando las últimas 20 interacciones."""

        agent_name = "Sofia"
        chat_history = MongoDBChatMessageHistory(agent_name, session_id)
        
        # 🔹 Precargar historial en memoria antes de usarlo
        await chat_history._load_messages()

        # 🔹 Inicializar memoria con historial de MongoDB
        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        # 🔹 Verificar que la memoria contenga mensajes previos
        print(f"Mensajes antes de agregar nuevos: {[msg.content for msg in memory.chat_memory.messages]}")
        
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            agent_kwargs={ "system_message": system_prompt,},
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            output_key="output",
            max_iterations=3,
            early_stopping_method="generate"
        )

        # Agregar la consulta del usuario al historial
        await chat_history.add_message(HumanMessage(content=question))
        result = await agent_executor.ainvoke({"input": question})

        # Extraer la respuesta y las acciones intermedias
        last_20_messages = await chat_history.aget_messages()
        formatted_messages = "\n".join(
        [f"{'User' if isinstance(msg, HumanMessage) else 'Sofia'}: {msg.content}" for msg in last_20_messages]
        )
        print(f"Formateado: {formatted_messages}")
        if isinstance(result, dict):
            response = result.get('output', '')
            intermediate_steps = result.get('intermediate_steps', [])
        else:
            response = result
            intermediate_steps = []
        # Procesar la respuesta final
        if isinstance(response, BaseMessage):
            response_text = response.content
        else:
            response_text = response

        # === CHANGE ===
        # Asegurar que la respuesta final sea en HTML
        response_text = ensure_html_format(response_text)

        # Guardar la respuesta en el historial
        if response_text:
            await chat_history.add_message(AIMessage(content=response_text))

        # Productos recomendados (si aplica)
        products = {"principal": None, "secondary": []}
        for action, observation in intermediate_steps:
            if action.tool in ["get_principal_product", "get_secondary_products", "get_all_products"]:
                if action.tool == "get_principal_product":
                    principal_product, _ = await SearchService.search_products_by_query(action.tool_input)
                    products["principal"] = principal_product
                elif action.tool == "get_secondary_products":
                    _, secondary_products = await SearchService.search_products_by_query(action.tool_input)
                    products["secondary"] = secondary_products
                elif action.tool == "get_all_products":
                    principal_product, secondary_products = await SearchService.search_products_by_query(action.tool_input)
                    products["principal"] = principal_product
                    products["secondary"] = secondary_products

        return {"response": response_text, "products": products}


async def get_principal_product(question: str) -> str:
    principal_product, _ = await SearchService.search_products_by_query(question)
    if principal_product:
        # Preparar los datos para el agente (sin 'image_url')
        product_for_agent = principal_product.copy()
        product_for_agent.pop('image_url', None)
        observation = json.dumps({
            "principal": product_for_agent
        })
        return observation
    else:
        return json.dumps({})


async def get_secondary_products(question: str) -> str:
    _, secondary_products = await SearchService.search_products_by_query(question)
    if secondary_products:
        products_for_agent = []
        for product in secondary_products:
            product_copy = product.copy()
            product_copy.pop('image_url', None)
            products_for_agent.append(product_copy)

        observation = json.dumps({
            "secondary": products_for_agent
        })

        return observation
    else:
        return json.dumps({})


async def get_all_products(question: str) -> str:
    principal_product, secondary_products = await SearchService.search_products_by_query(question)

    if principal_product:
        principal_product_for_agent = principal_product.copy()
        principal_product_for_agent.pop('image_url', None)
    else:
        principal_product_for_agent = None

    secondary_products_for_agent = []
    for product in secondary_products:
        product_copy = product.copy()
        product_copy.pop('image_url', None)
        secondary_products_for_agent.append(product_copy)

    observation = json.dumps({
        "principal": principal_product_for_agent,
        "secondary": secondary_products_for_agent
    })

    return observation


tools = [
    Tool(
        name="get_principal_product",
        func=get_principal_product,
        coroutine=get_principal_product,
        description="Use this tool to get the main product recommendation based on the user's question.",
    ),
    Tool(
        name="get_secondary_products",
        func=get_secondary_products,
        coroutine=get_secondary_products,
        description="Use this tool to get secondary product recommendations based on the user's question.",
    ),
    Tool(
        name="get_all_products",
        func=get_all_products,
        coroutine=get_all_products,
        description="Use this tool to get both main and secondary product recommendations based on the user's question.",
    ),
]

# === CHANGE ===
# Hemos reforzado la parte final del system_prompt con "IMPORTANT"
system_prompt = """
You are Sofía, a professional sports coach and an expert in selling sports products for Pandorafit.

Your role is to generate sales by offering products that can help users achieve their fitness goals.

If the user shares personal details, such as their fitness level or goals (e.g., losing weight, building muscle, increasing energy), tailor your recommendations accordingly by providing personalized suggestions.

If the user asks questions unrelated to Pandorafit's products, politely inform them that you can only provide guidance on the sports products in your inventory.

Always maintain your responses as professional, ethical, empathetic, and solution-focused.

Respond conversationally, asking questions only when appropriate, and provide detailed information when the user requests it.

It is essential to always reply in the language of the user's question: if it's in English, reply in English; if it's in Spanish, reply in Spanish, etc.

Additional Instructions:

When recommending products, include the product name and a brief description, but do not include links, URLs, or references to images in your response.

Focus on communicating the value and benefits of the product to the user.

Carefully analyze the question:
- If the question is highly relevant to a specific product or goal, use the appropriate tools to provide product recommendations.
- If the question does not directly indicate a need for product recommendations (e.g., greetings, general inquiries), start the conversation by greeting the user and asking how you can help them.
- If the user's question is unclear or vague about their needs or wants, use the 'generate_follow_up_question' tool only once every four interactions to generate a follow-up question that helps you better understand their needs. After obtaining the follow-up question, present it to the user and wait for their response. Do not use the tool again until the user responds.
- Avoid entering a loop of generating follow-up questions without interacting with the user.
- Use tools when appropriate to obtain product recommendations or generate follow-up questions.
- Never answer questions unrelated to sports products or recommendations. If someone asks how much 2 + 2 is or what day it is, respond that you are here to answer questions about Pandorafit and assist customers.
- If you cannot find the product the user is looking for, say you do not have it and apologize.
- Recommend only products that are in your inventory and truly align with the user's request. If you do not have the specific product the user wants, say you do not have that product but can recommend a similar one. Otherwise, do not recommend anything.

- If you can't find the product the user is looking for, say you don't have it, briefly apologize, and suggest a close alternative if one exists.
- If the user asks for a cheaper option, but you can't find anything cheaper or there's nothing cheaper, say so clearly and then (only if relevant) offer other products that might be a bit more expensive, honestly explaining that they aren't cheaper but might meet the user's needs.

IMPORTANT:
1) ALL your responses MUST be in valid HTML.
2) Never return plain text, JSON, Markdown triple backticks, or code fences unless it is wrapped in HTML tags.
3) If you provide a list, use <ul> <li> or <ol> <li>.
4) Always ensure the final message is properly closed in HTML.
5) Only use the 'generate_follow_up_question' tool if the user’s request is unclear or ambiguous.
6) You must not use 'generate_follow_up_question' more than once every three user interactions.
7) If the user's question is direct and clear, answer concisely without asking an additional question.
8) Do not ask irrelevant or overly generic follow-up questions unless they are necessary to clarify user needs.
9)Use ONLY HTML tags to format your responses. 
10)Do NOT use Markdown syntax (like ** or ```).
11)If you format text as bold, use <strong>. For italics, use <em>.

"""

human_prompt = "{input}"

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template(human_prompt)
])




