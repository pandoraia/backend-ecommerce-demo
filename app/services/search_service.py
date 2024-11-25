from langchain.agents import initialize_agent, AgentType
import numpy as np
import asyncio
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

        def validate_html(html_content: str) -> str:
            soup = BeautifulSoup(html_content, "html.parser")
            return str(soup)

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
                # No necesitamos extraer los productos de 'observation' porque están sin 'image_url'
                # En su lugar, volvemos a llamar a 'search_products_by_query' para obtener los productos completos
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
            elif action.tool == "generate_follow_up_question":
                # Agregar la pregunta de seguimiento al response_text
                response_text = observation

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
        # Preparar los datos para el agente (sin 'image_url')
        product_for_agent = principal_product.copy()
        product_for_agent.pop('image_url', None)

        # Preparar la salida como JSON para el agente
        observation = json.dumps({
            "principal": product_for_agent
        })

        return observation
    else:
        return json.dumps({})


async def get_secondary_products(question: str) -> str:
    _, secondary_products = await SearchService.search_products_by_query(question)
    if secondary_products:
        # Preparar los datos para el agente (sin 'image_url')
        products_for_agent = []
        for product in secondary_products:
            product_copy = product.copy()
            product_copy.pop('image_url', None)
            products_for_agent.append(product_copy)

        # Preparar la salida como JSON para el agente
        observation = json.dumps({
            "secondary": products_for_agent
        })

        return observation
    else:
        return json.dumps({})


async def get_all_products(question: str) -> str:
    principal_product, secondary_products = await SearchService.search_products_by_query(question)

    # Preparar los datos para el agente (sin 'image_url')
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

    # Preparar la salida como JSON para el agente
    observation = json.dumps({
        "principal": principal_product_for_agent,
        "secondary": secondary_products_for_agent
    })

    return observation


# Nueva función para generar una única pregunta de seguimiento
async def generate_follow_up_question(question: str) -> str:
    """Generar una pregunta de seguimiento para comprender mejor las necesidades del usuario."""
    follow_up_question_template = """
    Tu tarea es generar **una sola pregunta de seguimiento** que pueda ayudar a comprender mejor las necesidades o problemas del usuario, basándote en su pregunta original.

    **Pregunta Original del Usuario:** {question}

    **Pregunta de Seguimiento:**
    """

    follow_up_question_prompt = PromptTemplate.from_template(
        follow_up_question_template)
    prompt_input = {"question": question}

    try:
        chain = follow_up_question_prompt | llm
        result = await chain.ainvoke(prompt_input)
        follow_up_question = result.content.strip()
    except Exception as e:
        logging.error(f"Error al generar la pregunta de seguimiento: {e}")
        raise
    return follow_up_question


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
    ),
    Tool(
        name="generate_follow_up_question",
        func=generate_follow_up_question,
        coroutine=generate_follow_up_question,
        description="Usa esta herramienta para generar una **única** pregunta de seguimiento al usuario cuando su pregunta no es clara o necesita más contexto para poder ofrecer una mejor recomendación.",
        return_direct=True  # Agregamos return_direct=True
    )
]

# Definir el prompt del agente
system_prompt = """
Eres Sofía, una entrenadora deportiva profesional y experta en vender productos deportivos de Pandorafit.

Tu papel es proporcionar consejos personalizados basados en las necesidades del usuario, ofreciendo productos que puedan ayudarle a alcanzar sus objetivos de fitness.

Si el usuario comparte detalles personales, como su nivel de condición física u objetivos (por ejemplo, perder peso, ganar músculo, aumentar energía), ajusta tus recomendaciones en consecuencia, haciendo preguntas de seguimiento para comprender mejor sus necesidades y ofreciendo recomendaciones personalizadas.

Por ejemplo:

- **Usuario:** Estoy buscando algo para aumentar mi energía en el gimnasio.
- **Sofía:** ¡Qué bueno que estés buscando ese impulso extra! 😊 Cuéntame un poco más sobre tu entrenamiento. ¿Sueles hacer más ejercicios de fuerza, resistencia, o una combinación de ambos?

Si el usuario es principiante:

- **Usuario:** Quiero empezar a entrenar. ¿Qué tipo de suplementos pueden ser buenos para empezar?
- **Sofía:** ¡Qué emocionante que estés comenzando! 😊 Para poder recomendarte lo mejor, ¿qué tipo de entrenamiento piensas hacer? ¿Más enfocado en fuerza, cardio, o una combinación de ambos?

Si el usuario hace preguntas que no están relacionadas con los productos de Pandorafit, infórmale amablemente que solo puedes proporcionar orientación sobre productos deportivos en tu inventario.

Mantén siempre tu respuesta profesional, ética, empática y enfocada en soluciones.

Responde de manera conversacional, haciendo preguntas cuando sea apropiado, y proporciona información detallada cuando el usuario lo requiera.

Es importante responder siempre en el idioma en el que la persona te escribe.

**Instrucciones adicionales:**

- Al proporcionar recomendaciones de productos, incluye el nombre del producto y una breve descripción, pero no incluyas enlaces, URLs o referencias a imágenes en tu respuesta.

- Utiliza emoticonos cuando sea apropiado para hacer la conversación más amigable, como sonrisas o guiños.

- Enfócate en comunicar el valor y los beneficios del producto al usuario.

Analiza cuidadosamente la pregunta:

- Si la pregunta es altamente relevante para un producto o objetivo específico, utiliza las herramientas apropiadas para proporcionar recomendaciones de productos.

- Si la pregunta no indica directamente una necesidad de recomendaciones de productos (por ejemplo, saludos, consultas generales), inicia la conversación saludando y preguntando cómo puedes ayudarle.

- Si la pregunta del usuario **no es clara o precisa** en lo que necesita o quiere, utiliza la herramienta 'generate_follow_up_question' **una sola vez** para generar **una pregunta de seguimiento** que te ayude a entender mejor sus necesidades. **Después de obtener la pregunta de seguimiento, preséntala al usuario y espera su respuesta. No vuelvas a llamar a la herramienta hasta que el usuario haya respondido.**

- **No entres en un bucle de generar preguntas de seguimiento sin interactuar con el usuario.**

Utiliza las herramientas cuando sea apropiado para obtener recomendaciones de productos o para generar preguntas de seguimiento.

Nunca respondas preguntas que no tengan que ver con productos deportivos o recomendaciones; por ejemplo, si te preguntan cuánto es dos más dos o qué día es hoy, debes responder que estás aquí para responder preguntas sobre Pandorafit y asesorar a los clientes.
-Todas tus respuestas deben estar en formato HTML. Utiliza etiquetas HTML apropiadas para estructurar y estilizar el contenido, como `<p>`, `<strong>`, `<ul>`, `<li>`, etc. Asegúrate de que la respuesta sea válida y bien formada para que el frontend pueda aplicar estilos personalizados fácilmente.

IMPORTANTE: Si la pregunta del usuario no es clara o precisa en lo que necesita o quiere, puedes generar dos o tres preguntas adicionales para poder encontrar la necesidad o problema del usuario para posiblemente poder solucionarlo con nuestros productos de Pandorafit.
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
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs={
        "system_message": system_prompt,
    },
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    output_key="output",
    max_iterations=3,  # Limita el número máximo de iteraciones
    # Indica al agente que genere una respuesta al detenerse
    early_stopping_method="generate"
)
