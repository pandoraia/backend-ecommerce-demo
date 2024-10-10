from langchain_openai import OpenAI
from app.services.vectorization_service import embedder, index
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from app.core.config import settings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging
from app.services.product_service import get_product_by_slug

# Initialize OpenAI model
llm = ChatOpenAI(openai_api_key=settings.openai_api_key)

# Create standalone question prompt template
standalone_question_template = """Given some question, convert the question to a standalone question. 
question: {question}
standalone question:"""
standalone_question_prompt = PromptTemplate.from_template(standalone_question_template)

# Updated answer prompt template
answer_template = """Yall way convertsacional you el texto debe ser mu corto are a helpful and enthusiastic support bot. Based on the products provided, recommend the best product that most closely fits the user's question, explaining why it is the best option. Additionally, suggest two complementary products that could help improve the user's experience.
Ensure that the main product recommendation is directly related to the user's query (for example, if the user is looking to improve cardio or muscle building, select products that address that need).
Clearly explain why that main product is recommended based on the user's question.
Return the answer in the same language in which the user made the query.
Main product recommendation: {recommended_product}
Complementary products: {complementary_products}
User question: {question}
Answer:"""
answer_prompt = PromptTemplate.from_template(answer_template)

class SearchService:
    @staticmethod
    async def generate_standalone_question(question: str ) -> str:
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
    async def search_products_by_query(user_query: str) -> List[Dict[str, Any]]:
        """Search for products using a standalone question and Pinecone."""
        # Generar una pregunta autónoma
        standalone_question = await SearchService.generate_standalone_question(user_query)
        print(f"Here is the standalone question: >>> {standalone_question}")
        
        # Generar embedding para la pregunta autónoma
        embedding = embedder.embed_query(standalone_question)
        results = index.query(vector=embedding, top_k=2)  # Ajusta top_k según sea necesario
        
        # Extraer los productos recomendados
        recommended_products = []
        for match in results['matches']:
            recommended_products.append({"id": match['id']})
        
        if not recommended_products:
            return []
        
        # Procesar la recomendación principal
        main_recommendation = recommended_products[0]['id']
        complementary_recommendations = [product['id'] for product in recommended_products[1:]]
        
        # Procesar la recomendación principal
        main_recommendation_slug = main_recommendation.split("_")[0]
        main_recommendation_lang = main_recommendation.split("_")[1]
        
        main_recommendation_product_details = await get_product_by_slug(main_recommendation_slug)
        
        # Acceder a la traducción basada en el idioma extraído
        product_translations = main_recommendation_product_details.translations
        selected_translation = product_translations.get(main_recommendation_lang) or product_translations.get('en', {})
        
        product_name = selected_translation.name if selected_translation else "Nombre no disponible"
        product_description = selected_translation.description if selected_translation else "Descripción no disponible"
        
        print(f"Producto recomendado: {product_name}")
        print(f"Descripción: {product_description}")
        
        main_product_details = {
            "name": product_name,
            "description": product_description,
            "image_url": main_recommendation_product_details.images
        }
        
        # Ahora procesamos las recomendaciones complementarias
        complementary_product_details = []
        
        for complementary_id in complementary_recommendations:
            complementary_slug = complementary_id.split("_")[0]
            complementary_lang = complementary_id.split("_")[1]
            
            # Cambiamos el nombre de la variable para evitar sobrescribir la lista
            complementary_product_obj = await get_product_by_slug(complementary_slug)
            
            # Acceder a la traducción del producto complementario basada en el idioma extraído
            complementary_translations = complementary_product_obj.translations
            complementary_selected_translation = complementary_translations.get(complementary_lang) or complementary_translations.get('en', {})
            
            complementary_name = complementary_selected_translation.name if complementary_selected_translation else "Nombre no disponible"
            complementary_description = complementary_selected_translation.description if complementary_selected_translation else "Descripción no disponible"
            
            # Añadir los detalles del producto complementario a la lista
            complementary_product_details.append({
                "name": complementary_name,
                "description": complementary_description,
                # "category": complementary_product_obj.category,
                # "subCategory": complementary_product_obj.subCategory
                "image_url": complementary_product_obj.images
                
            })
            
            print(f"Producto complementario: {complementary_name}")
            print(f"Descripción: {complementary_description}")
        
        # Ahora puedes devolver los productos recomendados y complementarios
        return {
            "main_product": main_product_details,
            "complementary_products": complementary_product_details
        }

    @staticmethod
    async def generate_answer(recommended_product: str, complementary_products: str, question: str) -> str:
        """Generate a final answer to the user's question based on the recommended and complementary products."""
        prompt_input = {
            "recommended_product": recommended_product,
            "complementary_products": complementary_products,
            "question": question
        }
        try:
            chain = LLMChain(llm=llm, prompt=answer_prompt)
            answer = await chain.apredict(**prompt_input)
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            raise
        return answer
