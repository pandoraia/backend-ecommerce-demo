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
standalone_question_prompt = PromptTemplate.from_template(
    standalone_question_template)

# Updated answer prompt template
answer_template = """You are the top sales assistant specializing in providing clear and effective responses to guide customers in purchasing sports products from Pandorifit. Use the following products and their descriptions to provide a precise recommendation that answers the customer's question and highlights the product `{recommended_product}` as the best choice.

If the customer's question is not related to sports products, training, or health advice, kindly inform them that you can only assist with recommendations for sports products from Pandorifit.

Please recommend the best product `{recommended_product}` that meets the customer's needs, explaining why it is the most suitable option.

Examples:

**Example 1: Weighted Vest**
Customer: "I'm looking for something to intensify my bodyweight workouts. What do you recommend?"  
Chatbot: "Absolutely! To increase the intensity of your bodyweight workouts, I recommend our weighted vest. It's perfect for exercises like squats, push-ups, and pull-ups, adding extra resistance to enhance your strength and endurance. Would you like more details on the available weights or how to incorporate it into your routine?"

**Example 2: Speed Rope**
Customer: "I want to improve my speed and cardiovascular endurance, but I'm not sure what equipment is best for that."  
Chatbot: "Great choice! To improve your speed and cardiovascular endurance, our speed rope is ideal. It's perfect for high-intensity workouts like crossfit or for those looking for quick calorie-burning sessions. Would you like to learn some exercises or techniques you can do with it?"

**Example 3: Kettlebell**
Customer: "I'm looking for something that helps me work on both strength and cardio at home, any suggestions?"  
Chatbot: "For a versatile workout that combines strength and cardio, our kettlebells are a fantastic option. They allow you to do exercises like swings, squats, and snatches, targeting multiple muscle groups while improving your cardiovascular endurance. Would you like advice on the most suitable weight for you or how to integrate it into your routine?"

**Example 4: Resistance Bands**
Customer: "I need equipment that is easy to store and use at home but also effective for full-body workouts."  
Chatbot: "For full-body workouts that you can easily do at home, I recommend our resistance bands. They are compact, versatile, and can be used for a wide range of exercises, from strength training to stretching. Would you like suggestions on specific exercises for different muscle groups?"

**Example 5: Foam Roller**
Customer: "I'm looking for something that can help with muscle recovery after intense workouts. What would you suggest?"  
Chatbot: "For effective muscle recovery, I highly recommend our foam roller. It's perfect for easing muscle tension, improving flexibility, and speeding up recovery after intense workouts. Would you like to know how to use it effectively for different muscle groups?"

**Important notes:**
1. The response must be given in the same language the customer uses to ask their question.
2. If relevant and logical, suggest complementary products to help the customer achieve better results in their training or health.

**Additional example:**
Customer: "I want to improve my physical endurance, but I don't have much time to train."  
Chatbot: "To improve your endurance in short sessions, I recommend our speed rope. It's great for high-intensity workouts that quickly boost cardiovascular capacity. Additionally, you can complement your routine with a kettlebell `{complementary_products}`, which helps build strength while you continue improving your endurance. Would you like to know how to combine them into an effective routine?"

**Main product recommendation:** {recommended_product}  
**Complementary products:** {complementary_products}  
**Customer question:** {question}  
**Answer:**
 """
answer_prompt = PromptTemplate.from_template(answer_template)


class SearchService:
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
    async def search_products_by_query(user_query: str) -> List[Dict[str, Any]]:
        """Search for products using a standalone question and Pinecone."""
        # Generar una pregunta autónoma
        standalone_question = await SearchService.generate_standalone_question(user_query)
        print(f"Here is the standalone question: >>> {standalone_question}")

        # Generar embedding para la pregunta autónoma
        embedding = embedder.embed_query(standalone_question)
        # Ajusta top_k según sea necesario
        results = index.query(vector=embedding, top_k=2)

        # Extraer los productos recomendados
        recommended_products = []
        for match in results['matches']:
            recommended_products.append({"id": match['id']})

        if not recommended_products:
            return []

        # Procesar la recomendación principal
        main_recommendation = recommended_products[0]['id']
        complementary_recommendations = [product['id']
                                         for product in recommended_products[1:]]

        # Procesar la recomendación principal
        main_recommendation_slug = main_recommendation.split("_")[0]
        main_recommendation_lang = main_recommendation.split("_")[1]

        main_recommendation_product_details = await get_product_by_slug(main_recommendation_slug)

        # Acceder a la traducción basada en el idioma extraído
        product_translations = main_recommendation_product_details.translations
        selected_translation = product_translations.get(
            main_recommendation_lang) or product_translations.get('en', {})

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
            complementary_selected_translation = complementary_translations.get(
                complementary_lang) or complementary_translations.get('en', {})

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
