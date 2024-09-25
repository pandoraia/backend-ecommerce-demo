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
answer_template = """You are a helpful and enthusiastic support bot. Based on the products retrieved, recommend the best product and suggest two complementary options.
Top product recommendation: {recommended_product}
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
        # Generate a standalone question
        standalone_question = await SearchService.generate_standalone_question(user_query)
        print(f"Here is the standalone question: >>> {standalone_question}")

        # Generate embedding for the standalone question
        embedding = embedder.embed_query(standalone_question)

        # Search Pinecone index
        results = index.query(vector=embedding, top_k=3)  # Adjust top_k as necessary
        
        # print(results)

        # Extract product metadata
        recommended_products = []
        for match in results['matches']:
            recommended_products.append({
                "id": match['id'],
                # You can include additional metadata as needed
                #  "name": match['metadata'].get("name", ""),
                # "description": match['metadata'].get("description", ""),
                # Optionally include the relevance score
            })
            
        # print(recommended_products)
        main_recommendation = recommended_products[0]['id'] if recommended_products else None
        complementary_recommendations = [product['id'] for product in recommended_products[1:]]
        print(f"Main recommendation: {main_recommendation}")
        print(f"Complementary recommendations: {complementary_recommendations}")
        
        # parts_main_product_principal = main_recommendation.split("_")
        # # parts_main_product_complementari_1 = main_recommendation.split("_")
        # # parts_main_product_complementari_2 = main_recommendation.split("_")
        
        # slug_principal = parts_main_product_principal[0]
        # # slug_complementari_1 = parts_main_product[1]
        # # slug_complementari_2= parts_main_product[2]
        
        # product_principal = await get_product_by_slug(slug_principal)
        # # product_complementari_1 = await get_product_by_slug(slug_complementari_1)
        # # product_complementari_2 = await get_product_by_slug(slug_complementari_2)
        # print(product_principal)
        

        return recommended_products

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
