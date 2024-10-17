

import re
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from app.models.product import Product
from app.core.config import settings
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# Inicializar Pinecone usando la nueva clase Pinecone
pc = Pinecone(api_key=settings.pinecone_api_key)

# Nombre del índice
index_name = settings.pinecone_index

# Comprobar si el índice ya existe
existing_indexes = pc.list_indexes()

# Acceder al índice creado o ya existente
index = pc.Index(index_name)

# Crear el almacén de vectores de Langchain para Pinecone
embedder = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
vector_store = PineconeVectorStore(index=index, embedding=embedder)



async def split_text_optimally(text: str) -> list:
    """
    Divide el texto en segmentos más pequeños para capturar mejor el contexto.
    """
    sections = re.split(r'\n\n+', text.strip())
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len
    )

    chunks = []
    for section in sections:
        if len(section) > 200:
            sub_chunks = section_splitter.split_text(section)
            chunks.extend(sub_chunks)
        else:
            chunks.append(section)

    return chunks

async def generate_product_embeddings(product: Product) -> None:
    """
    Genera embeddings para un producto y sus traducciones en diferentes idiomas.
    """
    for lang, translation in product.translations.items():
        # Datos del producto por idioma
        name_text = translation.name
        description_text = translation.description
        full_text_sections = [
            translation.destovect,  # Secciones detalladas del producto
            f"Category: {translation.category}",
            f"SubCategory: {translation.subCategory}"
        ]

        # Generar embeddings para el nombre del producto y la descripción general
        name_embedding = embedder.embed_query(name_text)
        description_embedding = embedder.embed_query(description_text)

        # Crear una lista para almacenar todos los embeddings bajo el mismo vector_id
        vectors_to_upsert = []

        # Agregar embedding del nombre del producto con metadatos
        metadata_name = {
            "product_slug": product.slug,
            "language": lang,
            "type": "name"
        }
        vectors_to_upsert.append((f"{product.slug}_{lang}_name", name_embedding, metadata_name))

        # Agregar embedding de la descripción general con metadatos
        metadata_description = {
            "product_slug": product.slug,
            "language": lang,
            "type": "description"
        }
        vectors_to_upsert.append((f"{product.slug}_{lang}_description", description_embedding, metadata_description))

        # Procesar y dividir las secciones detalladas del producto para embeddings más específicos
        for i, section in enumerate(full_text_sections):
            section_segments = await split_text_optimally(section)
            segment_embeddings = embedder.embed_documents(section_segments)

            # Agregar cada segmento como un embedding individual con metadatos específicos y manteniendo el orden
            for j, segment_embedding in enumerate(segment_embeddings):
                segment_metadata = {
                    "product_slug": product.slug,
                    "language": lang,
                    "type": "section_segment",
                    "section_index": i,
                    "segment_index": j
                }
                vectors_to_upsert.append((f"{product.slug}_{lang}_segment_{i}_{j}", segment_embedding, segment_metadata))

        # Subir los embeddings al índice de Pinecone asegurando el orden coherente
        index.upsert(vectors=vectors_to_upsert)

        print(f"Product embeddings for {product.slug} in language {lang} have been successfully upserted.")



# import re
# from pinecone import Pinecone
# from langchain_openai import OpenAIEmbeddings
# from app.models.product import Product
# from app.core.config import settings
# from typing import Dict, Any
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Inicializar Pinecone usando la clase Pinecone
# pc = Pinecone(api_key=settings.pinecone_api_key)

# # Nombre del índice
# index_name = settings.pinecone_index

# # Acceder al índice creado o ya existente
# index = pc.Index(index_name)

# # Crear el almacén de vectores para embeddings usando LangChain
# embedder = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)

# async def split_text_optimally(text: str) -> list:
#     """
#     Divide el texto en segmentos más pequeños para capturar mejor el contexto.
#     """
#     sections = re.split(r'\n\n+', text.strip())
#     section_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=200,
#         chunk_overlap=20,
#         length_function=len 
#     )

#     chunks = []
#     for section in sections:
#         if len(section) > 200:
#             sub_chunks = section_splitter.split_text(section)
#             chunks.extend(sub_chunks)
#         else:
#             chunks.append(section)

#     return chunks

# async def generate_product_embeddings(product: Product) -> None:
#     """
#     Genera embeddings para un producto y sus traducciones en diferentes idiomas.
#     """
#     for lang, translation in product.translations.items():
#         # Datos del producto por idioma
#         name_text = translation.name
#         description_text = translation.description
#         full_text_sections = [
#             translation.destovect,  # Secciones detalladas del producto
#             f"Category: {translation.category}",
#             f"SubCategory: {translation.subCategory}"
#         ]

#         # Generar embeddings para el nombre del producto y la descripción general
#         name_embedding = embedder.embed_query(name_text)
#         description_embedding = embedder.embed_query(description_text)

#         # Crear una lista para almacenar todos los embeddings bajo el mismo vector_id
#         vectors_to_upsert = []

#         # Agregar embedding del nombre del producto con metadatos
#         metadata_name = {
#             "product_slug": product.slug,
#             "language": lang,
#             "type": "name"
#         }
#         vectors_to_upsert.append((f"{product.slug}_{lang}_name", name_embedding, metadata_name))

#         # Agregar embedding de la descripción general con metadatos
#         metadata_description = {
#             "product_slug": product.slug,
#             "language": lang,
#             "type": "description"
#         }
#         vectors_to_upsert.append((f"{product.slug}_{lang}_description", description_embedding, metadata_description))

#         # Procesar y dividir las secciones detalladas del producto para embeddings más específicos
#         for section in full_text_sections:
#             section_segments = await split_text_optimally(section)
#             segment_embeddings = embedder.embed_documents(section_segments)

#             # Agregar cada segmento como un embedding individual con metadatos específicos
#             for i, segment_embedding in enumerate(segment_embeddings):
#                 segment_metadata = {
#                     "product_slug": product.slug,
#                     "language": lang,
#                     "type": "section_segment",
#                     "section_index": i
#                 }
#                 vectors_to_upsert.append((f"{product.slug}_{lang}_segment_{i}", segment_embedding, segment_metadata))

#         # Subir los embeddings al índice de Pinecone
#         index.upsert(vectors=vectors_to_upsert)

#         print(f"Product embeddings for {product.slug} in language {lang} have been successfully upserted.")
# import time
# import os
# import re
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
# from app.models.product import Product
# from app.core.config import settings
# from typing import Dict, Any
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# # Inicializar Pinecone usando la nueva clase Pinecone
# pc = Pinecone(api_key=settings.pinecone_api_key)

# # Nombre del índice
# index_name = settings.pinecone_index

# # Comprobar si el índice ya existe
# existing_indexes = pc.list_indexes()

# # Acceder al índice creado o ya existente
# index = pc.Index(index_name)

# # Crear el almacén de vectores de Langchain para Pinecone
# embedder = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
# vector_store = PineconeVectorStore(index=index, embedding=embedder)

# async def split_text_optimally(text: str) -> list:
#     """
#     Divide el texto de forma óptima basándose en saltos de línea, encabezados y longitud de caracteres.
#     """
#     sections = re.split(r'\n\n+', text.strip())  # Dividir en bloques usando saltos de línea dobles

#     section_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=200,   # Tamaño del chunk
#         chunk_overlap=20, # Superposición entre chunks
#         length_function=len 
#     )

#     chunks = []
#     for section in sections:
#         if len(section) > 200:  # Si la sección es muy grande, dividirla en subpartes
#             sub_chunks = section_splitter.split_text(section)
#             chunks.extend(sub_chunks)
#         else:
#             chunks.append(section)

#     return chunks

# async def generate_product_embeddings(product: Product) -> None:
#     for lang, translation in product.translations.items():
#         name_text = translation.name
#         description_text = translation.description
#         description_vect_text = translation.destovect

#         # Generar embeddings para el nombre y la descripción del producto
#         name_embedding = embedder.embed_query(name_text)
#         description_embedding = embedder.embed_query(description_text)

#         # Dividir el texto en segmentos más pequeños de forma óptima
#         description_vect_segments = await split_text_optimally(description_vect_text)
#         description_vect_embeddings = embedder.embed_documents(description_vect_segments)

#         # Metadata general para relacionar con MongoDB (usamos el _id del producto)
#         metadata_base = {
#             "product_slug": str(product.slug),  # Asegúrate de tener `product.slug` en MongoDB
#             "language": lang,
#         }

#         # Crear una lista para almacenar todos los embeddings bajo el mismo vector_id
#         vectors_to_upsert = []

#         # 1. Embedding del nombre del producto
#         metadata_name = {**metadata_base, "type": "name"}
#         vectors_to_upsert.append((f"{product.slug}_{lang}", name_embedding, metadata_name))

#         # 2. Embedding de la descripción del producto
#         metadata_description = {**metadata_base, "type": "description"}
#         vectors_to_upsert.append((f"{product.slug}_{lang}", description_embedding, metadata_description))

#         # 3. Embeddings de los segmentos de la descripción
#         for i, segment_embedding in enumerate(description_vect_embeddings):
#             segment_metadata = {**metadata_base, "type": "description_segment", "segment": i}
#             vectors_to_upsert.append((f"{product.slug}_{lang}", segment_embedding, segment_metadata))

#         # Subir los embeddings a Pinecone bajo el mismo vector_id para cada idioma del producto
#         index.upsert(vectors=vectors_to_upsert)