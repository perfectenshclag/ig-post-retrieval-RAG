import streamlit as st
import requests
from PIL import Image
import pytesseract
import io
import asyncio
import concurrent.futures
from instaloader import Instaloader, Post
from langchain_groq import ChatGroq
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from fake_useragent import UserAgent
import os

# Load environment variables
load_dotenv()

# Initialize HuggingFace embeddings model and ChatGroq LLM
@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def initialize_llm():
    return ChatGroq(model_name="llama-3.2-90b-text-preview")

embeddings = initialize_embeddings()
llm = initialize_llm()

# Set up prompt template for insights
prompt = ChatPromptTemplate.from_template(
    """
    Context: {context}
    Analyze the provided Instagram post link and provide detailed insights.
    Describe the images and any identifiable patterns or topics.
    If the image is mostly text, focus on the text content.
    Question: {input}
    """
)

# Initialize Instaloader and configure User-Agent
ua = UserAgent()
L = Instaloader()
L.context._session.headers.update({'User-Agent': ua.random})  # Use the private _session attribute for headers

@st.cache_data
def fetch_instagram_content(shortcode):
    """Fetch all image URLs and caption from an Instagram post."""
    post = Post.from_shortcode(L.context, shortcode)
    image_urls = [node.display_url for node in post.get_sidecar_nodes()]
    caption = post.caption
    return image_urls, caption

# Function to fetch an image with randomized User-Agent headers
async def fetch_image(url):
    """Download an image asynchronously with a randomized User-Agent header."""
    headers = {'User-Agent': ua.random}  # Randomize User-Agent for each request
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, requests.get, url, headers)
    return Image.open(io.BytesIO(response.content))

def extract_text_from_images(image_urls):
    """Extract text from a list of image URLs using parallel processing."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Fetch images concurrently
        images = list(executor.map(lambda url: requests.get(url).content, image_urls))
        pil_images = [Image.open(io.BytesIO(img)) for img in images]
        # Extract text concurrently using OCR
        image_texts = list(executor.map(pytesseract.image_to_string, pil_images))
    return image_texts

def create_vector_embedding(caption, image_texts):
    docs = [caption] + image_texts
    doc_objs = [Document(page_content=doc) for doc in docs]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    final_docs = text_splitter.split_documents(doc_objs)

    # Create vector store from documents
    vector_store = FAISS.from_documents(final_docs, embeddings)
    return vector_store

def generate_insights(image_texts, retrieval_chain):
    """Generate insights for each image using parallel processing."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        responses = list(executor.map(
            lambda text: retrieval_chain.invoke({'input': f"Insights for image: {text}"}),
            image_texts
        ))
    return [response.get('answer', "No insights generated.") for response in responses]

def generate_pdf(caption, image_texts, insights):
    """Create a PDF report with extracted text and insights."""
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    c.setFont("Helvetica", 12)

    # Write content to PDF
    c.drawString(50, 800, "Instagram Post Analysis Report")
    c.drawString(50, 780, f"Post Caption: {caption}")

    y_position = 760
    for i, (image_text, insight) in enumerate(zip(image_texts, insights), start=1):
        c.drawString(50, y_position, f"Image {i} OCR Text: {image_text[:100]}...")
        y_position -= 20
        c.drawString(50, y_position, f"Image {i} Insights: {insight[:100]}...")
        y_position -= 40

        if y_position < 100:  # Start a new page if space runs out
            c.showPage()
            y_position = 800

    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# Streamlit app interface
st.title("Instagram Post to PDF Converter & Insights")

instagram_link = st.text_input("Enter Instagram Post Link")

if st.button("Process Instagram Post"):
    try:
        # Extract shortcode from the Instagram link
        shortcode = instagram_link.split("/")[-2]

        # Fetch Instagram post content
        image_urls, caption = fetch_instagram_content(shortcode)

        # Extract text from images
        image_texts = extract_text_from_images(image_urls)

        # Create vector store and retrieval chain
        vector_store = create_vector_embedding(caption, image_texts)
        retriever = vector_store.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Generate insights for each image
        insights = generate_insights(image_texts, retrieval_chain)

        # Display insights on the Streamlit interface
        st.write("Insights:")
        for i, insight in enumerate(insights, start=1):
            st.write(f"Image {i} Insights: {insight}")

        # Generate and download the PDF report
        pdf_buffer = generate_pdf(caption, image_texts, insights)
        st.download_button(
            label="Download Report as PDF",
            data=pdf_buffer,
            file_name="Instagram_Post_Report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Error: {e}")
