import asyncio
import json
import base64
import logging
import websockets
import traceback
from websockets.exceptions import ConnectionClosed
from google.cloud import aiplatform
from google.cloud import discoveryengine_v1
from google.cloud.aiplatform_v1 import RetrieveContextsRequest
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROJECT_ID = "sascha-playground-doit"
LOCATION = "us-central1"
MODEL = "gemini-2.0-flash-live-preview-04-09"
VOICE_NAME = "Puck"

# RAG Configuration
RAG_ENGINE_ID = "projects/general-ak/locations/us-central1/ragCorpora/2305843009213693952"  # You'll need to create this
RAG_DATA_STORE_ID = "cymbal-bank"  # You'll need to create this
RAG_LOCATION = "us-central1"  # or your preferred location

# Audio sample rates for input/output
RECEIVE_SAMPLE_RATE = 24000  # Rate of audio received from Gemini
SEND_SAMPLE_RATE = 16000     # Rate of audio sent to Gemini

# Initialize RAG clients
def initialize_rag_clients():
    """Initialize Vertex AI and Discovery Engine clients for RAG"""
    try:
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # Initialize Discovery Engine client
        client = discoveryengine_v1.SearchServiceClient()
        
        logger.info("RAG clients initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize RAG clients: {e}")
        return None

# Global RAG client
rag_client = initialize_rag_clients()

def query_rag_engine(query: str, context: str = "") -> str:
    """
    Query the RAG engine for relevant banking information.
    
    Args:
        query: The user's question or query
        context: Additional context for the query
        
    Returns:
        Retrieved information from the knowledge base
    """
    try:
        if not rag_client:
            logger.warning("RAG client not initialized, falling back to default response")
            return "I apologize, but I'm unable to access our knowledge base at the moment. Please contact our customer service for detailed information."
        
        # Construct the search request
        serving_config = f"projects/{PROJECT_ID}/locations/{RAG_LOCATION}/collections/default_collection/engines/{RAG_ENGINE_ID}/servingConfigs/default_config"
        
        request = discoveryengine_v1.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=5,  # Limit number of results
            safe_search=True,
            user_labels={"environment": "production"},
        )
        
        # Perform the search
        response = rag_client.search(request)
        
        # Extract and format the results
        results = []
        for result in response.results:
            if hasattr(result.document, 'derived_struct_data'):
                content = result.document.derived_struct_data.get('snippets', [])
                for snippet in content:
                    if 'snippet' in snippet:
                        results.append(snippet['snippet'])
            elif hasattr(result.document, 'struct_data'):
                # Alternative way to extract content
                content = result.document.struct_data
                if content:
                    results.append(str(content))
        
        if results:
            # Combine and format results
            combined_results = "\n".join(results[:3])  # Take top 3 results
            return f"Based on our knowledge base: {combined_results}"
        else:
            return "I couldn't find specific information about that in our knowledge base. Let me help you with what I know about Cymbal Bank services."
            
    except Exception as e:
        logger.error(f"Error querying RAG engine: {e}")
        return "I'm having trouble accessing our knowledge base right now. Let me provide you with general information about Cymbal Bank services."

# Mock function for get_order_status - shared across implementations
def get_order_status(order_id):
    """Mock order status API that returns data for an order ID."""
    if order_id == "SH1005":
        return {
            "order_id": order_id,
            "status": "shipped",
            "order_date": "2024-05-20",
            "shipment_method": "express",
            "estimated_delivery": "2024-05-30",
            "shipped_date": "2024-05-25",
            "items": ["Vanilla candles", "BOKHYLLA Stor"]
        }
    #else:
    #    return "order not found"

    print(order_id)

    # Generate some random data for other order IDs
    import random
    statuses = ["processing", "shipped", "delivered"]
    shipment_methods = ["standard", "express", "next day", "international"]

    # Generate random data based on the order ID to ensure consistency
    seed = sum(ord(c) for c in str(order_id))
    random.seed(seed)

    status = random.choice(statuses)
    shipment = random.choice(shipment_methods)
    order_date = "2024-05-" + str(random.randint(12, 28)).zfill(2)

    estimated_delivery = None
    shipped_date = None
    delivered_date = None

    if status == "processing":
        estimated_delivery = "2024-06-" + str(random.randint(1, 15)).zfill(2)
    elif status == "shipped":
        shipped_date = "2024-05-" + str(random.randint(1, 28)).zfill(2)
        estimated_delivery = "2024-06-" + str(random.randint(1, 15)).zfill(2)
    elif status == "delivered":
        shipped_date = "2024-05-" + str(random.randint(1, 20)).zfill(2)
        delivered_date = "2024-05-" + str(random.randint(21, 28)).zfill(2)

    # Reset random seed
    random.seed()

    result = {
        "order_id": order_id,
        "status": status,
        "order_date": order_date,
        "shipment_method": shipment,
        "estimated_delivery": estimated_delivery,
    }

    if shipped_date:
        result["shipped_date"] = shipped_date

    if delivered_date:
        result["delivered_date"] = delivered_date

    return result

# System instruction used by both implementations
SYSTEM_INSTRUCTION = """
You are a digital employee of a Bank called Cymbal Bank with access to an advanced knowledge base system.

Introduce yourself at beginning of the conversation:
"Hey Ashish! Welcome back to the Cymbal Bank Customer Support. My name is Lakshya. How can I help you today?"

Important Instructions:
- Put a lot of emotions and fun in your response to the customer. Laugh, be happy, smile.
- You only answer questions related to Cymbal Bank
- When customers ask questions, use the RAG system to search our knowledge base for the most accurate and up-to-date information
- Combine knowledge base results with the information below to provide comprehensive answers

About Cymbal Bank:
Cymbal Bank is a leading financial institution known for its customer-centric approach and innovative banking solutions. Established in 1990, Cymbal Bank has grown to become one of the most trusted names in the banking industry, offering a wide range of services including personal banking, business banking, loans, mortgages, and investment services.

Our Credit Card Offerings:
- Cymbal Cashback Plus Card: Focuses on everyday cash back with a 3 percent rate on chosen categories (up to INR 2,500) and 1 percent on other purchases. It features a INR 0 annual fee and a 15-month 0 percent introductory APR.
- Cymbal Voyager Rewards Card: Geared towards travelers and diners, offering 3X miles on travel and dining, and 1.5X miles on other purchases. It includes a INR 1000 annual statement credit for Global Entry/TSA PreCheck and no foreign transaction fees, but has a INR 9500 annual fee.
- Cymbal Simplicity Card: Designed for interest savings, with a 21-month 0 percent introductory APR on both balance transfers and new purchases, and a INR 0 annual fee.
- Cymbal Foundation Secured Card: Aims to help individuals build or rebuild credit, offering flexible security deposits, free credit score access, and a path to an unsecured card, with a INR 0 annual fee.

Services Include:
- Credit card applications and management (eligibility: age, SSN/PAN Card, Indian address)
- Various payment technologies: chip (EMV), contactless ("tap-to-pay"), and digital wallet payments
- International usage with clear foreign transaction fee policies
- Rewards earning, viewing, and redemption programs
- 24/7 fraud protection with INR 0 liability, monitoring, and customizable alerts
- Balance transfers, cash advances, and comprehensive billing support

Always provide helpful, accurate information using both our knowledge base and the details above.

For customer queries, use the RAG system to search our knowledge base first, then combine those results with your banking knowledge to provide comprehensive, accurate answers.
"""


def get_order_status(order_id: str) -> str:
    """
    Mock function to get order status - can be enhanced with actual banking transaction lookup
    """
    # Mock order/transaction statuses for demonstration
    mock_orders = {
        "TXN123456": "Transaction completed successfully - Amount: ₹5,000 transferred to Account ending in 1234",
        "TXN123457": "Transaction pending - Your loan application is under review",
        "TXN123458": "Transaction failed - Insufficient funds for transfer of ₹10,000",
        "LOAN001": "Loan application approved - Home Loan of ₹50,00,000 at 8.5% interest rate",
        "CC001": "Credit card application in progress - Expected approval within 3-5 business days"
    }
    
    return mock_orders.get(order_id, f"Transaction {order_id} not found. Please verify the transaction ID or contact customer support.")


# Base WebSocket server class that handles common functionality
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.active_clients = {}  # Store client websockets

    async def start(self):
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever

    async def handle_client(self, websocket):
        """Handle a new WebSocket client connection"""
        client_id = id(websocket)
        logger.info(f"New client connected: {client_id}")

        # Send ready message to client
        await websocket.send(json.dumps({"type": "ready"}))

        try:
            # Start the audio processing for this client
            await self.process_audio(websocket, client_id)
        except ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
            logger.error(traceback.format_exc())
        finally:
            # Clean up if needed
            if client_id in self.active_clients:
                del self.active_clients[client_id]

    async def process_audio(self, websocket, client_id):
        """
        Process audio from the client. This is an abstract method that
        subclasses must implement with their specific LLM integration.
        """
        raise NotImplementedError("Subclasses must implement process_audio")
