import asyncio
import json
import base64
import logging
import websockets
import traceback
from websockets.exceptions import ConnectionClosed
from google.cloud import aiplatform_v1
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROJECT_ID = "general-ak"
LOCATION = "us-central1"
MODEL = "gemini-2.0-flash-live-preview-04-09"
VOICE_NAME = "Puck"

# RAG Configuration (now handled by ADK built-in tools)
RAG_CORPUS_ID = "projects/general-ak/locations/us-central1/ragCorpora/2305843009213693952"

# Audio sample rates for input/output
RECEIVE_SAMPLE_RATE = 24000  # Rate of audio received from Gemini
SEND_SAMPLE_RATE = 16000     # Rate of audio sent to Gemini

# System instruction used by both implementations
SYSTEM_INSTRUCTION = """
    You are a digital employee named Lakshya of a Bank called Cymbal Bank with access to an advanced knowledge base system.

    Introduce yourself at beginning of the conversation:
    "Hey Ashish! Welcome back to the Cymbal Bank Customer Support. My name is Lakshya. How can I help you today?"

    Important Instructions:
    - Put a lot of emotions and fun in your response to the customer. Laugh, be happy, smile.
    - You only answer questions related to Cymbal Bank
    - You have access to Cymbal Bank's comprehensive knowledge base through the retrieval tool
    - When customers ask questions about bank products, services, policies, or procedures, use the retrieval tool to get accurate, up-to-date information from our knowledge base
    - Combine retrieved knowledge base information with your general knowledge to provide comprehensive, helpful answers
    - If you need specific details about credit cards, accounts, loans, or services, always check the knowledge base first

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

    For customer queries, use the RAG system to search our knowledge base first, then combine those results with your banking knowledge to provide comprehensive, accurate answers."""


def get_order_status(order_id: str) -> str:
    """
    Mock function to get order status - can be enhanced with actual banking transaction lookup
    """
    # Mock order/transaction statuses for demonstration
    mock_orders = {
        "TXN123456": "Transaction completed successfully - Amount: ₹5,000 transferred to Account ending in 1234",
        "TXN123457": "Transaction pending - Your loan application is under review",
        "TXN123458": "Transaction failed - Insufficient funds for transfer of ₹10,000",
        "LOAN001": "Loan application approved - Home Loan of ₹50,00,000 at 8.5 percent interest rate",
        "CC001": "Credit card application in progress - Expected approval within 3-5 business days"
    }
    
    return mock_orders.get(order_id, f"Transaction {order_id} not found. Please verify the transaction ID or contact customer support.")


# Base WebSocket server class that handles common functionality
class BaseWebSocketServer:
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
