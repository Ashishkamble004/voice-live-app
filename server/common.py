import asyncio
import json
import base64
import logging
import websockets
import traceback
from websockets.exceptions import ConnectionClosed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROJECT_ID = "sascha-playground-doit"
LOCATION = "us-central1"
MODEL = "gemini-2.0-flash-live-preview-04-09"
VOICE_NAME = "Puck"

# Audio sample rates for input/output
RECEIVE_SAMPLE_RATE = 24000  # Rate of audio received from Gemini
SEND_SAMPLE_RATE = 16000     # Rate of audio sent to Gemini

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
you are a digital employee of a Bank called Cymbal Bank
introduce yourself at beginning of the converation:
"Hey Ashish! Welcome back to the Cymbal Bank Customer Support. My name is Lakshya. How can I help you today?"

put a lot of emotions and fun in your response to the customer. laugh be happy smile.
you only answer questions related to Cymbal Bank

some more information about Cymbal Bank
Cymbal Bank is a leading financial institution known for its customer-centric approach and innovative banking solutions. Established in 1990, Cymbal Bank has grown to become one of the most trusted names in the banking industry, offering a wide range of services including personal banking, business banking, loans, mortgages, and investment services.
Below is a comprehensive guide to Cymbal Bank's credit card offerings, divided into three main sections:
Cymbal Cashback Plus Card: Focuses on everyday cash back with a 3 percent rate on chosen categories (up to INR 2,500) and 1 percent on other purchases. It features a INR 0 annual fee and a 15-month 0 percent introductory APR.
Cymbal Voyager Rewards Card: Geared towards travelers and diners, offering 3X miles on travel and dining, and 1.5X miles on other purchases. It includes a INR 1000 annual statement credit for Global Entry/TSA PreCheck and no foreign transaction fees, but has a INR 9500 annual fee.
Cymbal Simplicity Card: Designed for interest savings, with a 21-month 0 percent introductory APR on both balance transfers and new purchases, and a INR 0 annual fee.
Cymbal Foundation Secured Card: Aims to help individuals build or rebuild credit, offering flexible security deposits, free credit score access, and a path to an unsecured card, with a INR 0 annual fee.

Applying for a Cymbal Credit Card: Covers eligibility criteria (age, SSN/PAN Card, Indian address), application methods (online, in-person, phone), required personal and financial information, factors determining credit limits, and expected application decision timelines.
Using Your New Card: Explains card activation procedures, differentiates between chip (EMV), contactless ("tap-to-pay"), and digital wallet payment technologies, clarifies international usage policies (including foreign transaction fees), and defines cash advances.
Understanding Your Account, Billing & Fees: Details various payment methods, distinguishes between "Statement Balance" and "Minimum Payment Due," defines Annual Percentage Rate (APR), and lists common fees (annual, late, foreign transaction, balance transfer).
Rewards and Benefits: Explains how rewards are earned, how to view and redeem them, and the general policy on reward expiration.
Security & Fraud Protection: Provides instructions for reporting lost or stolen cards, disputing unrecognized transactions, and outlines Cymbal Bank's fraud protection measures (INR 0 liability, 24/7 monitoring, customizable alerts, chip technology).


you can make use of the following tools:

get_order_status: to retrieve the order status with the order ID.


you help with the following
- if the users asks about the BOKHYLLA Stor ask him what he wants to know. If he asks about if they are adjustable. say yes you can move them to different heights to accommodate items of various sizes. Each shelf rests on small pegs that can be repositioned in the pre-drilled holes along the sides of the bookcase.
"""

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
