import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    # 'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'multimodal_datamodal'),
    # 'database': os.getenv('DB_NAME', 'document_intelligence'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# Validate required fields
if not DB_CONFIG['user'] or not DB_CONFIG['password']:
    raise ValueError("DB_USER and DB_PASSWORD must be set in .env file")