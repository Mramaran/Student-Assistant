import logging
import os
from datetime import datetime

def setup_logging(log_file="system.log"):
    """Initialize professional rotating logs."""
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("RAGAssistant")
    logger.info("Logging system initialized.")
    return logger

# Singleton instance
logger = setup_logging()
