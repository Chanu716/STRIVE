import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def download_assets_from_supabase() -> None:
    """Download ML models and graph files from Supabase Storage on startup."""
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    bucket_name = os.getenv("SUPABASE_BUCKET_NAME", "strive-assets")
    
    if not supabase_url or not supabase_key:
        logger.warning("Supabase credentials not found. Skipping asset download.")
        return

    try:
        from supabase import create_client, Client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Files to download and their local destination paths
        assets = [
            ("model.pkl", os.getenv("MODEL_PATH", "models/model.pkl")),
            ("feature_config.json", os.getenv("FEATURE_CONFIG_PATH", "models/feature_config.json")),
            ("road_network.graphml.gz", os.getenv("GRAPH_PATH", "data/raw/road_network.graphml"))
        ]
        
        for file_name, local_path_str in assets:
            local_path = Path(local_path_str)
            
            # Create parent directories if they don't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Don't download if the file already exists locally (speeds up local dev)
            if local_path.exists():
                logger.info(f"Asset already exists locally, skipping download: {local_path}")
                continue
                
            logger.info(f"Downloading {file_name} from Supabase Storage to {local_path}")
            try:
                # In Supabase Storage, we expect files to be in the root of the bucket
                response = supabase.storage.from_(bucket_name).download(file_name)
                
                if file_name.endswith('.gz'):
                    import gzip
                    decompressed_data = gzip.decompress(response)
                    with open(local_path, "wb") as f:
                        f.write(decompressed_data)
                else:
                    with open(local_path, "wb") as f:
                        f.write(response)
                        
                logger.info(f"Successfully downloaded {file_name}")
            except Exception as e:
                logger.error(f"Failed to download {file_name} from Supabase: {e}")
                
    except ImportError:
        logger.warning("supabase package not installed. Skipping asset download.")
    except Exception as e:
        logger.error(f"Error during Supabase asset download: {e}")
