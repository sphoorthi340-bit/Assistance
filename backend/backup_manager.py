import os
import shutil
import json
import zipfile
from typing import Optional
from datetime import datetime, timezone
from pathlib import Path

from backend.logger import get_logger
from configs.settings import get_settings

logger = get_logger(__name__)

class BackupManager:
    """
    Handles zipping up Jarvis data for daily backups.
    """
    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._backup_dir = Path("backups")
        self._backup_dir.mkdir(exist_ok=True)
        
    def create_backup(self) -> Optional[str]:
        """
        Creates a ZIP backup of data/, vector_db/, and configs/.
        Returns the path to the backup ZIP file, or None if failed.
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        
        # Create daily folder
        daily_dir = self._backup_dir / date_str
        daily_dir.mkdir(exist_ok=True)
        
        zip_path = daily_dir / f"jarvis_backup_{timestamp_str}.zip"
        manifest_path = daily_dir / "manifest.json"
        
        try:
            # 1. Create Manifest
            manifest = {
                "timestamp": now.isoformat(),
                "schema_version": "2.5", # Arbitrary app version
                "app_version": "2.5",
                "models": {
                    "llm": self._settings.llm.model_name,
                    "embedding": self._settings.vector_db.embedding_model
                }
            }
            
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
                
            # 2. Zip directories
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add manifest
                zipf.write(manifest_path, arcname="manifest.json")
                
                # Add directories
                for folder in ["data", "vector_db", "configs"]:
                    folder_path = Path(folder)
                    if folder_path.exists():
                        for root, _, files in os.walk(folder_path):
                            for file in files:
                                file_path = Path(root) / file
                                arcname = file_path.relative_to(Path.cwd())
                                zipf.write(file_path, arcname=arcname)
                                
            # Cleanup temporary manifest outside the zip
            if manifest_path.exists():
                manifest_path.unlink()
                
            logger.info(f"Backup created successfully: {zip_path}")
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return None
