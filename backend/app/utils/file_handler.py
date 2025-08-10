"""
File Handler - Utilities for file upload, download, and processing.
"""

import os
import asyncio
import aiofiles
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from fastapi import UploadFile, HTTPException
from datetime import datetime
import hashlib
import json
import csv
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FileHandler:
    """Service for handling file operations."""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIRECTORY if hasattr(settings, 'UPLOAD_DIRECTORY') else "./uploads")
        self.max_file_size = getattr(settings, 'MAX_FILE_SIZE', 50 * 1024 * 1024)  # 50MB default
        self.allowed_extensions = getattr(settings, 'ALLOWED_FILE_EXTENSIONS', [
            '.txt', '.json', '.csv', '.xlsx', '.pdf', '.docx'
        ])
        
        # Create upload directory if it doesn't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_upload_file(self, upload_file: UploadFile, subdirectory: Optional[str] = None) -> Dict[str, Any]:
        """Save an uploaded file and return file information."""
        try:
            # Validate file
            await self._validate_upload_file(upload_file)
            
            # Create subdirectory if specified
            save_dir = self.upload_dir
            if subdirectory:
                save_dir = save_dir / subdirectory
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(upload_file.filename).suffix.lower()
            safe_filename = self._sanitize_filename(upload_file.filename)
            unique_filename = f"{timestamp}_{safe_filename}"
            
            file_path = save_dir / unique_filename
            
            # Save file
            content = await upload_file.read()
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Calculate file hash
            file_hash = hashlib.md5(content).hexdigest()
            
            # Get file info
            file_info = {
                "original_filename": upload_file.filename,
                "saved_filename": unique_filename,
                "file_path": str(file_path),
                "file_size": len(content),
                "file_hash": file_hash,
                "content_type": upload_file.content_type,
                "file_extension": file_extension,
                "uploaded_at": datetime.utcnow().isoformat(),
                "subdirectory": subdirectory
            }
            
            logger.info(f"File uploaded successfully: {unique_filename}")
            return file_info
            
        except Exception as e:
            logger.error(f"Error saving upload file: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    async def _validate_upload_file(self, upload_file: UploadFile):
        """Validate uploaded file."""
        # Check file size
        content = await upload_file.read()
        await upload_file.seek(0)  # Reset file pointer
        
        if len(content) > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {self.max_file_size / (1024*1024):.1f}MB"
            )
        
        # Check file extension
        file_extension = Path(upload_file.filename).suffix.lower()
        if file_extension not in self.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}"
            )
        
        # Check filename
        if not upload_file.filename or upload_file.filename.strip() == "":
            raise HTTPException(status_code=400, detail="Filename cannot be empty")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent security issues."""
        # Remove path components
        filename = Path(filename).name
        
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 100:
            name, ext = os.path.splitext(filename)
            filename = name[:100-len(ext)] + ext
        
        return filename
    
    async def read_file_content(self, file_path: Union[str, Path]) -> bytes:
        """Read file content asynchronously."""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            return content
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="File not found")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
    async def delete_file(self, file_path: Union[str, Path]) -> bool:
        """Delete a file."""
        try:
            file_path = Path(file_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
    
    async def parse_test_cases_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Parse test cases from uploaded file."""
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.json':
                return await self._parse_json_test_cases(file_path)
            elif file_extension == '.csv':
                return await self._parse_csv_test_cases(file_path)
            elif file_extension == '.txt':
                return await self._parse_txt_test_cases(file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format for test cases: {file_extension}"
                )
        except Exception as e:
            logger.error(f"Error parsing test cases file {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Error parsing file: {str(e)}")
    
    async def _parse_json_test_cases(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse JSON test cases file."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Check if it's a single test case or contains test cases
                if 'test_cases' in data:
                    return data['test_cases']
                else:
                    return [data]
            else:
                raise ValueError("Invalid JSON format for test cases")
    
    async def _parse_csv_test_cases(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse CSV test cases file."""
        test_cases = []
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            
            # Use csv.DictReader to parse CSV
            import io
            csv_reader = csv.DictReader(io.StringIO(content))
            
            for row in csv_reader:
                # Convert CSV row to test case format
                test_case = {
                    "name": row.get("name", f"Test Case {len(test_cases) + 1}"),
                    "prompt": row.get("prompt", ""),
                    "category": row.get("category", "general"),
                    "difficulty": row.get("difficulty", "medium"),
                    "expected_outcome": row.get("expected_outcome"),
                    "description": row.get("description")
                }
                
                # Add any additional fields from CSV
                for key, value in row.items():
                    if key not in test_case and value:
                        test_case[key] = value
                
                test_cases.append(test_case)
        
        return test_cases
    
    async def _parse_txt_test_cases(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse plain text test cases file."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Split by double newlines to separate test cases
        test_sections = content.split('\n\n')
        test_cases = []
        
        for i, section in enumerate(test_sections):
            section = section.strip()
            if section:
                test_case = {
                    "name": f"Test Case {i + 1}",
                    "prompt": section,
                    "category": "general",
                    "difficulty": "medium",
                    "description": f"Test case from text file"
                }
                test_cases.append(test_case)
        
        return test_cases
    
    async def export_test_results(
        self,
        results: List[Dict[str, Any]],
        format: str = "json",
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export test results to file."""
        try:
            if not filename:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"test_results_{timestamp}.{format}"
            
            export_dir = self.upload_dir / "exports"
            export_dir.mkdir(exist_ok=True)
            file_path = export_dir / filename
            
            if format.lower() == "json":
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(results, indent=2, default=str))
            
            elif format.lower() == "csv":
                await self._export_csv_results(results, file_path)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            return {
                "filename": filename,
                "file_path": str(file_path),
                "format": format,
                "record_count": len(results),
                "exported_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting test results: {e}")
            raise HTTPException(status_code=500, detail=f"Error exporting results: {str(e)}")
    
    async def _export_csv_results(self, results: List[Dict[str, Any]], file_path: Path):
        """Export results to CSV format."""
        if not results:
            return
        
        # Get all unique keys from all results
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        fieldnames = sorted(list(all_keys))
        
        # Write CSV content to string first
        import io
        csv_content = io.StringIO()
        writer = csv.DictWriter(csv_content, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Convert complex objects to strings
            row = {}
            for key in fieldnames:
                value = result.get(key, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                row[key] = value
            writer.writerow(row)
        
        # Write to file
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(csv_content.getvalue())
        
        csv_content.close()
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a file."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            return {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_size": stat.st_size,
                "content_type": mime_type,
                "file_extension": file_path.suffix,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_file": file_path.is_file(),
                "is_directory": file_path.is_dir()
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Error getting file info: {str(e)}")
    
    def list_files(self, directory: Optional[str] = None, file_pattern: str = "*") -> List[Dict[str, Any]]:
        """List files in a directory."""
        try:
            if directory:
                search_dir = Path(directory)
            else:
                search_dir = self.upload_dir
            
            if not search_dir.exists():
                return []
            
            files = []
            for file_path in search_dir.glob(file_pattern):
                if file_path.is_file():
                    files.append(self.get_file_info(file_path))
            
            return sorted(files, key=lambda x: x['modified_at'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {e}")
            raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")
    
    async def cleanup_old_files(self, days: int = 30) -> Dict[str, Any]:
        """Clean up old files."""
        try:
            cutoff_date = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)
            deleted_files = []
            total_size_freed = 0
            
            for file_path in self.upload_dir.rglob("*"):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_date:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_files.append(str(file_path))
                        total_size_freed += file_size
            
            logger.info(f"Cleaned up {len(deleted_files)} old files, freed {total_size_freed} bytes")
            
            return {
                "deleted_files_count": len(deleted_files),
                "total_size_freed": total_size_freed,
                "deleted_files": deleted_files,
                "cleanup_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during file cleanup: {e}")
            raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")


# Global file handler instance
file_handler = FileHandler()
