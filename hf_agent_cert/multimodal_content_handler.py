"""
Message builder for Gemini API multimodal inputs.
Handles proper formatting of text, images, and other media types.
"""

import io
from typing import Dict, List, Any, Optional
import re
from urllib.parse import urlparse
from pathlib import Path
import mimetypes
import requests
from google.genai import types
from langsmith import traceable
import logging

logger = logging.getLogger(__name__)


class MultiModalContentHandler:
    """Handles multimodal content formatting for API requests with proper media type support."""
    
    # Gemini supported image formats
    SUPPORTED_IMAGE_FORMATS = {'png', 'jpeg', 'jpg', 'webp', 'heic', 'heif'}
    SUPPORTED_AUDIO_FORMATS = {'wav', 'mp3', 'flac', 'aac', 'm4a', 'ogg', 'opus'}
    SUPPORTED_VIDEO_FORMATS = {'mp4', 'mpeg', 'mov', 'avi', 'flv', 'mpg', 'webm', 'wmv', '3gp', '3gpp'}
    
    def __init__(self, genai_client, api_base_url: str = "https://agents-course-unit4-scoring.hf.space"):
        self.api_base_url = api_base_url.rstrip('/')
        self.genai_client = genai_client
        
    @traceable(name="build_multimodal_content")
    def build_multimodal_content(self, 
                                text: str,
                                task_id: Optional[str] = None,
                                file_name: Optional[str] = None,
                                detected_urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Build multimodal content parts for Gemini API.
        
        Returns a list of content parts that can be used in a Gemini message.
        """
        logger.debug(f"Building multimodal content - task_id: {task_id}, file: {file_name}")
        content_parts = []
        
        # Always add the main text content first
        content_parts.append(types.Part(text=f'{text}'))
        
        # Handle file attachment from API
        if task_id and file_name:
            file_part = self._create_file_part(task_id, file_name)
            if file_part:
                content_parts.append(file_part)
        
        # Handle URLs detected in text
        if detected_urls:
            url_parts = self._process_urls(detected_urls)
            content_parts.extend(url_parts)

        return content_parts

    def _create_file_part(self, task_id: str, file_name: str) -> Optional[Dict[str, Any]]:
        """Create a file content part for Gemini API using file_data for URL-based media."""
        # Construct resource URL - Fixed to match API docs
        resource_url = f"{self.api_base_url}/files/{task_id}"
        
        try:
            # Download file from the resource URL
            logger.debug(f"Downloading file from: {resource_url}")
            response = requests.get(resource_url, timeout=30)
            response.raise_for_status()
            file_obj = io.BytesIO(response.content)
            
            # Upload the file to Gemini
            uploaded_file = self.genai_client.files.upload(
                file=file_obj,
                config = types.UploadFileConfig(
                    display_name=file_name,
                    mime_type=mimetypes.guess_type(file_name)[0]
                )
            )

            logger.info(f"File uploaded to Gemini: {uploaded_file.name}, uri: {uploaded_file.uri}")

            return uploaded_file
            # types.Part.from_uri(
            #     file_uri=uploaded_file.uri,
            #     mime_type=mimetypes.guess_type(file_name)[0]
            # )

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file {file_name} from {resource_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
            return None 
    
    def _process_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process detected URLs and create appropriate content parts."""
        url_parts = []
        
        for url in urls:
            url_parts.append(types.Part(file_data=types.FileData(file_uri=url)))
        return url_parts

class URLDetector:
    """Detects and extracts URLs/URIs from text."""
    
    # Comprehensive URL regex pattern
    URL_PATTERN = re.compile(
        r'(?:(?:https?|ftp):\/\/|www\.)[^\s<>"{}|\\^`\[\]]+(?:\.[^\s<>"{}|\\^`\[\]]+)*'
        r'|(?:[\w\-]+\.)+[a-zA-Z]{2,}(?:\/[^\s<>"{}|\\^`\[\]]*)?'
        r'|(?:file:\/\/[^\s<>"{}|\\^`\[\]]+)'
        r'|(?:data:[^\s<>"{}|\\^`\[\]]+)',
        re.IGNORECASE
    )
    
    @classmethod
    @traceable(name="extract_urls")
    def extract_urls(cls, text: str) -> List[str]:
        """Extract all URLs from text."""
        if not text:
            return []
        
        urls = []
        matches = cls.URL_PATTERN.findall(text)
        
        for match in matches:
            # Clean and validate URL
            url = match.strip()
            
            # Add protocol if missing
            if not re.match(r'^[a-zA-Z][a-zA-Z\d+\-.]*:', url):
                if url.startswith('www.'):
                    url = 'https://' + url
                elif '.' in url and not url.startswith('.'):
                    url = 'https://' + url
            
            # Validate URL structure
            try:
                parsed = urlparse(url)
                if parsed.scheme and (parsed.netloc or parsed.scheme == 'file'):
                    urls.append(url)
            except Exception:
                continue
        
        return list(set(urls))  # Remove duplicates
    
    @classmethod
    def contains_urls(cls, text: str) -> bool:
        """Check if text contains any URLs."""
        return bool(cls.URL_PATTERN.search(text))
