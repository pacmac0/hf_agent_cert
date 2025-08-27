"""
Clean Gemini-based agent with LangGraph for the HF certification.
Optimized for multimodal inputs and Gemini capabilities.
"""

from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from google.genai import Client, types
from langsmith import traceable
import logging

from .prompts import Prompts
from .multimodal_content_handler import MultiModalContentHandler, URLDetector
from .tools import CUSTOM_TOOLS, GOOGLE_GEMINI_TOOLS
from .config import Config

from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger(__name__)


class AgentState(Dict):
    """Agent state for LangGraph."""
    question_text: str
    task_id: Optional[str]
    file_name: Optional[str] 
    detected_urls: List[str]
    multimodal_content_parts: List[types.Part]
    final_answer: str
    

class Agent:
    """Research agent with multimodal support."""

    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_base_url: Optional[str] = None,
                 model_name: Optional[str] = None):
        """Initialize the Gemini agent."""
        # Load configuration
        self.config = Config()
        
        # Override config with provided values
        if api_key:
            self.config.gemini_api_key = api_key
        if api_base_url:
            self.config.api_base_url = api_base_url
        if model_name:
            self.config.model_name = model_name
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration")
        
        # Setup LangSmith
        self.config.setup_langsmith()
        
        # Set instance variables
        self.api_key = self.config.gemini_api_key
        self.api_base_url = self.config.api_base_url
        self.model_name = self.config.model_name
        
        self.genai_client = Client(api_key=self.api_key)
        self.multimodal_content_handler = MultiModalContentHandler(
            genai_client=self.genai_client, 
            api_base_url=self.api_base_url
        )
        
        # Build graph
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph graph."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("analyze", self._analyze_input)
        graph.add_node("agent", self._call_agent)
        graph.add_node("cleanup", self._cleanup_files)
        
        # Define flow
        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "agent")
        graph.add_edge("agent", "cleanup")
        graph.add_edge("cleanup", END)
        
        return graph
    
    @traceable(name="analyze_input")
    def _analyze_input(self, state: AgentState) -> AgentState:
        """Analyze input and prepare multimodal content."""
        logger.debug("Analyzing input for multimodal content")
        
        # Extract text content
        text_content = str(state.get("question_text"))
        
        # Detect URLs
        detected_urls = URLDetector.extract_urls(text_content)
        
        # Build multimodal content if we have additional resources
        task_id = state.get("task_id")
        file_name = state.get("file_name")
        # Build enhanced multimodal content
        multimodal_content_parts = self.multimodal_content_handler.build_multimodal_content(
                text=text_content,
                task_id=task_id,
                file_name=file_name,
                detected_urls=detected_urls
            )
        
        return {
            "multimodal_content_parts": multimodal_content_parts,
            "detected_urls": detected_urls
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _generate_content_with_retry(self, contents: List[types.Part]) -> types.GenerateContentResponse:
        """Generate content with simple retry logic."""
        return self.genai_client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction = Prompts.get_system_prompt(),
                tools = GOOGLE_GEMINI_TOOLS,
                temperature = self.config.temperature,
                max_output_tokens = self.config.max_output_tokens,
                thinking_config = types.ThinkingConfig(
                    includeThoughts=True,
                    thinking_budget=-1
                ),
            )
        )

    @traceable(name="call_gemini_agent", metadata={"model": "gemini", "provider": "google"})
    def _call_agent(self, state: AgentState) -> AgentState:
        """Call the Gemini model."""
        logger.debug("Calling Gemini model")
        
        # Call the Gemini model with retry logic
        response = self._generate_content_with_retry(state.get("multimodal_content_parts"))

        return {"final_answer": response.text}

    @traceable(name="cleanup_files")
    def _cleanup_files(self, state: AgentState) -> AgentState:
        """Clean up (delete) any files uploaded to Gemini."""
        logger.debug("Cleaning up uploaded files")
        
        files = self.genai_client.files.list()
        for file in files:
            try:
                logger.info(f"Deleting file: {file.name}")
                self.genai_client.files.delete(name=file.name)
            except Exception as e:
                logger.error(f"Failed to delete file {file.name}: {e}")
        

        print(self.genai_client.files.list())
        return {}

    @traceable(name="process_question", metadata={"agent_type": "gemini"})
    def process_question(self, question_data: Dict[str, Any]) -> str:
        """
        Process a question and return the answer.
        
        Args:
            question_data: Dict with question data
        
        Returns:
            Answer string
        """
        logger.info(f"Processing question: {question_data}")
        # Parse input
        question_text = question_data.get("question", "")
        task_id = question_data.get("task_id")
        file_name = question_data.get("file_name")
        
        # Create initial state
        initial_state = {
            "question_text": question_text,
            "task_id": task_id,
            "file_name": file_name,
            "detected_urls": [],
            "multimodal_content_parts": [],
            "final_answer": ""
        }
        
        try:
            # Run the graph
            result = self.compiled_graph.invoke(initial_state, config={"recursion_limit": self.config.recursion_limit})
            
            return str(result.get("final_answer", ""))
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    def __call__(self, question_input: Dict[str, Any]) -> str:
        """Make the agent callable."""
        return self.process_question(question_input)
