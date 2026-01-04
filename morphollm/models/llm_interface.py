"""
LLM Interface for MorphoLLM.

Provides unified interface to various LLM backends:
- OpenAI GPT-4
- Anthropic Claude
- Local models (LLaMA, etc.)

Author: H M Shujaat Zaheer
"""

import os
from typing import Optional, Dict, List
from dataclasses import dataclass
import json


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""
    backend: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: Optional[str] = None


class LLMInterface:
    """
    Unified interface to LLM backends.
    
    Supports:
        - OpenAI GPT-4/GPT-3.5
        - Anthropic Claude
        - Local models via Ollama/vLLM
        - Mock backend for testing
    
    Example:
        >>> llm = LLMInterface(backend="gpt-4")
        >>> response = llm.generate("What morphology is best for grasping fragile objects?")
    """
    
    def __init__(
        self,
        backend: str = "gpt-4",
        config: Optional[LLMConfig] = None
    ):
        """
        Initialize LLM interface.
        
        Args:
            backend: LLM backend to use
            config: Optional configuration
        """
        self.config = config or LLMConfig(backend=backend)
        self.backend = backend
        
        # Initialize backend
        self._init_backend()
        
    def _init_backend(self):
        """Initialize the selected backend."""
        if self.backend in ["gpt-4", "gpt-3.5-turbo"]:
            self._init_openai()
        elif self.backend in ["claude", "claude-3"]:
            self._init_anthropic()
        elif self.backend == "mock":
            self._client = None
        else:
            # Try local backend
            self._init_local()
    
    def _init_openai(self):
        """Initialize OpenAI backend."""
        try:
            import openai
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=api_key)
        except ImportError:
            print("Warning: openai package not installed. Using mock backend.")
            self._client = None
    
    def _init_anthropic(self):
        """Initialize Anthropic backend."""
        try:
            import anthropic
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("Warning: anthropic package not installed. Using mock backend.")
            self._client = None
    
    def _init_local(self):
        """Initialize local LLM backend."""
        try:
            import requests
            self._client = requests.Session()
            self._local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
        except Exception:
            self._client = None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments for backend
            
        Returns:
            Generated response string
        """
        if self._client is None:
            return self._mock_generate(prompt)
        
        if self.backend in ["gpt-4", "gpt-3.5-turbo"]:
            return self._generate_openai(prompt, system_prompt, **kwargs)
        elif self.backend in ["claude", "claude-3"]:
            return self._generate_anthropic(prompt, system_prompt, **kwargs)
        else:
            return self._generate_local(prompt, system_prompt, **kwargs)
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        **kwargs
    ) -> str:
        """Generate using OpenAI API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self.backend,
            messages=messages,
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        **kwargs
    ) -> str:
        """Generate using Anthropic API."""
        message = self._client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            system=system_prompt or "You are a helpful robotics assistant.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    
    def _generate_local(
        self,
        prompt: str,
        system_prompt: Optional[str],
        **kwargs
    ) -> str:
        """Generate using local LLM."""
        import requests
        
        response = requests.post(
            f"{self._local_url}/api/generate",
            json={
                "model": self.backend,
                "prompt": f"{system_prompt or ''}\n\n{prompt}",
                "stream": False
            }
        )
        
        return response.json().get("response", "")
    
    def _mock_generate(self, prompt: str) -> str:
        """Mock generation for testing."""
        # Return sensible defaults for common prompts
        if "morphology" in prompt.lower() and "fragile" in prompt.lower():
            return json.dumps({
                "gripper_width": 0.4,
                "stiffness": 0.3,
                "compliance": 0.8
            })
        elif "phase" in prompt.lower() or "decompose" in prompt.lower():
            return json.dumps([
                {
                    "description": "Approach object",
                    "start": 0.0,
                    "end": 0.3,
                    "tags": ["reach"],
                    "morphology": {"gripper_width": 0.8, "stiffness": 0.5, "compliance": 0.3}
                },
                {
                    "description": "Grasp object",
                    "start": 0.3,
                    "end": 0.6,
                    "tags": ["grasp", "compliance"],
                    "morphology": {"gripper_width": 0.4, "stiffness": 0.3, "compliance": 0.7}
                },
                {
                    "description": "Lift and move",
                    "start": 0.6,
                    "end": 1.0,
                    "tags": ["force"],
                    "morphology": {"gripper_width": 0.4, "stiffness": 0.6, "compliance": 0.5}
                }
            ])
        elif "adapt" in prompt.lower():
            return "increase_compliance"
        else:
            return "no_change"
    
    def assess_adaptation_need(
        self,
        perception: Dict,
        state: Dict
    ) -> bool:
        """
        Assess if morphology adaptation is needed.
        
        Args:
            perception: Visual/tactile perception data
            state: Current robot state
            
        Returns:
            True if adaptation recommended
        """
        prompt = f"""
Based on the following perception and state, should the robot adapt its morphology?

Perception: {json.dumps(perception)}
State: {json.dumps(state)}

Answer only "yes" or "no".
"""
        response = self.generate(prompt)
        return "yes" in response.lower()
    
    def analyze_situation(
        self,
        perception: Dict,
        state: Dict,
        task: str
    ) -> str:
        """
        Analyze current situation and suggest adaptation.
        
        Args:
            perception: Perception data
            state: Robot state
            task: Current task
            
        Returns:
            Semantic context string for adaptation
        """
        prompt = f"""
Analyze this manipulation situation:

Task: {task}
Perception: {json.dumps(perception)}
Current state: {json.dumps(state)}

Describe any necessary morphology adaptations in one sentence.
"""
        return self.generate(prompt)
