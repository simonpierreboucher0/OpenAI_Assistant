"""
This module implements OpenAI language models and chatbot interfaces.
It provides functionality to interact with OpenAI's API for text generation and chat completion,
as well as conversation management with different display options.

Classes:
    OpenAI_LLM: A class that handles communication with OpenAI's API.
    OpenAI_Chatbot: A class that implements chatbot functionality using OpenAI's API.
"""

import json
import requests
import os
import sys
import uuid
from typing import Optional, Dict, List, Union, Literal
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAI_LLM:
    """
    A class to handle interactions with OpenAI's API.

    This class manages the configuration and execution of requests to OpenAI's API,
    providing a streamlined interface for text generation and chat completion tasks.

    Attributes:
        model (str): The OpenAI model to use (default: "gpt-3.5-turbo")
        temperature (float): Controls randomness in responses (0-1)
        max_tokens (int): Maximum number of tokens in the response
        stream (bool): Whether to stream the response
        top_p (float): Controls diversity via nucleus sampling
        frequency_penalty (Optional[float]): Adjusts frequency of token usage
        presence_penalty (Optional[float]): Adjusts presence of token usage
        api_key (str): OpenAI API key from environment variables

    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment variables
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = True,
        top_p: float = 0,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ):
        """
        Initialize the OpenAI LLM interface with specified parameters.

        Args:
            model (str): OpenAI model identifier
            temperature (float): Response randomness (0-1)
            max_tokens (int): Maximum response length
            stream (bool): Enable/disable response streaming
            top_p (float): Nucleus sampling parameter
            frequency_penalty (Optional[float]): Token frequency adjustment
            presence_penalty (Optional[float]): Token presence adjustment
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def _make_request(self, messages: List[Dict]) -> requests.Response:
        """
        Make a request to OpenAI's API with the specified messages.

        Args:
            messages (List[Dict]): List of message dictionaries containing role and content

        Returns:
            requests.Response: The API response object

        Note:
            The response format is set to "text" and includes all specified parameters
            in the request payload.
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "text"},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream
        }

        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty

        return requests.post(url, headers=headers, json=payload, stream=self.stream)


class OpenAI_Chatbot:
    """
    A chatbot implementation using OpenAI's language models.

    This class manages conversations, maintains chat history, and handles
    interactions with the OpenAI LLM. It includes features for conversation
    persistence and management.

    Attributes:
        _chatbot_counter (int): Class-level counter for unique chatbot IDs
        provider (str): The LLM provider name ("openai")
        llm (OpenAI_LLM): Instance of the OpenAI LLM interface
        system_prompt (str): Initial system prompt for the conversation
        verbose (bool): Enable/disable detailed output
        display_mode (str): Display mode for output ("full", "response_only", "none")
        name (str): Chatbot instance name
        conversation_folder (str): Path to conversation storage
        history (List[Dict]): Conversation history
    """

    _chatbot_counter = 0
    provider = "openai"

    def __init__(
        self,
        llm: OpenAI_LLM,
        system_prompt: str = "You are a helpful assistant.",
        verbose: bool = True,
        display_mode: Union[Literal["full"], Literal["response_only"], Literal["none"]] = "full",
        name: Optional[str] = None
    ):
        """
        Initialize the chatbot with specified parameters.

        Args:
            llm (OpenAI_LLM): The language model interface
            system_prompt (str): Initial system message
            verbose (bool): Enable/disable detailed output (deprecated, use display_mode instead)
            display_mode (str): Controls output display ("full", "response_only", "none")
            name (Optional[str]): Custom name for the chatbot
        """
        OpenAI_Chatbot._chatbot_counter += 1
        self.llm = llm
        self.system_prompt = system_prompt
        self.display_mode = display_mode
        # Keep verbose for backward compatibility, but prefer display_mode
        self.verbose = verbose if display_mode == "full" else False
        self.chatbot_id = OpenAI_Chatbot._chatbot_counter
        self.name = name or f"chatbot_{self.chatbot_id}"
        self.conversation_folder = self._create_conversation_folder()
        self.history: List[Dict] = []
        self._initialize_conversation()

    def _create_conversation_folder(self) -> str:
        """
        Create and set up the conversation storage structure.

        Returns:
            str: Path to the chatbot's conversation folder

        Note:
            Creates necessary directories and metadata files for
            both provider and chatbot-specific information.
        """
        base_folder = "conversations"
        provider_folder = f"{base_folder}/{self.provider}"
        chatbot_folder = f"{provider_folder}/{self.name}"
        
        os.makedirs(provider_folder, exist_ok=True)
        os.makedirs(chatbot_folder, exist_ok=True)
        
        provider_metadata = {
            "provider": self.provider,
            "total_chatbots": self._chatbot_counter,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(f"{provider_folder}/provider_metadata.json", 'w') as f:
            json.dump(provider_metadata, f, indent=2)
        
        chatbot_metadata = {
            "provider": self.provider,
            "chatbot_id": self.chatbot_id,
            "name": self.name,
            "created_at": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "model": self.llm.model,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "top_p": self.llm.top_p,
            "frequency_penalty": self.llm.frequency_penalty,
            "presence_penalty": self.llm.presence_penalty
        }
        
        with open(f"{chatbot_folder}/metadata.json", 'w') as f:
            json.dump(chatbot_metadata, f, indent=2)
            
        return chatbot_folder

    def _initialize_conversation(self):
        """
        Initialize a new conversation with the system prompt.
        
        Creates a new conversation ID and sets up initial history
        with the system prompt.
        """
        self.conversation_id = str(uuid.uuid4())
        self.history = [{
            "role": "system",
            "content": [{"type": "text", "text": self.system_prompt}]
        }]
        self._save_conversation()

    def _save_conversation(self):
        """
        Save the current conversation state to a JSON file.
        
        Stores complete conversation metadata and history in the
        chatbot's conversation folder.
        """
        filename = f"{self.conversation_folder}/conversation_{self.conversation_id}.json"
        
        conversation_data = {
            "provider": self.provider,
            "conversation_id": self.conversation_id,
            "chatbot_name": self.name,
            "chatbot_id": self.chatbot_id,
            "timestamp": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "model": self.llm.model,
            "history": self.history
        }
        
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)

    def start_new_conversation(self):
        """
        Start a new conversation while maintaining chatbot identity.
        
        Reinitializes the conversation with a new ID while keeping
        the chatbot's configuration.
        """
        self._initialize_conversation()
        if self.display_mode == "full":
            print(f"\nStarted new conversation with ID: {self.conversation_id}")

    def list_conversations(self) -> List[str]:
        """
        List all conversations for this chatbot.

        Returns:
            List[str]: List of conversation filenames
        """
        conversations = [f for f in os.listdir(self.conversation_folder) 
                        if f.startswith('conversation_') and f.endswith('.json')]
        return conversations

    def load_conversation(self, conversation_id: str):
        """
        Load a specific conversation by ID.

        Args:
            conversation_id (str): ID of the conversation to load

        Raises:
            FileNotFoundError: If the specified conversation doesn't exist
        """
        filename = f"{self.conversation_folder}/conversation_{conversation_id}.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.conversation_id = data["conversation_id"]
                self.history = data["history"]
                if self.display_mode == "full":
                    print(f"\nLoaded conversation: {conversation_id}")
        else:
            raise FileNotFoundError(f"Conversation {conversation_id} not found")

    def _prepare_messages(self, message: str) -> List[Dict]:
        """
        Prepare messages for API request.

        Args:
            message (str): User input message

        Returns:
            List[Dict]: Formatted message history for API request
        """
        # Transformer les messages pour le format d'API OpenAI
        formatted_messages = []
        for msg in self.history:
            role = msg["role"]
            # OpenAI ne supporte que le type 'text' pour le contenu, 
            # alors on extrait juste le texte du premier élément
            content = msg["content"][0]["text"] if isinstance(msg["content"], list) else msg["content"]
            formatted_messages.append({"role": role, "content": content})
        
        # Ajouter le nouveau message de l'utilisateur
        formatted_messages.append({"role": "user", "content": message})
        
        # Également mettre à jour notre historique interne
        self.history.append({
            "role": "user",
            "content": [{"type": "text", "text": message}]
        })
        
        return formatted_messages

    def _print_streaming_response(self, content: str):
        """
        Print streaming response according to display mode.

        Args:
            content (str): Content chunk to print
        """
        sys.stdout.write(content)
        sys.stdout.flush()

    def __call__(self, message: str) -> str:
        """
        Process user message and return response.

        Args:
            message (str): User input message

        Returns:
            str: Assistant's response
        """
        messages = self._prepare_messages(message)
        response = self.llm._make_request(messages)

        # Handle display of user message based on display_mode
        if self.display_mode == "full":
            print(f"\n{self.name} - User: ", message)
            print(f"\n{self.name} - Assistant: ", end="")

        full_response = ""
        
        if self.llm.stream:
            collected_messages = []
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        line = line[6:]
                        if line != "[DONE]":
                            try:
                                data = json.loads(line)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content = delta["content"]
                                        collected_messages.append(content)
                                        if self.display_mode in ["full", "response_only"]:
                                            self._print_streaming_response(content)
                            except json.JSONDecodeError:
                                continue

            full_response = "".join(collected_messages)
            # Ajouter juste un saut de ligne à la fin du streaming
            if self.display_mode in ["full", "response_only"]:
                print()
        else:
            # Cas non-streaming
            response_data = response.json()
            full_response = response_data["choices"][0]["message"]["content"]
            if self.display_mode in ["full", "response_only"]:
                print(full_response)

        # Mettre à jour l'historique avec la réponse complète
        self.history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": full_response}]
        })
        self._save_conversation()

        # Ne pas imprimer la réponse ici, elle a déjà été affichée si nécessaire
        return full_response


def ask(chatbot, message):
    """
    Fonction utilitaire pour poser une question à un chatbot sans afficher la valeur de retour.
    
    Cette fonction permet d'éviter l'affichage automatique du résultat retourné par le chatbot
    lorsqu'il est utilisé dans un environnement interactif comme un notebook ou un shell Python.
    
    Args:
        chatbot (OpenAI_Chatbot): L'instance du chatbot
        message (str): Le message à envoyer au chatbot
    
    Returns:
        None
    """
    _ = chatbot(message)  # Appelle le chatbot mais n'affiche pas la valeur retournée ¨

