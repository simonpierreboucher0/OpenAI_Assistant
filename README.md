# 🤖 OpenAI Assistant

![OpenAI Logo](https://img.shields.io/badge/Powered%20by-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/simonpierreboucher0/OpenAI_Assistant?style=for-the-badge)](https://github.com/simonpierreboucher0/OpenAI_Assistant/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/simonpierreboucher0/OpenAI_Assistant?style=for-the-badge)](https://github.com/simonpierreboucher0/OpenAI_Assistant/stargazers)

## 📋 Vue d'ensemble

Ce module implémente des interfaces pour les modèles de langage OpenAI et des chatbots interactifs. Il fournit des fonctionnalités pour interagir avec l'API d'OpenAI pour la génération de texte et la complétion de chat, ainsi que la gestion des conversations avec différentes options d'affichage.

## ✨ Fonctionnalités

- 🔄 Communication fluide avec l'API OpenAI
- 💬 Gestion complète des conversations
- 📝 Sauvegarde automatique des historiques de chat
- 📊 Différents modes d'affichage pour les réponses
- 🌊 Support du streaming pour les réponses en temps réel
- 🔄 Possibilité de charger des conversations précédentes

## 🛠️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/simonpierreboucher0/OpenAI_Assistant.git
cd OpenAI_Assistant

# Installer les dépendances
pip install -r requirements.txt

# Configurer votre clé API OpenAI
echo "OPENAI_API_KEY=votre-clé-api" > .env
```

## 🚀 Utilisation rapide

```python
from openai_assistant import OpenAI_LLM, OpenAI_Chatbot

# Créer une instance du modèle de langage
llm = OpenAI_LLM(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=2000
)

# Initialiser un chatbot
chatbot = OpenAI_Chatbot(
    llm=llm,
    system_prompt="Vous êtes un assistant IA utile et amical.",
    display_mode="full"
)

# Poser des questions au chatbot
response = chatbot("Qu'est-ce que l'intelligence artificielle?")

# Ou utiliser la fonction ask pour éviter l'affichage du retour dans les notebooks
from openai_assistant import ask
ask(chatbot, "Explique-moi le concept de machine learning.")

# Démarrer une nouvelle conversation
chatbot.start_new_conversation()
```

## 📚 Classes principales

### 🧠 `OpenAI_LLM`

Gère la communication avec l'API OpenAI, configurant les paramètres de requête et gérant les réponses.

```python
llm = OpenAI_LLM(
    model="gpt-4",                # Modèle à utiliser
    temperature=0.5,              # Contrôle la créativité (0-1)
    max_tokens=1000,              # Limite de longueur de réponse
    stream=True,                  # Streaming des réponses
    top_p=0.9,                    # Contrôle la diversité
    frequency_penalty=0.5,        # Pénalité de fréquence 
    presence_penalty=0.2          # Pénalité de présence
)
```

### 💬 `OpenAI_Chatbot`

Implémente l'interface de chatbot avec gestion d'historique et persistance des conversations.

```python
chatbot = OpenAI_Chatbot(
    llm=llm,                       # Instance d'OpenAI_LLM
    system_prompt="Instructions",  # Prompt système initial
    display_mode="full",           # Mode d'affichage: "full", "response_only", ou "none"
    name="my_assistant"            # Nom personnalisé (optionnel)
)
```

## 🗂️ Gestion des conversations

```python
# Lister les conversations disponibles
conversations = chatbot.list_conversations()
print(conversations)

# Charger une conversation existante
chatbot.load_conversation("conversation_id")

# Commencer une nouvelle conversation
chatbot.start_new_conversation()
```

## 🎛️ Modes d'affichage

- `"full"` : Affiche le message de l'utilisateur et la réponse de l'assistant
- `"response_only"` : Affiche uniquement la réponse de l'assistant
- `"none"` : N'affiche rien automatiquement (utile pour les applications)

## 📄 Structure des conversations

Les conversations sont sauvegardées dans le dossier `conversations/openai/{nom_du_chatbot}/` avec les métadonnées associées pour faciliter la gestion et le suivi.

## ⚠️ Prérequis

- Python 3.6 ou supérieur
- Une clé API OpenAI valide

## 📝 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](https://github.com/simonpierreboucher0/OpenAI_Assistant/blob/main/LICENSE) pour plus de détails.
