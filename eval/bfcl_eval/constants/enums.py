from enum import Enum

class ModelStyle(Enum):
    GORILLA = "gorilla"
    OPENAI_COMPLETIONS = "openai-completions"
    OPENAI_RESPONSES = "openai-responses"
    ANTHROPIC = "claude"
    MISTRAL = "mistral"
    GOOGLE = "google"
    OSSMODEL = "ossmodel"

class Language(Enum):
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"

class ReturnFormat(Enum):
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    JSON = "json"
