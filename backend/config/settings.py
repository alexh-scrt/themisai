# config/settings.py
class Settings(BaseSettings):
    # Hot-reloadable settings
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "mxbai-embed-large"
    
    class Config:
        env_file = ".env"
        json_file = "config/runtime_config.json"