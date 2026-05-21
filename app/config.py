from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="AIME Bridge Backend", validation_alias=AliasChoices("APP_NAME"))
    app_env: str = Field(default="dev", validation_alias=AliasChoices("APP_ENV"))
    frontend_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        validation_alias=AliasChoices("FRONTEND_ORIGINS"),
    )

    llm_provider: str = Field(default="deepseek", validation_alias=AliasChoices("LLM_PROVIDER"))
    langchain_enabled: bool = Field(default=False, validation_alias=AliasChoices("LANGCHAIN_ENABLED"))
    llm_api_key: str = Field(default="", validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"))
    llm_model: str = Field(default="deepseek-v4-flash", validation_alias=AliasChoices("LLM_MODEL", "OPENAI_MODEL"))
    llm_base_url: str = Field(default="https://api.deepseek.com", validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"))

    deepseek_api_key: str = Field(default="", validation_alias=AliasChoices("DEEPSEEK_API_KEY"))
    deepseek_model: str = Field(default="deepseek-v4-flash", validation_alias=AliasChoices("DEEPSEEK_MODEL"))
    deepseek_base_url: str = Field(default="https://api.deepseek.com", validation_alias=AliasChoices("DEEPSEEK_BASE_URL"))
    hunyuan_api_key: str = Field(default="", validation_alias=AliasChoices("HUNYUAN_API_KEY"))
    hunyuan_model: str = Field(default="", validation_alias=AliasChoices("HUNYUAN_MODEL"))
    hunyuan_base_url: str = Field(default="", validation_alias=AliasChoices("HUNYUAN_BASE_URL"))
    xfyun_app_id: str = Field(default="", validation_alias=AliasChoices("XFYUN_APP_ID"))
    xfyun_api_key: str = Field(default="", validation_alias=AliasChoices("XFYUN_API_KEY"))
    xfyun_api_secret: str = Field(default="", validation_alias=AliasChoices("XFYUN_API_SECRET"))

    chat_store_path: str = Field(default="app/data/chat_messages.json", validation_alias=AliasChoices("CHAT_STORE_PATH"))
    upload_dir: str = Field(default="app/data/uploads", validation_alias=AliasChoices("UPLOAD_DIR"))
    public_base_url: str = Field(
        default="https://api.hgshouse.com/aimebridge",
        validation_alias=AliasChoices("PUBLIC_BASE_URL", "AIME_PUBLIC_BASE_URL"),
    )
    chat_max_history_messages: int = Field(
        default=0,
        ge=0,
        le=1000000,
        validation_alias=AliasChoices("CHAT_MAX_HISTORY_MESSAGES"),
    )
    ai_context_messages: int = Field(
        default=80,
        ge=12,
        le=200,
        validation_alias=AliasChoices("AI_CONTEXT_MESSAGES"),
    )

    model_config = SettingsConfigDict(env_file=(".env", ".env.runtime"), env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.frontend_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
