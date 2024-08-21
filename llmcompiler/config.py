from environs import Env

env = Env()
env.read_env(override=True)  # read .env file, if it exists


class Config:
    # LLM configurations
    LLM_MODEL_NAME = env.str("LLM_MODEL_NAME", "deepseek-chat")
    OPENAI_API_BASE = env.str("OPENAI_API_BASE", "https://api.deepseek.com")
    OPENAI_API_KEY = env.str("OPENAI_API_KEY")


config = Config()
