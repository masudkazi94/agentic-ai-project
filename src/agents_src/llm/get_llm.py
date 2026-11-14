from crewai import LLM

from src.agents_src.llm.llm_configuration import LLM_CONFIG


def get_llm_for_agent(agent_name):  # 2 usages  new *
    model = LLM_CONFIG.get(agent_name, {}).get("model", "groq/openai/gpt-oss-120b")
    temperature = LLM_CONFIG.get(agent_name, {}).get("temperature", 0.0)
    llm = LLM(
        model=model,
        temperature=temperature,
        stream=False
    )
    return llm