from crewai import Task
from pydantic import BaseModel

from src.agents_src.agents.question_answer_agent import qa_agent


class AnswerStructure(BaseModel):  # 1 usage  new *
    answer: str
    sources: list[str]
    tool_used: str
    rationale: str

qa_task = Task(
    agent=qa_agent,
    name="Question Answering Task",
    description="""
    You are provided with the following conversation history:
    ---
    {chat_history}
    ---
    
    The user's latest query is: "{user_query}"
    
    Instructions:
    1. First, check the conversation history. If the query is vague, use the history to understand exactly what the user is referring to.
    2. Use the rag_query_tool to search the document store for relevant context based on the resolved meaning of the query.
    3. Prioritize evidence that directly addresses the query.
    4. Synthesize a clear, accurate answer grounded in the retrieved sources.
    5. If the query cannot be answered from the knowledge source, state clearly that the information is not available.
    """,
    expected_output="""
    A structured JSON object with the following fields:

    {
      "answer": "Direct response to the query (1-3 paragraphs, clear and accurate).
                  If no answer is found, return: 'The knowledge source does not contain the required information.'",
      "sources": ["List of document titles, sections, or citations used (empty list if none)"],
      "tool_used": "Name of the retrieval/analysis tool invoked (e.g., RAG Retriever, VectorDB, ChatHistory, etc.)",
      "rationale": "Brief explanation of why this answer was chosen, or why no relevant information was found"
    }
    """,
    output_pydantic=AnswerStructure,
)