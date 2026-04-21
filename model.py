import ollama
from rag import SalesRAG
from prompts import SalesPrompts


class SalesLLM:
    MODEL = "llama3.2"

    def __init__(self):
        self.rag = SalesRAG()

    def generate_followup_message(self, enquiry_data: dict):
        similar_docs = self.rag.search_similar(str(enquiry_data))
        prompt = SalesPrompts.followup_message(enquiry_data, similar_docs)
        response = ollama.chat(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    def analyze_lead(self, enquiry_data: dict):
        similar_docs = self.rag.search_similar(str(enquiry_data))
        prompt = SalesPrompts.analyze_lead(enquiry_data, similar_docs)
        response = ollama.chat(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    def summarize_feedback(self, feedback_text: str):
        similar_docs = self.rag.search_similar(feedback_text)
        prompt = SalesPrompts.summarize_feedback(feedback_text, similar_docs)
        response = ollama.chat(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    def chat_with_data(self, user_query: str):
        """
        Smart routing:
          1. Try structured pandas query (exact counts, date filters, vehicle stats)
          2. Fall back to vector similarity search if no structured match
        Then stream the LLM response live.
        """
        # Step 1: Try structured/analytical path first
        structured_context = self.rag.get_structured_context(user_query)

        if structured_context:
            # We have exact data — pass it directly as context
            context = structured_context
        else:
            # Fall back to semantic similarity search
            similar_docs = self.rag.search_similar(user_query, n_results=6)
            if similar_docs:
                context = "\n".join(similar_docs)
            else:
                context = "No relevant records found in the CRM database."

        prompt = SalesPrompts.general_chat(user_query, context)

        response = ollama.chat(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in response:
            print(chunk["message"]["content"], end="", flush=True)
        print()