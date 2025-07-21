from langchain.schema import BaseOutputParser
import re

class HTMLStringOutputParser(BaseOutputParser):
    """
    LangChain output parser that extracts raw HTML from a markdown-style code block.
    """

    def parse(self, text: str) -> str:
        """
        Extracts and returns the HTML content between ```html and ```.

        Args:
            text (str): The full response from the LLM.

        Returns:
            str: Clean HTML string.
        """
        match = re.search(r"```html\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        raise ValueError("No HTML code block found")
