from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from models.financial_models import NotesWrapper
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
model = ChatOpenAI(model="gpt-4o", temperature=0.4)

def recreate_html_table(extracted_text):
    prompt_template = ChatPromptTemplate.from_template(
    """
    You are given text extracted from a PDF file, which originally contained a table. 
    Reconstruct it into clean, semantic HTML that represents the original layout as closely as possible.

    Follow these rules:

    1. **Preserve narrative text** before or after the table as `<p>` tags.  
    2. **Build table structure** with `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, and `<td>`.  
    3. **Headers**:
    - If there are multi-level headers, use multiple `<tr>` rows in `<thead>` with proper `colspan`/`rowspan`.  
    - Ensure `colspan` matches the number of underlying columns.  
    - Use `<br>` inside `<th>` for long headers if needed.
    4. **Row hierarchy**:
    - If there are grouped rows (e.g., Employees → Permanent → Male/Female), show hierarchy by prefixing sub-level rows with `&nbsp;&nbsp;` or making parent rows bold.  
    - Include “Total” rows where present and make them bold.
    5. **Grouped parameters**:
    - If rows have sub-rows (e.g., “No treatment” vs “With treatment”), expand them as separate rows with full context in each cell (Destination + Treatment).  
    6. **Values**:
    - Retain values exactly as extracted (e.g., `NIL`, `NA`, `0.00`).  
    - Leave empty `<td>` for missing data.  
    7. **Footnotes**:
    - Extract footnotes or annotations (`*`, `^`) and place them as `<p><em>…</em></p>` after the table.
    8. **Totals**:
    - Clearly highlight total rows with `<strong>` (or place them in a `<tfoot>` if appropriate).
    9. **Do not infer or invent data**. Only structure what is present.
    10. **Hierarchy Detection:** 
        - Detect hierarchical relationships from the text (e.g., categories with subcategories or numbered sections like (i), (ii)). 
        - Render parent categories in bold, and indent subcategories using `&nbsp;&nbsp;`. 
        - Do NOT assume or add groupings that are not explicitly present in the text.

    Here is the extracted text:
    {extracted_text}
    """
    )

    chain = prompt_template | model
    result = chain.invoke({
        "extracted_text": extracted_text,
    })

    match = re.search(r"```html\s*(.*?)\s*```", result.text(), re.DOTALL)
    if match:
        return match.group(1).strip()

    return result.text()


def fill_observations(notes_wrapper):
    updated_notes = {}

    for note_no, note in notes_wrapper.notes.items():
        print("reading note no:", note_no)
        try:
            if note.observation:
                # Skip if observation already present
                updated_notes[note_no] = note
                continue

            prompt_template = ChatPromptTemplate.from_template(
                """
            You are a financial analyst reviewing detailed notes from the financial statements of a company.

            Your tasks are:
            - Carefully read the provided note description.
            - Derive meaningful financial insights, risks, opportunities, or noteworthy points from the description.
            - The observation should be concise, professional, and reflect expertise in financial analysis.
            - DO NOT modify the original description text.
            - DO NOT repeat the description.
            - Focus on adding additional analysis, implications, or important highlights.

            Return the output in the following format:

            {format_instructions}

            Description:
            {description}
            """
            )

            parser = PydanticOutputParser(pydantic_object=NotesWrapper)

            chain = prompt_template | model | parser
            result = chain.invoke(
                {
                    "description": note,
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            updated_note = {
                "note_no": note.note_no,
                "description": note.description,  # Keep original description
                "observation": result.notes[note_no].observation,
                # "html_str": note.html_str,
            }
            updated_notes[note_no] = updated_note
        except Exception as e:
            print(f"Error processing note {note_no}: {e}")

    return updated_notes
