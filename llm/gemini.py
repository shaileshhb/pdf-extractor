from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import re

from models.financial_models import NotesWrapper

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


def recreate_html_table(extracted_text):
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a reliable assistant that reconstructs HTML tables from raw OCR-extracted text.

        The input is text extracted from a scanned table in a company’s annual report.
        The formatting has been lost, but the structure should be reconstructed accurately.

        Your task are:
        - Carefully read the provided note description.
        - Parse the input text and convert it into structured HTML using:
        - <h1>, <h2>, etc. for **section titles or headings**
        - <table>, <thead>, <tbody>, <tr>, <th>, and <td> for **tabular data**
        - Detect and preserve any **non-tabular headings** or **titles** by wrapping them in heading tags (<h4> for main titles, <h5> for subtitles, etc.) based on their appearance or position.
        - Ensure **all rows and columns** from the original text are included in the output. **Do not skip or truncate** any rows.
        - **Do not add, modify, or guess any data** — use only the text given.
        - Convert the **actual table** into valid HTML:
        - Use the first row of the table as the header and place it inside <thead>.
        - Place all remaining rows inside <tbody>.
        - Maintain the **original order of rows and columns**.
        - Ensure the **umber of columns remains consistent** across rows. Add empty <td></td> cells if needed.
        - Use **only** the content provided. Do not **fabricate, infer, or alter** any data.
        - Output a **single, complete block of valid HTML** that includes both headings and the table.
        - Do not include any additional explanation or comments — output only the HTML.
        - Do not explain anything. Return **only the HTML table**, with no extra commentary.
        - Return only the complete HTML table. Do **not** include any explanations, summaries, or additional text.

        Here is the extracted text:
        ---
        {extracted_text}
        ---
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
