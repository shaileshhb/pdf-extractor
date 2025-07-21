from typing import Dict, Optional
from pydantic import BaseModel

class Note(BaseModel):
    note_no: str = None
    description: str  # Include full note text: title + content
    observation: Optional[str] = None
    # html_str: Optional[str]

class NotesWrapper(BaseModel):
    notes: Dict[str, Note]
