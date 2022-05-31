from typing import List, Optional

from pydantic import BaseModel
from pydantic.typing import Literal

from larch.context_creator import Directory


class GenerationRequest(BaseModel):
    files: Directory
    model: str
    prompt: str
    project_name: str


class Edit(BaseModel):
    type: Literal['insertion', 'deletion']
    start: int
    end: Optional[int]
    text: Optional[str]


class Generation(BaseModel):
    text: str
    edits: List[Edit]
    index: int
    logprobs: float


class Response(BaseModel):
    id: str
    model: str
    choices: List[Generation]


class GenerationModel(BaseModel):
    id: str
    description: str
    owned_by: str


class GenerationModels(BaseModel):
    data: List[GenerationModel]
