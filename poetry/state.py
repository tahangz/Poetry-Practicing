from enum import Enum

class AppState(Enum):
    WAITING_FOR_TITLE = "waiting_for_title"
    TITLE_FOUND = "title_found"
    RECITING_POEM = "reciting_poem"
    POEM_COMPLETE = "poem_complete"
