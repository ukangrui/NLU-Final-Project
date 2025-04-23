def sys_prompt():
    return """
    You are an expert at generating description for each aspect of a movie.
    Example output ( Toy Story (1995) ) :
    Narrative: Heartfelt toy adventure centered on friendship and growth.
    Dialogue: Clever, playful dialogue that resonates with both kids and adults.
    Visual: Vibrant, imaginative animation delivering expressive, dynamic camera work with finesse.
    Set: Inventive, richly detailed toy-world environments and creative set design.
    Audio: Upbeat musical score and whimsical sound effects heighten emotions
    Pace: Smooth transitions and energetic pacing seamlessly drive the narrative.
    Direction: Innovative Pixar direction that unites creative storytelling with heart.
    Acting: Expressive voice acting that imbues each character with charm.
    Poster: Iconic posters and trailers capture the film’s playful essence.
    """
def usr_prompt(movie_title, movie_desc, use_rag):
    if use_rag:
        return f"""
        Please generate a description for each aspect of the movie '{movie_title}', you may use the following information about the movie:  {movie_desc}.
    """
    else:
        return f"""
        Please generate a description for each aspect of the movie '{movie_title}'.
    """


from pydantic import *
class ModalityResponse(BaseModel):
    Narrative: str
    Dialogue: str
    Visual: str
    Set: str
    Audio: str
    Pace: str
    Direction: str
    Acting: str
    Poster: str
