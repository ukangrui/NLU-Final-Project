import asyncio
from asyncio import Semaphore
import torch
import pandas as pd
import json
from tqdm import tqdm
import argparse

from utils import LlmModel, EmbeddingModel
from prompts import sys_prompt, usr_prompt, ModalityResponse

Modality2Id = {
    'Narrative': 0,
    'Dialogue': 1,
    'Visual': 2,
    'Set': 3,
    'Audio': 4,
    'Pace': 5,
    'Direction': 6,
    'Acting': 7,
    'Poster': 8,
}

Emb_dim = {
    'text-embedding-ada-002': 1536,
    'bert-base-uncased':     768,
    'all-MiniLM-L6-v2':      384,
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',         type=str,  default='ml-1m')
parser.add_argument('--use_rag',         type=bool, default=True)
parser.add_argument('--llm_model',       type=str,  default='gpt-4o-mini',
                    choices=['gpt-4o-mini'])
parser.add_argument('--embedding_model', type=str,  default='text-embedding-ada-002',
                    choices=['text-embedding-ada-002',
                             'bert-base-uncased',
                             'all-MiniLM-L6-v2'])
args = parser.parse_args()

MovieId2Index = {
    v:k for k,v in json.load(open(f'data/{args.dataset}/index2id.json')).items()
}
movies_dataset = pd.read_csv(
    f'data/{args.dataset}/movies.dat',
    sep='::', header=None,
    names=['movie_id','title','genres'],
    engine='python', encoding='latin1'
)
movies_dataset = movies_dataset[
    movies_dataset['movie_id'].isin(MovieId2Index.keys())
]
print("Encoding", len(movies_dataset), "movies in dataset")

rag_dataset = pd.read_csv(f'meta/{args.dataset}/rag_dataset.csv')
rag_dataset = rag_dataset.drop_duplicates(subset=['MovieID'], keep='first')

llm_model  = LlmModel(args.llm_model, sys_prompt=sys_prompt())
emb_model  = EmbeddingModel(args.embedding_model)


def build_emb(movie_title: str, movie_desc: str, use_rag: bool):
    llm_response  = llm_model(
        usr_prompt(movie_title, movie_desc, use_rag),
        ModalityResponse
    )[0]
    modality_embs = torch.stack(
        emb_model([llm_response[m] for m in Modality2Id.keys()]),
        dim=0
    )
    return modality_embs


async def worker(idx: int,
                 movie_title: str,
                 movie_desc: str,
                 use_rag: bool,
                 sem: Semaphore):
    """Acquire semaphore, run build_emb in a thread, then release."""
    async with sem:
        emb = await asyncio.to_thread(
            build_emb, movie_title, movie_desc, use_rag
        )
        return idx, emb


async def main():
    all_embs = {}
    sem = Semaphore(50)  # max concurrency

    # schedule one worker per movie
    tasks = []
    for _, row in movies_dataset.iterrows():
        movie_id    = row['movie_id']
        idx         = int(MovieId2Index[movie_id])
        title       = row['title']
        desc        = rag_dataset[
                          rag_dataset['MovieID']==movie_id
                      ]['overview'].item()
        tasks.append(
            asyncio.create_task(
                worker(idx, title, desc, args.use_rag, sem)
            )
        )

    # gather results as they complete, with a tqdm progress bar
    for fut in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Building embeddings"
    ):
        idx, emb = await fut
        all_embs[idx] = emb

    # pack into one big tensor
    num_movies = len(movies_dataset)
    emb_dim    = Emb_dim[args.embedding_model]
    embeddings = torch.zeros(
        num_movies+1, 9, emb_dim,
    )
    for idx, emb in all_embs.items():
        embeddings[idx] = emb

    torch.save(
        embeddings,
        f'meta/{args.dataset}/modality_embeddings_llm={args.llm_model}_emb={args.embedding_model}_rag={args.use_rag}.pt'
    )
    print(f"Embeddings saved to meta/{args.dataset}/modality_embeddings_llm={args.llm_model}_emb={args.embedding_model}_rag={args.use_rag}.pt")


if __name__ == "__main__":
    asyncio.run(main())
