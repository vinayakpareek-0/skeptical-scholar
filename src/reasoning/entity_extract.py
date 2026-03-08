from gliner import GLiNER
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retrieval.run_rag import run_rag

def load_extractor():
    return GLiNER.from_pretrained("urchade/gliner_medium-v2.1")


def extract_entities(model, text:str)->list[dict]:
    labels = ["model", "dataset","metric","task","method"]
    entities = model.predict_entities(text,labels , threshold=0.3)
    return entities
    

def extract_from_chunks(model , chunks:list[dict])-> list[dict]:
    for chunk in chunks:
        chunk["entities"]= extract_entities(model , chunk["text"][:512]) 
    return chunks

if __name__=="__main__":
    results =run_rag("vision transformers robustness")
    if not results:
        print("No results to extract entities")
        sys.exit()
    model=load_extractor()
    chunks = extract_from_chunks(model , results)
    for chunk in chunks:
        print(chunk["text"][:200])
        print(chunk["entities"])
        print("\t")