from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import json
import os
from pathlib import Path

class FileSearcher:
    def __init__(self):
        # Load the sentence transformer model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 512
        self.index_path = "file_indexes"
        os.makedirs(self.index_path, exist_ok=True)
        
    def process_file(self, file_path: str) -> None:
        """Process and index a file's content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into chunks
            chunks = self._split_into_chunks(content)
            
            # Generate embeddings for chunks
            embeddings = self.model.encode(chunks)
            
            # Save index
            index_data = {
                'file_path': file_path,
                'chunks': chunks,
                'embeddings': embeddings.tolist()
            }
            
            index_file = Path(self.index_path) / f"{Path(file_path).name}.json"
            with open(index_file, 'w') as f:
                json.dump(index_data, f)
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant content across indexed files"""
        query_embedding = self.model.encode(query)
        results = []
        
        # Search through all indexed files
        for index_file in Path(self.index_path).glob("*.json"):
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            embeddings = np.array(index_data['embeddings'])
            similarities = np.dot(embeddings, query_embedding)
            
            # Get top matches from this file
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Similarity threshold
                    results.append({
                        'file_path': index_data['file_path'],
                        'content': index_data['chunks'][idx],
                        'similarity': float(similarities[idx])
                    })
        
        # Sort all results by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size // 2):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks 