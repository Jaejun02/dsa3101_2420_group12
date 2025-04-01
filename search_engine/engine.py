from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
import faiss
from rank_bm25 import BM25Okapi
import numpy as np


class ESGSearchEngine:
    """
    A class that can perform both semantic search and keyword search for related contents based on the user's query.
    
    Attributes:
        name (str): The name of the employee.
        age (int): The age of the employee.
        department (str): The department the employee works in.
        salary (float): The salary of the employee.
    """
    
    def __init__(self, model_path, tokenizer_model_path, use_gpu=False):
        """
        The constructor for ESGSearchEngine class.
        
        Parameters:
            model_path (str): The path to the SentenceTransformer model.
            tokenizer_model_path (str): Model path for tokenizer.
            use_gpu (bool): A flag to indicate whether to use GPU for encoding and indexing.
        """
        self.model = SentenceTransformer(model_path, model_kwargs={"torch_dtype": "float16"})
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        self.use_gpu = use_gpu
        self.chunks = []
        self.index = None
        self.bm25 = None
        
    
    def initialize(self, text, chunk_size=256, chunk_overlap=128):
        """
        Initialize the Faiss index and BM25 keyword search for the given text.
        
        Parameters:
            text (str): A text to be indexed.
            chunk_size (int): The size of the text chunk.
            chunk_overlap (int): The overlap between chunks.
        """
        
        # split text into chunks
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        self.chunks = chunks
        
        # initialize faiss index
        embeddings = self.model.encode(chunks, device="cuda:0", convert_to_numpy=True, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
        self.index.add(embeddings)
        
        # initialize bm25 keyword search
        tokenized_chunks = [word_tokenize(text.lower()) for text in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
            
    def semantic_search(self, query, k):
        """
        Perform semantic search for the given query.
        
        Parameters:
            query (str): The query to search for.
            k (int): The number of results to return.
            
        Returns:
            List[str]: A list of chunks that match the query.
            Dict[int, float]: A dictionary of chunk indices and their scores.
        """
        
        if self.index is None:
            raise ValueError("Index is not initialized. Please call initialize() method first.")
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, k)
        scores = {i: D[0][idx] for idx, i in enumerate(I[0])}
        chunks = [self.chunks[i] for i in scores.keys()]
        
        return chunks, scores
    
    def keyword_search(self, keyword_query_list, k):
        """
        Perform keyword search for the given query.
        
        Parameters:
            keyword_query_list (List[str]): The list of keywords to search for.
            k (int): The number of results to return.
            
        Returns:
            List[str]: A list of chunks that match the query.
            Dict[int, float]: A dictionary of chunk indices and their scores.
        """
        
        if self.bm25 is None:
            raise ValueError("BM25 is not initialized. Please call initialize() method first.")
        
        scores = self.bm25.get_scores(keyword_query_list)
        top_indices = np.argsort(scores)[::-1][:k]
        chunks = [self.chunks[i] for i in top_indices]
        
        return chunks, {i: scores[i] for i in top_indices}
    
    def combined_search(self, query, keyword_query_list, rerank_k=200, top_k=50, alpha=0.7):
        """
        Perform combined semantic and keyword search for the given query.
        
        Parameters:
            query (str): The query to search for.
            rerank_k (int): The number of results to rerank.
            top_k (int): The number of results to return.
            alpha (float): The weight for semantic search.
            
        Returns:
            List[str]: A list of chunks that match the query.
            Dict[int, float]: A dictionary of chunk indices and their scores.
        """
        
        if self.index is None or self.bm25 is None:
            raise ValueError("Index or BM25 is not initialized. Please call initialize() method first.")
        
        # semantic search
        _, semantic_scores = self.semantic_search(query, rerank_k)
        
        # keyword search
        _, keyword_scores = self.keyword_search(keyword_query_list, rerank_k)
        
        # combine and rerank
        combined_scores = {}
        for i, score in semantic_scores.items():
            combined_scores[i] = alpha * score

        for i, score in keyword_scores.items():
            if i in combined_scores:
                combined_scores[i] += (1 - alpha) * score
            else:
                combined_scores[i] = (1 - alpha) * score
                
        combined_scores_tuple = list(combined_scores.items())
        combined_scores_tuple.sort(key=lambda x: x[1], reverse=True)
        top_k_c = combined_scores_tuple[:top_k]
        chunks = [self.chunks[i] for i, _ in top_k_c]
        
        return chunks, {i: combined_scores[i] for i, _ in top_k_c}
        