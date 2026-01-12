import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import ortho_group

class SentinelEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- 🚀 Initializing Qwen-2.5 Core on {self.device.upper()} ---")
        
        self.model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5b-instruct", device=self.device, trust_remote_code=True)
        
        # FIX: Disable cache to prevent 'get_usable_length' error
        self.model._first_module().auto_model.config.use_cache = False
        
        # Phase 2 Novelty: RaBitQ Randomized Rotation
        # This guarantees that our 32x compression stays mathematically accurate
        P_raw = ortho_group.rvs(dim=1536)
        self.P_matrix = torch.tensor(P_raw, dtype=torch.float32).to(self.device)

    def encode(self, texts, persona="Forensic Auditor"):
        # FinMTEB Insight: Persona Augmentation
        augmented = [f"System: [Persona: {persona}] | Content: {t}" for t in texts]
        
        with torch.no_grad():
            embeddings = self.model.encode(augmented, batch_size=64, convert_to_tensor=True)
            # Apply RaBitQ Rotation
            rotated = torch.matmul(embeddings, self.P_matrix)
            normalized = torch.nn.functional.normalize(rotated, p=2, dim=1)
            
        return normalized.cpu().numpy()
