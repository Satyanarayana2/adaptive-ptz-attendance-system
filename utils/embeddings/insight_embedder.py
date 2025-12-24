import numpy as np
import cv2
from .base_embedder import BaseEmbedder
from insightface.model_zoo import model_zoo

class InsightEmbedder(BaseEmbedder):
    """
    ArcFace embedder using InsightFace ONNX model (buffalo_l).
    Uses get_feat([img]) => [embedding] API.
    Produces 512D L2-normalized embeddings.
    """

    def __init__(self, model_name="buffalo_l"):
        self.model_name = model_name
        self.embedder = None

    def prepare(self):
        """Load the ArcFace ONNX model."""
        self.embedder = model_zoo.get_model(self.model_name)
        self.embedder.prepare(ctx_id=-1)  # CPU

    def preprocess_face(self, aligned_face):
        """
        Convert BGR -> RGB and resize to 112×112.
        ArcFace requires exact 112×112 RGB input.
        """
        if aligned_face is None:
            return None

        # Convert BGR → RGB
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # Resize to 112×112
        rgb = cv2.resize(rgb, (112, 112))

        return rgb

    def get_embedding(self, aligned_face):
        """Returns a 512D L2-normalized ArcFace embedding."""
        if self.embedder is None:
            raise RuntimeError("Embedder not prepared. Call prepare().")

        # Preprocess
        face_rgb = self.preprocess_face(aligned_face)
        if face_rgb is None:
            return None

        # ---- THE CORRECT CALL FOR YOUR VERSION ----
        emb_list = self.embedder.get_feat([face_rgb])  # list output
        emb = emb_list[0].astype(np.float32)
        # -------------------------------------------

        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb
