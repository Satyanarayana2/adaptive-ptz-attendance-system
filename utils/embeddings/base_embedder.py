# utils/embeddings/base_embedder.py

class BaseEmbedder:
    """
    Abstract interface for all face embedding models.
    ArcFace, AdaFace, ElasticFace, etc. will inherit this.
    """

    def prepare(self):
        """
        Load model weights, initialize runtime, set device.
        Called exactly once.
        """
        raise NotImplementedError

    def get_embedding(self, aligned_face):
        """
        aligned_face: Expected BGR cropped aligned face (112x112)
        Returns: 1D L2-normalized embedding vector or None
        """
        raise NotImplementedError
