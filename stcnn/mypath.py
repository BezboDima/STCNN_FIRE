class Path(object):
    """Central path configuration — edit these to match your local setup."""

    @staticmethod
    def db_root_dir():
        """Root of the DAVIS-2016 dataset.
        Expected layout:
            <root>/JPEGImages/480p/<sequence>/
            <root>/Annotations/480p/<sequence>/
        """
        return '/path/to/DAVIS'

    @staticmethod
    def save_root_dir():
        """Where checkpoints and TensorBoard logs are written."""
        return '/path/to/output'

    @staticmethod
    def models_dir():
        return "./models"

    @staticmethod
    def data_dir():
        return "./data"

    @staticmethod
    def VID_list_file():
        return "./data/VID_seqs_list.txt"

    @staticmethod
    def DAVIS_list_file():
        return "./data/DAVIS_seqs_list.txt"

    @staticmethod
    def MSRAdataset_dir():
        return '/path/to/MSRA10K'

    @staticmethod
    def VOC_dir():
        return '/path/to/VOC'