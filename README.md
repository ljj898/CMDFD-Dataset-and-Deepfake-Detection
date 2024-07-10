# CMDFD-Dataset-and-Deepfake-Detection

This repository contains the code and dataset associated with the ICME paper titled "Explicit Correlation Learning for Generalizable Cross-Modal Deepfake Detection".

## Contents

1. **Code**
   - The source code used for the experiments described in the paper.
   - **Evaluation**: Run `python test.py` to perform the evaluation.
   - To download our weights mentioned in the paper, trained on FakeAVCeleb. You can download the weights from [this link](https://drive.google.com/drive/folders/11K22EGH-I_vy6vMohQ7g3NjNAfr8pFZ4).
   - In the `CSVfile` directory, we provide our train/test file organization format. Remember to modify the paths in these files to match your local paths for the FakeAVCeleb or CMDFD dataset.
   - The current code provides an option to test on CMDFD. You can choose the forgery type you want to test. If you want to perform an intra-dataset test, change `--testData` to `"FAV"`.

2. **CMDFD Dataset**
   - The proposed Cross-Modal Deepfake Dataset (CMDFD) is available for download via [this link](https://drive.google.com/drive/folders/198w2kdRmf64lrELJ2H1df66PFJmb64DG).

Please ensure you follow the dataset's license agreement and citation requirements when using it for your research.
