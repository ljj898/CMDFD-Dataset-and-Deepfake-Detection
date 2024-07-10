# CMDFD-Dataset-and-Deepfake-Detection

This repository contains the code and dataset associated with the ICME paper titled "Explicit Correlation Learning for Generalizable Cross-Modal Deepfake Detection".

## Contents
1. **Dependency** create a conda environment
   ```bash
   conda create -n ExpCorre python=3.8
   conda activate ExpCorre
   pip install -r requirement.txt
   ```
3. **Code**
  
   - **Evaluation**: Run `python test.py` to perform the evaluation.
   - To download our weights mentioned in the paper, trained on FakeAVCeleb, you can download the weights from [this link](https://drive.google.com/drive/folders/11K22EGH-I_vy6vMohQ7g3NjNAfr8pFZ4).
   - In the `CSVfile` directory, we provide our train/test file organization format. Remember to modify the paths in these files to match your local paths for the FakeAVCeleb or CMDFD dataset.
   - The current code provides an option to test on CMDFD. You can choose the forgery type you want to test. If you want to perform an intra-dataset test, change `--testData` to `"FAV"`.

4. **CMDFD Dataset**
   - The proposed Cross-Modal Deepfake Dataset (CMDFD) is available for download via [this link](https://drive.google.com/drive/folders/198w2kdRmf64lrELJ2H1df66PFJmb64DG).


## Citation

Please cite the following if our paper or code is helpful to your research:

```bibtex
@article{yu2024explicit,
  title={Explicit Correlation Learning for Generalizable Cross-Modal Deepfake Detection},
  author={Yu, Cai and Jia, Shan and Fu, Xiaomeng and Liu, Jin and Tian, Jiahe and Dai, Jiao and Wang, Xi and Lyu, Siwei and Han, Jizhong},
  journal={arXiv preprint arXiv:2404.19171},
  year={2024}
}
```

## Acknowledgments

We studied many useful projects during our coding process, which include:

- The structure of the audio/visual encoder and decoder is learned from [this repo](https://github.com/TaoRuijie/TalkNet-ASD).
- The teacher model of ASR and VSR is learned from [this repo](https://github.com/smeetrs/deep_avsr/tree/master).
