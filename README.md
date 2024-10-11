# VietnameseTextToxicClassify
I trained this model using PyTorch to detect toxic comments for Projectube.

I used VNCoreNLP to preprocess the raw Vietnamese sentence data and PhoBERT to train the model for text classification. I applied these technologies from https://github.com/VinAIResearch/PhoBERT.

## Use my code

### 1. Git clone my repository:
```
git clone https://github.com/hoangcaobao/Vietnamese_Text_Toxic_Classify.git
```

### 2. Change the directory to my folder and install VNCoreNLP:
```
cd VietnameseTextToxicClassify
pip3 install vncorenlp
mkdir -p vncorenlp/models/wordsegmenter
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
mv VnCoreNLP-1.1.1.jar vncorenlp/ 
mv vi-vocab vncorenlp/models/wordsegmenter/
mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
```
### 3. Add more data in 2 JSON files

### 4. Run training file:
```
python3 training.py
```
---
### Bao Hoang
