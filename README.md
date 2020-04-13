# Spoken-Language-Identification-RNN

## Objective
Spoken  Language  Identification  (LID)  is  broadly  defined  as  recognizing  the  language  of  a  given speech utterance.  It has numerous applications in automated language and speech recognition,multilingual machine translations,  speech-to-speech translations,  and emergency call routing.  Inthis  homework,  we  will  try  to  classify  three  languages  (English,  Hindi  and  Mandarin)  from  the spoken utterances that have been crowd-sourced from the class.

## Method

```Python
mapping = {'english': 0, 'hindi': 1, ' mandarin': 2}
```

Extract MFCC features from audio files, build up Recurrent Neural Network (GRU/LSTM) to train the model to output 3-class probability.

### Duel with silence

Mark silence audios with label -1, and omit them both in loss and accuracy measurement.

