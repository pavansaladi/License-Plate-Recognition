# License-Plate-Recognition

CRNN is a network that combines CNN and RNN to process images containing sequence information such as letters.

It is mainly used for OCR technology and has the following advantages.

End-to-end learning is possible.
Sequence data of arbitrary length can be processed because of LSTM which is free in size of input and output sequence.
There is no need for a detector or cropping technique to find each character one by one.
You can use CRNN for OCR, license plate recognition, text recognition, and so on. It depends on what data you are training

Convolutional Layer
Extracts features through CNN Layer (VGGNet, ResNet ...).

Recurrent Layer
Splits the features into a certain size and inserts them into the input of the Bidirectional LSTM or GRU.

Transcription Layer
Conversion of Feature-specific predictions to Label using CTC (Connectionist Temporal Classification).
