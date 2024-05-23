# S8P_Image_Recognition

Create conda env
```
conda env create -f envs/image-recognition-project.yml
```
Remove if needed
```
conda env remove --name image-recognition-project
```
Activate env
```
source activate image-recognition-project
```
or
```
conda activate image-recognition-project
```

# S8P_Voice_Modulation

https://keithito.com/LJ-Speech-Dataset

https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2

| Tool | Usecase |
| --- | --- |
| `Tacotron2` | Learn the model, based on person's voice. The model creates text read by specific person's voice |
| `AutoVC` / `StarGAN-VC` | Transforms the voice of read lyrics into singing style |
| `WaveNet` | Transforms singing voice into music |
| `Librosa` | Extraction of waveform |
