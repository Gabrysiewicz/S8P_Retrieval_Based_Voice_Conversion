<h1 align="center"> Retrieval-Based Voice Conversion </h1>

# Goal
The aim of the project was to train a model based on a voice and use it for interference with another existing voice, in this case the voice of an artist singing a musical piece. The purpose of this is to create a phenomenon known as Cover AI, which means overlaying one person’s voice onto the original artist’s voice.  

One of the elements was preparing the dataset and training an artificial intelligence model. This project focused on models used to work with voice and sound waves. To train the model, appropriate audio files are required, which must meet certain criteria:  
- the person’s voice,  
- files must be free from artifacts,  
- fragments containing silence must be short or removed,  
- only one voice may be present,  
- the recommended format is WAV,  
- file names cannot contain Polish characters,  
- one large file and many small files can be used,  
- apart from the voice, no other sounds should be audible.  

To train the model, RVC-WebUI available on GitHub will be used: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI. The main algorithm for training the model will be **rmvpe_gpu**, while the algorithm used for inference will also be **rmvpe**. To prepare the dataset, I asked a colleague to record several audio files and also used some unofficially available sources with audio materials.  


# Context
In the original plan, the final project was supposed to focus on image recognition using the YOLO model. However, due to my personal interests and after consulting with my supervisor, the topic of my project was changed to the use of artificial intelligence in AI Covers. This is a subject that interests me and in which I see great potential.  

# Introduction
There are many models used for training based on audio files, differing in their applications. One of the popular models is **Tacotron 2**, released by Google. In the training process, a large number of short audio files are collected, and then a single consolidated `metadata.csv` file is prepared, containing the file name and the text spoken in the given fragment. Tacotron 2 enables the creation of text-to-speech models, which, after entering text, read it using the trained voice. The drawback of this method is the lack of emotional reproduction and articulation in speech, which is why I used the **RVC (Retrieval-based Voice Conversion)** model instead.  

**RVC (Retrieval-based Voice Conversion)** is a modern voice conversion model that transforms the speech of one speaker into the speech of another while preserving the unique vocal characteristics of the target speaker. Thanks to advanced machine learning and deep learning techniques, RVC stands out for its effectiveness and precision. The model uses a large database of speech samples, analyzing the acoustic features of the source speech and comparing them with the target speaker’s samples, allowing for natural and realistic voice reproduction.  

RVC finds applications in many fields, such as film production, video games, voice assistant technologies, and telecommunication systems. It enables the creation of characters speaking in the voices of well-known actors and the personalization of assistant voices. However, this technology requires appropriate legal regulations to protect privacy and copyright. Despite these challenges, the potential of RVC is enormous, and its further development may bring even more advanced and diverse applications.  


# RVC WEBUI

Autor RVC-WebUI: `https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI`

<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/RVC-WebUI.png" />  
Retrieval-based Voice Conversion WebUI is a user interface based on speech processing technology that enables voice conversion using retrieval-based models. The main idea of this approach is that instead of creating separate conversion models for each individual, existing recordings or data are used to transform one person’s voice into another’s.  

The web interface allows users to upload recordings, which are then analyzed and converted using the appropriate algorithms and models available on the server. This is an example of an application that leverages advanced audio signal processing technologies to create new functionalities in the field of voice interaction and digital assistants.  

RVC-WebUI by default offers several models:  
- rmvpe,  
- crepe,  
- harvest,  
- dio.  

Since the **rmvpe** model is the most efficient and effective, it was used for both training and inference. In the following section, all of these models will be briefly described, but **rmvpe** was the main model applied in this project.  

The RVC-WebUI repository provides detailed installation instructions. Everything should work out of the box, however, it may be necessary to install additional packages to ensure proper support for the graphics card. One such element is the **Nvidia CUDA Toolkit** along with libraries that support a specific version of the toolkit. Detailed instructions can be found in the *Issues* section of the repository.  


# RVC2  
RVC2, the second generation of Retrieval-based Voice Conversion, is an advanced voice conversion model that builds on the success of its predecessor. With improved algorithms, RVC2 offers higher-quality conversion, more accurately reproducing the intonation, timbre, and dynamics of the target speaker’s voice. Enhanced methods of data acquisition and processing allow for faster and more precise conversion, which is practical for live applications. The model is also more flexible, supporting various speaking styles and languages, making it a versatile tool in different contexts. However, privacy and ethical concerns remain crucial for its appropriate use.  

# RMVPE  
RMVPE (Retrieval-based Multi-view Prediction and Embedding) is an advanced prediction and embedding model that leverages deep learning techniques to analyze multi-faceted data. Its main function is to forecast future values based on complex patterns in datasets, which makes it useful in areas such as sales forecasting, consumer behavior analysis, and recommendation systems. RMVPE analyzes data from multiple perspectives (multi-view), enabling a more precise and comprehensive understanding of relationships between data points. By utilizing large datasets and advanced algorithms, RMVPE delivers both effectiveness and accuracy, making it a valuable tool in the context of data analysis and artificial intelligence.  

# Crepe  
CREPE (Convolutional Representation for Pitch Estimation) is a modern pitch estimation model that uses convolutional neural networks to precisely analyze audio signals. Its main task is to accurately determine pitch in real time, which is crucial in fields such as music, speech, and acoustic research. CREPE stands out for its high accuracy and efficiency thanks to advanced deep learning techniques that allow effective processing and interpretation of complex audio signals. Due to its precision and speed, this model is widely applied in applications requiring accurate pitch analysis.  

# Harvest  
Harvest is an advanced pitch estimation algorithm designed for precise analysis of audio signals. Its primary goal is to accurately determine pitch in recordings, which is essential in areas such as music, speech, and acoustic research. Harvest leverages advanced signal processing techniques and time-frequency analysis to ensure high accuracy and reliability of pitch estimation. Thanks to its precision and efficiency, this algorithm is widely used in various applications, including music editing software, speech recognition systems, and scientific research in acoustics and phonetics.  

# PM  
PM (Parselmouth) is a tool for speech analysis and processing that leverages the Praat library, which is widely used among phonetics and linguistics researchers. PM enables precise acoustic measurements such as pitch, formants, intensity, and spectral energy flow, which are crucial for speech and sound analysis. Thanks to its integration with Praat, PM allows the automation of many analytical processes, significantly streamlining work with large datasets. This tool is valued for its accuracy, flexibility, and adaptability to specific research needs, finding applications in fields such as linguistics, phonetics, speech psychology, and speech technology.  

# Dio  
The DIO (Distributed Input Distributed Output) algorithm is a specialized tool for accurate estimation of tones or sound frequencies in audio signals. Developed in the context of neural networks, DIO employs distributed input and output techniques, enabling simultaneous processing of multiple data segments, which increases its efficiency. This algorithm is crucial in applications requiring precise time-frequency analysis, such as speech processing, music analysis, and speech recognition systems, providing high accuracy in pitch estimation and tracking tonal variations over time.  

# Data Cleaning  
Example of input data:  
<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/audacity_przed.png" />  

Result of cleaned data:  
<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/audacity_po.png" />  

As a result, a higher-quality file was obtained, which provides better training outcomes and shorter duration. This is a tedious process, as it requires listening to the entire file and often ends up with the need to remove a significant portion of the material due to noise. The Audacity program was used for cleaning, along with ffmpeg to convert the file into a format supported by Audacity.  

# Training Process  
The previously obtained dataset is placed in a convenient directory, and we can proceed to the training process. After installing RVC-WebUI, we run the command `python infer-web.py` in the repository directory, which will start a local server with the WebUI.  
```
(voice-ai-38) PS X:\AI\Retrieval-based-Voice-Conversion-WebUI> python .\infer-web.py
2024-06-13 02:05:27 | INFO | configs.config | Found GPU NVIDIA GeForce RTX 3050 Laptop GPU
2024-06-13 02:05:27 | INFO | configs.config | Half-precision floating-point: True, device: cuda:0
2024-06-13 02:05:37 | INFO | __main__ | Use Language: en_US
Running on local URL:  http://0.0.0.0:7865
```

<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/train.png" />

If everything is installed correctly, the GPU name should be visible. In the WebUI, we can:  
1. Go to the **Train** section,  
2. Set the model name,  
3. Specify the audio sampling quality,  
4. Set **pitch extraction** to true (required for singing, but unnecessary for normal speech),  
5. Set the RVC version to 2,  
6. Set the number of CPU threads according to your needs (this option does not matter when using a GPU),  
7. Set the path to the dataset directory,  
8. In the **Step2b** section, set the training model to **rmvpe_gpu** or **rmvpe**,  
8b. You can leave the number of epochs at 20 (recommended value is 20–30; increasing further does not improve model quality),  
8c. The batch size can also be increased, but it is not necessary,  
9. Start the training process by pressing the **One-trick training** button.  

RVC will first load all files that have the appropriate format.  
```
X:\AI\Kamyk\dataset-2/steam_07_q.wav    -> Success
X:\AI\Kamyk\dataset-2/steam_01_q.wav    -> Success
X:\AI\Kamyk\dataset-2/steam_09.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_05.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_06.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_03.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_10.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_02_q.wav    -> Success
X:\AI\Kamyk\dataset-2/wiersz_q.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_08.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_11.wav      -> Success
```

RVC will then split our dataset into smaller subsets.  
```
now-269,all-0,0_0.wav,(149, 768)
now-269,all-26,11_1.wav,(67, 768)
now-269,all-52,1_38.wav,(149, 768)
now-269,all-78,3_10.wav,(136, 768)
now-269,all-104,6_11.wav,(149, 768)
now-269,all-130,6_35.wav,(149, 768)
now-269,all-156,6_59.wav,(149, 768)
now-269,all-182,8_0.wav,(149, 768)
now-269,all-208,9_12.wav,(149, 768)
now-269,all-234,9_36.wav,(149, 768)
now-269,all-260,9_60.wav,(149, 768)
```

Finally, it will proceed to the model training process, which will consist of 20 epochs.  
Each epoch, assuming consistent hardware load, should take roughly the same amount of time.  
So, if the first epoch takes 10 minutes, we can estimate: 10 minutes * 20 epochs = 200 minutes / 60 = 3 hours 20 minutes.  
Additionally, if the dataset contains 12 minutes of material, the training time for one epoch will likely be about 12 minutes. At least in my case and with my GPU, this was how it proceeded.  
```
INFO:Kamyk-12M:Train Epoch: 1 [0%]
INFO:Kamyk-12M:[0, 0.0001]
INFO:Kamyk-12M:loss_disc=4.107, loss_gen=4.434, loss_fm=10.796,loss_mel=26.703, loss_kl=9.000
DEBUG:matplotlib:matplotlib data path: X:\miniconda3\envs\voice-ai-38\lib\site-packages\matplotlib\mpl-data
DEBUG:matplotlib:CONFIGDIR=C:\Users\kamil\.matplotlib
DEBUG:matplotlib:interactive is False
DEBUG:matplotlib:platform is win32
INFO:Kamyk-12M:====> Epoch: 1 [2024-06-13 02:20:09] | (0:12:58.170309)
INFO:Kamyk-12M:Train Epoch: 2 [47%]
INFO:Kamyk-12M:[200, 9.99875e-05]
INFO:Kamyk-12M:loss_disc=3.728, loss_gen=3.328, loss_fm=5.902,loss_mel=23.647, loss_kl=2.332
INFO:Kamyk-12M:====> Epoch: 2 [2024-06-13 02:33:38] | (0:13:28.865716)
INFO:Kamyk-12M:Train Epoch: 3 [94%]
INFO:Kamyk-12M:[400, 9.99750015625e-05]
INFO:Kamyk-12M:loss_disc=3.762, loss_gen=3.529, loss_fm=9.578,loss_mel=22.949, loss_kl=2.516
INFO:Kamyk-12M:====> Epoch: 3 [2024-06-13 02:46:11] | (0:12:32.941089)
INFO:Kamyk-12M:====> Epoch: 4 [2024-06-13 02:58:04] | (0:11:52.737444)
INFO:Kamyk-12M:Train Epoch: 5 [41%]
INFO:Kamyk-12M:[600, 9.995000937421877e-05]
INFO:Kamyk-12M:loss_disc=4.085, loss_gen=3.188, loss_fm=4.325,loss_mel=21.281, loss_kl=2.375
INFO:root:Saving model and optimizer state at epoch 5 to ./logs\Kamyk-12M\G_680.pth
INFO:root:Saving model and optimizer state at epoch 5 to ./logs\Kamyk-12M\D_680.pth
INFO:Kamyk-12M:====> Epoch: 5 [2024-06-13 03:10:15] | (0:12:11.262043)
INFO:Kamyk-12M:Train Epoch: 6 [88%]
INFO:Kamyk-12M:[800, 9.993751562304699e-05]
INFO:Kamyk-12M:loss_disc=3.936, loss_gen=3.558, loss_fm=9.403,loss_mel=21.352, loss_kl=1.536
INFO:Kamyk-12M:====> Epoch: 6 [2024-06-13 03:22:33] | (0:12:17.753304)
INFO:Kamyk-12M:====> Epoch: 7 [2024-06-13 03:34:33] | (0:12:00.374585)
INFO:Kamyk-12M:Train Epoch: 8 [35%]
INFO:Kamyk-12M:[1000, 9.991253280566489e-05]
INFO:Kamyk-12M:loss_disc=3.439, loss_gen=3.976, loss_fm=10.359,loss_mel=19.853, loss_kl=1.834
INFO:Kamyk-12M:====> Epoch: 8 [2024-06-13 03:46:31] | (0:11:57.633450)
INFO:Kamyk-12M:Train Epoch: 9 [82%]
INFO:Kamyk-12M:[1200, 9.990004373906418e-05]
INFO:Kamyk-12M:loss_disc=3.971, loss_gen=3.518, loss_fm=12.544,loss_mel=22.534, loss_kl=2.195
INFO:Kamyk-12M:====> Epoch: 9 [2024-06-13 03:58:36] | (0:12:04.944736)
INFO:root:Saving model and optimizer state at epoch 10 to ./logs\Kamyk-12M\G_1360.pth
INFO:root:Saving model and optimizer state at epoch 10 to ./logs\Kamyk-12M\D_1360.pth
INFO:Kamyk-12M:====> Epoch: 10 [2024-06-13 04:10:53] | (0:12:16.573370)
INFO:Kamyk-12M:Train Epoch: 11 [29%]
INFO:Kamyk-12M:[1400, 9.987507028906759e-05]
INFO:Kamyk-12M:loss_disc=4.299, loss_gen=2.925, loss_fm=9.271,loss_mel=21.341, loss_kl=1.809
INFO:Kamyk-12M:====> Epoch: 11 [2024-06-13 04:23:02] | (0:12:09.339499)
INFO:Kamyk-12M:Train Epoch: 12 [76%]
INFO:Kamyk-12M:[1600, 9.986258590528146e-05]
INFO:Kamyk-12M:loss_disc=4.257, loss_gen=3.766, loss_fm=13.422,loss_mel=22.387, loss_kl=2.152
INFO:Kamyk-12M:====> Epoch: 12 [2024-06-13 04:35:20] | (0:12:17.908730)
INFO:Kamyk-12M:====> Epoch: 13 [2024-06-13 04:47:16] | (0:11:56.373045)
INFO:Kamyk-12M:Train Epoch: 14 [24%]
INFO:Kamyk-12M:[1800, 9.983762181915804e-05]
INFO:Kamyk-12M:loss_disc=4.075, loss_gen=2.999, loss_fm=9.753,loss_mel=22.000, loss_kl=1.708
INFO:Kamyk-12M:====> Epoch: 14 [2024-06-13 04:59:00] | (0:11:43.946182)
INFO:Kamyk-12M:Train Epoch: 15 [71%]
INFO:Kamyk-12M:[2000, 9.982514211643064e-05]
INFO:Kamyk-12M:loss_disc=3.967, loss_gen=3.317, loss_fm=9.745,loss_mel=20.010, loss_kl=2.327
INFO:root:Saving model and optimizer state at epoch 15 to ./logs\Kamyk-12M\G_2040.pth
INFO:root:Saving model and optimizer state at epoch 15 to ./logs\Kamyk-12M\D_2040.pth
INFO:Kamyk-12M:====> Epoch: 15 [2024-06-13 05:10:55] | (0:11:54.278821)
INFO:Kamyk-12M:====> Epoch: 16 [2024-06-13 05:22:55] | (0:12:00.282794)
INFO:Kamyk-12M:Train Epoch: 17 [18%]
INFO:Kamyk-12M:[2200, 9.980018739066937e-05]
INFO:Kamyk-12M:loss_disc=4.610, loss_gen=3.509, loss_fm=13.866,loss_mel=21.093, loss_kl=1.551
INFO:Kamyk-12M:====> Epoch: 17 [2024-06-13 05:34:46] | (0:11:51.657800)
INFO:Kamyk-12M:Train Epoch: 18 [65%]
INFO:Kamyk-12M:[2400, 9.978771236724554e-05]
INFO:Kamyk-12M:loss_disc=4.083, loss_gen=3.007, loss_fm=6.480,loss_mel=21.605, loss_kl=2.012
INFO:Kamyk-12M:====> Epoch: 18 [2024-06-13 05:46:49] | (0:12:02.831979)
INFO:Kamyk-12M:====> Epoch: 19 [2024-06-13 05:58:47] | (0:11:57.786244)
INFO:Kamyk-12M:Train Epoch: 20 [12%]
INFO:Kamyk-12M:[2600, 9.976276699833672e-05]
INFO:Kamyk-12M:loss_disc=4.007, loss_gen=3.607, loss_fm=10.471,loss_mel=23.102, loss_kl=2.028
INFO:root:Saving model and optimizer state at epoch 20 to ./logs\Kamyk-12M\G_2720.pth
INFO:root:Saving model and optimizer state at epoch 20 to ./logs\Kamyk-12M\D_2720.pth
INFO:Kamyk-12M:====> Epoch: 20 [2024-06-13 06:10:35] | (0:11:48.248414)
INFO:Kamyk-12M:Training is done. The program is closed.
INFO:Kamyk-12M:saving final ckpt:Success.
```

# Inference
<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/interferencja.png" />

The trained model can be used in the inference process. Inference is a physical phenomenon involving the superposition of waves, resulting in a new wave pattern. To create an AI cover, you will need a version of the track containing only vocals, for example, an acapella. From the dropdown list, select your model and provide the path to the vocal track.  

Pressing the **"Convert"** button starts the inference process. If the inference result is not satisfactory, you can try adjusting the relevant parameters. You can change the **octave**, for example lowering it to -12, which can be helpful if the model was trained on low-pitched voices and the song is performed by someone with a high-pitched voice.  

**Resampling** is used when the model was trained on a small number of samples and needs to increase them. It is recommended to avoid manipulating this option, as it may introduce artifacts. Another option allows for more accurate reproduction relative to the original artist (value 0) or more aligned with the model's voice (value 1). Adjusting this parameter can help achieve better results.  

Other parameters relate to silence, audible breaths during speech, and accent. These should also be adjusted depending on the expected outcome.  

The resulting file should be carefully listened to. If the result is satisfactory, it can be downloaded and further processed.  

# Summary

1. Start by collecting a dataset containing the voice of the `actor` whose voice will be overlaid.  
2. Next, prepare and clean the audio files, including removing noise and unwanted artifacts.  
3. Choose one of the available algorithms and proceed to train the model, which generates a `.pth` file containing the trained model.  
4. Obtain an audio file of the singer performing the song (acapella or voice only).  
4b. If necessary, clean the acapella or voice-only track as well, removing any unwanted noise.  
5. Perform the inference process between the trained model and the acapella track.  
6. The resulting audio file can be further edited, adding music or making other modifications in an audio editing program.  


<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/summary.png" />

The entire process requires relatively:  
- A small initial dataset.  
- Time-consuming data preparation, especially cleaning and preparing audio files.  
- Significant computing power or long computation times, depending on the chosen algorithm and dataset size.  
- Patience and skill to select appropriate model and inference parameters.  

Considering current hardware capabilities and the final result, creating an AI Cover is a fascinating opportunity, which can be easily extended with additional functionalities such as detecting potential voices suitable for singing, real-time translation using one’s own voice, attempts to detect deepfakes or spoofing. The project offers possibilities worth exploring at the level of a master’s thesis.  






