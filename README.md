## LASER: Lip Landmark Assisted Speaker Detection for Robustness

### [LASER-bench](https://huggingface.co/datasets/plnguyen2908/LASER-bench/tree/main) | [Project Page](https://plnguyen2908.github.io/LASER-ASD-project-page/) | [arXiv](https://arxiv.org/abs/2501.11899v2) | [Citation](#Citation)

[Le Thien Phuc Nguyen*](https://plnguyen2908.github.io/), [Zhuoran Yu*](https://www.zhuoranyu.com/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) (* *Equal Contribution*)

| Model           | Weight                                                                                        | Trained on        |
| --------------- | --------------------------------------------------------------------------------------------- | ----------------- |
| LoCoNet + LASER | [Checkpoint](https://drive.google.com/file/d/1IrntlKqzw5EYAVbyDupr5tk-H3q9kkoW/view?usp=sharing) | AVA-Activespeaker |
| TalkNCE + LASER | [Checkpoint](https://drive.google.com/file/d/1N8nFVybKXL7NFzJHMfo9x8FjJkUzEUc_/view?usp=sharing) | AVA-Activespeaker |
| TalkNet + LASER | [Checkpoint](https://drive.google.com/file/d/1-_9-_EWOVYerSmU7-XMtYLPwespfLuPW/view?usp=sharing) | AVA-Activespeaker |


### Dependencies


For LoCoNet environment, run the following command:

```
conda env create -f LoCoNet/landmark_loconet_environment.yml
conda activate landmark_loconet
```

For TalkNet environment, read the requirement.txt in TalkNet folder.

### Training Data preparation

We follow TalkNet's data preparation script to download and prepare the AVA dataset.

```
python LoCoNet/train.py --dataPathAVA AVADataPath --download 
```

Or equivalently, if you cannot run the above file, you can run

```
python LoCoNet/downloadAVA.py
```

, but you need to manually modify the savePath in that file

`AVADataPath` is the folder you want to save the AVA dataset and its preprocessing outputs, the details can be found in [here](https://github.com/TaoRuijie/TalkNet_ASD/blob/main/utils/tools.py#L34) . Please read them carefully.

After AVA dataset is downloaded, please change the DATA.dataPathAVA entry in the configs/multi.yaml file.

For Talkies and ASW, please refer to https://github.com/fuankarion/maas and https://github.com/clovaai/lookwhostalking respectively. Please make sure that the folder structure and csv file of the 2 datasets match AVA so that the all files can be run flawlessly.

### Download LASER-bench

You can run the file LASER_bench_download.py to download LASER_bench. 

```
python LASER_bench_download.py --save_path /path/to/data
```

The code will download 2 folders: noise_0 and noise_1 which stand for the subset of low noise and high noise in AVA format.

### Creating unsynchronized Dataset

Please modify the dataset path in shift_audio.py and then run the file to create the shifted audio version.

### Training script for LoCoNet

```
python -W ignore::UserWarning LoCoNet/train.py --cfg configs/multi.yaml OUTPUT_DIR <output directory>
```

### Training script for LASER on LoCoNet

Please modify hyperparameters in configs/multi.yaml accordingly before training

```
python -W ignore::UserWarning LoCoNet/train_landmark_loconet.py --cfg configs/multi.yaml OUTPUT_DIR <output directory>
```

### Creating landmark for the dataset

Please download mediapipe throught pip and modify the dataPath before running the following code:

```
python create_landmark.py
```

### Evaluation Script

Please modify the path to model's weight, model's hyperparameters, and dataPath in configs/multi.yaml before evaluating:

#### Synchronized dataset

For LASER on LoCoNet, please run:

```
python LoCoNet/test_mulicard_landmark.py --cfg configs/multi.yaml
```

For LoCoNet, please run:

```
python LoCoNet/test_mulicard.py --cfg configs/multi.yaml
```

#### Unsynchronized dataset

For LASER on LoCoNet, please run:

```
python LoCoNet/test_landmark_loconet.py --cfg configs/multi.yaml
```

For LoCoNet, please run:

```
python LoCoNet/test.py --cfg configs/multi.yaml
```

After running the evaluation file, please run

```
python utils/get_ava_active_speaker_performance_no_map.py -g <path to modified groundtruth csv file> -p <path to our result csv file>
```

because the get_ava_active_speaker_performance.py only calculates the mAP based on the number of positive examples which is not possible in our shifted dataset.

### Demo Code

Based on [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD)'s codebase, we have created demo codes for both LoCoNet and LASER. The codes are stored in demoLoCoNet.py and demoLoCoNet_landmark.py

Here are some instructions in using the code:

1. Create a demo folder and put the video you wanted in there.
2. in the demo python file, we support 3 features:

   * normal, synced audio
   * shifted audio by some amount of seconds
   * swap the first half and second half of the audio
3. Then, we support using different audio and video. You just need to specify the audio from a video you want to use:

   ```
   parser.add_argument('--videoName',             type=str,
                           default="demo_6_our",   help='Demo video name')
   parser.add_argument('--audioName',             type=str,
                           default="demo_6_our",   help='Demo video name')
   ```
4. Then, on line 515 and 516, if you want to shift by some second, change the shift_ms (in 1s = 1000ms). Or if you want to swap the audio, change swap = True. Otherwise, setting to 0 and false will do the normal demo.

### Citation

Please cite the following if our paper or code is helpful to your research.

```
@InProceedings{Nguyen_2026_WACV,
    author    = {Nguyen, Le Thien Phuc and Yu, Zhuoran and Lee, Yong Jae},
    title     = {LASER: Lip Landmark Assisted Speaker Detection for Robustness},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {7291-7300}
}
```

### Acknowledge

The code base of this project is studied from [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD) and [LoCoNet](https://github.com/SJTUwxz/LoCoNet_ASD/tree/main) which is a very easy-to-use ASD pipeline.

The contrative loss function is obtained through communication with the author of [TalkNCE](https://arxiv.org/pdf/2309.12306).



