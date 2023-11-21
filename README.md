# 1 SHA-SCP
This repo is the official implementation for A UI element spatial hierarchy aware smartphone user click behavior prediction method
## 1.1 The framework of SHA-SCP
![Example Image](./SHA-SCP-frame.jpg)

# 2 Prerequisites
- Python 3.10.4
- PyTorch 1.11.0
- math, sklearn, tensorboardX

# 3 Running

## 3.1 Training & Testing

- Change the config depending on what you want
```python
cd ..
# run on the PAMAP2
python main.py --dataset pamap
# run on the OPPORTUNITY
python main.py --dataset opp
# change the learning rate
python main.py --lr 0.0001
# change the batch size
python main.py --batch_size 64
```

# 4 Acknowledgements
This repo is based on [MCD_DA](https://github.com/mil-tokyo/MCD_DA). Great thanks to the original authors for their work!
 
 
# 5 Citation

Please cite this work if you find it useful.

If you have any question, feel free to contact: `shmiao@zju.edu.cn`
