## 下载数据集
1. 安装azcopy
2. 下载
```shell
python download_training.py --rgb --depth --only-left --output-dir ~/slam_data/TartanAir/ --azcopy
```

## DEMO: sfm
```shell 
 python demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt --disable_vis --reconstruction_path ./
```


## DEMO: EUROC
```shell
python demo.py --imagedir=data/mav0/cam0/data --calib=calib/euroc.txt --t0=150 --disable_vis --reconstruction_path ./output/euroc
```

convert format
```python
pose = np.load("reconstructions/euroc/poses.npy")
tstamps = np.load("reconstructions/euroc/tstamps.npy")
f = open('reconstructions/euroc/est_file.txt','w')
for (tstamp, pose_) in zip(tstamps, pose):
    str_pose = ''
    for num in pose_:
        str_pose += str(num) + ' '
    f.write("{} {}".format(tstamp+1540481046, str_pose[:-1])+'\n')
f.close()
```
