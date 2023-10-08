# ECE 276A - PR1

This code assumes you have the following libraries installed:

```
pandas
os
numpy
transforms3d
jax
matplotlib
tqdm
cv2
```

The code also assumes that in the directory where the code resides, also reside 2 folders
```
trainset
testset
```

The internal structure for these should be same although number of files inside imu, vicon, cam can change. It uses the number if files inside imu to decide how many files are there.

The variable ```panaroma_indices``` is used to specify which input imu files also have cam data. As in the case of the trainset [1,2,8,9] had it, so ```panaroma_indices = [0,1,7,8]```. This variable is located on line 465.

The file can be run via the command
```
python main.py
```

Please do let me know if there are any issues in running the file. The code was fully written on colab, I tested as a .py file once and it ran fully.

The code should create 21 files if ran with the same trainset and testset(11 graphs, 6 predicted panaromas, 4 vicon panaromas)

The code should show the following output on terminal:
```
Generating Graphs Training Set
File No: 1
Iteration 0 Cost: 155.09967
Iteration 499 Cost: 12.514511
File No: 2
Iteration 0 Cost: 316.99316
Iteration 499 Cost: 12.9544115
File No: 3
Iteration 0 Cost: 20.027517
Iteration 499 Cost: 8.566121
File No: 4
Iteration 0 Cost: 159.88147
Iteration 499 Cost: 22.315544
File No: 5
Iteration 0 Cost: 288.11798
Iteration 499 Cost: 18.128082
File No: 6
Iteration 0 Cost: 110.14618
Iteration 499 Cost: 17.27788
File No: 7
Iteration 0 Cost: 262.22284
Iteration 499 Cost: 29.144123
File No: 8
Iteration 0 Cost: 89.313156
Iteration 499 Cost: 11.568248
File No: 9
Iteration 0 Cost: 310.1938
Iteration 499 Cost: 10.985358
Generating Panaroma Training Set
File No: 1
Iteration 0 Cost: 155.09967
Iteration 499 Cost: 12.514511
  0% 0/1685 [00:00<?, ?it/s](76800, 4)
100% 1685/1685 [00:08<00:00, 200.99it/s]
100% 1685/1685 [00:05<00:00, 296.25it/s]
File No: 2
Iteration 0 Cost: 316.99316
Iteration 499 Cost: 12.9544115
100% 1259/1259 [00:05<00:00, 217.08it/s]
100% 1259/1259 [00:04<00:00, 273.57it/s]
File No: 8
Iteration 0 Cost: 89.313156
Iteration 499 Cost: 11.568248
100% 948/948 [00:04<00:00, 209.61it/s]
100% 948/948 [00:03<00:00, 293.94it/s]
File No: 9
Iteration 0 Cost: 310.1938
Iteration 499 Cost: 10.985358
100% 860/860 [00:03<00:00, 215.31it/s]
100% 860/860 [00:02<00:00, 297.60it/s]
Generating Graphs Test Set
File No: 1
Iteration 0 Cost: 51.856033
Iteration 499 Cost: 14.053486
File No: 2
Iteration 0 Cost: 25.764053
Iteration 499 Cost: 20.328323
Generating Panaroma Test Set
0
Iteration 0 Cost: 51.856033
Iteration 499 Cost: 14.053486
100% 911/911 [00:04<00:00, 208.57it/s]
1
Iteration 0 Cost: 25.764053
Iteration 499 Cost: 20.328323
100% 161/161 [00:00<00:00, 205.02it/s]
```