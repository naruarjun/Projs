# ParticleFilterSLAM

- Place the ParticleFilterSLAM folder inside the ECE276A_PR2(downloaded from the course website) directory.

- Place the Kinect data inside ECE276A_PR2/data as ECE276A_PR2/data/Disparity20, ECE276A_PR2/data/Disparity21, ECE276A_PR2/data/RGB20, ECE276A_PR2/data/RGB21

To run the script run:
- Change dataset in line 115 in texture_utils.py to the dataset you want to run on
```
cd ParticleFilterSLAM
python texture_utils.py    #This should create a pickle file
python particle_filter.py --skip 2 --particles 500 --dataset 20
```
- skip - How many LiDAR frames to skip, 1 for no skipping
- particles - Number of particles in the filter
- dataset - which dataset to run 20, 21

This should plot one-by-one

- Occupancy Grid

- Trajectory

- Occupancy Grid + Trajectory

- Texture Map