# Maze-Solving-Dobot-using-Agentic-AI

## For running use the following command

#### 1. First, run the calib file using the following command:
     python calib.py   --camera 0   --ids 0 1 2 3   --world "200.29,-76.15; 202.20,92.377; 353.35,-75.97; 358.35,83.20"   --dict 6X6_250  --mode centers   --save field_calib.yml

#### 2. Second, run the midterm2_1 fole
     python midterm2_1.py  --camera-index 0  --start-color red   --calib-yaml field_calib.yml  --safe-z 40  --trace-z -30  --mm-per-sec 150   --port COM3   --execute


## Note 
#### Before running the calib file make sure:
1. Dobot is at an optimal height such that you can get a sufficient amount of work area under tha camera vision (considering taht camera is attached to dobot)
2. ArUco markers should be placed inside the field vision of the camera in the four corners (tr, tl, br, bl)
3. Replace the world coordinates in the run command with the actual coordinates of the ArUco markers that you can find by manually moving the dobot end-effector to the ArUco marker and noting down the x and y coordinate
4. The ArUco marker's id: 0 = tl, 1 = tr, 2 = br, 3 = bl
5. The world coordinate should also be placed in the run command in the same sequence
