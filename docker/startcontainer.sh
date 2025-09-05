docker build -t jp/iaisimclr \
  --build-arg USER_ID=$(id -u) \
  --build-arg HOST_UID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .


docker run -it --gpus all --name iaisimclr --ipc=host \
-p 6082:6082 \
-v /home/jprexl/Code/IaI_SimCLR/src:/home/user/src/ \
-v /home/jprexl/Data/:/home/user/data/ \
-v /home/jprexl/Results/:/home/user/results/ \
jp/iaisimclr
