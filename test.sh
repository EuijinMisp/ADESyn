work_dir=$(pwd)
MODE="test"
IMG_SIZE=192
NGPU=0
BATCH_SIZE=3

TEST_OUT="${work_dir}/test"
MODEL_G_PATH="${work_dir}/train/netG_epoch_40.pth"
IMAGE_DIR="${work_dir}/data/NP_3D/setA"

cd ./code
python main.py \
  --mode="$MODE" \
  --imageSize=$IMG_SIZE \
  --ngpu=$NGPU \
  --test_out="$TEST_OUT" \
  --load_model_G="$MODEL_G_PATH" \
  --batch_size=$BATCH_SIZE \
  --img_dir="$IMAGE_DIR" 

done
