work_dir=$(pwd)
MODE="train"
IMG_SIZE=192
BATCH_SIZE=8
EPOCHS=40
N_ITER=5000
N_CRITIC=5
LR=1e-4
NGPU=0
RETRAIN=0
INIT_EPOCHS=0
RECON_LAMBDA=10
CLS_LAMBDA=30
NSERIAL=3

TRAIN_OUT="${work_dir}/train"
IMAGE_DIR="${work_dir}/data/NP_3D/setA"


cd ./code
python main.py \
  --mode="$MODE" \
  --retrain=$RETRAIN \
  --imageSize=$IMG_SIZE \
  --batch_size=$BATCH_SIZE \
  --epochs=$EPOCHS \
  --init_epochs=$INIT_EPOCHS \
  --niter=$N_ITER \
  --n_critic=$N_CRITIC \
  --lr=$LR \
  --ngpu=$NGPU \
  --lambda_recon=$RECON_LAMBDA \
  --lambda_cls=$CLS_LAMBDA \
  --train_out="$TRAIN_OUT" \
  --img_dir="$IMAGE_DIR" \
  --nserial=$NSERIAL 
done
