python ./ibc/ibc/train_eval_custom.py\
    --alsologtostderr \
    --gin_file=ibc/ibc/configs/particle/mlp_mse_best.gin \
    --task=GAZE \
    --tag=mse \
    --add_time=True \
    --gin_bindings="GazeEnv.n_dim=2" \
    --gin_bindings="train_eval.dataset_path='/app/ibc/data/gaze/2d_allsessions_data.tfrecord'"  \

python ./ibc/ibc/train_eval_custom.py\
    --alsologtostderr \
    --gin_file=ibc/ibc/configs/particle/mlp_ebm_langevin_best.gin \
    --task=GAZE \
    --tag=ebm \
    --add_time=True \
    --gin_bindings="GazeEnv.n_dim=2" \
    --gin_bindings="train_eval.dataset_path='/app/ibc/data/gaze/2d_allsessions_data.tfrecord'"  \

python ./ibc/ibc/train_eval_custom.py\
    --alsologtostderr \
    --gin_file=ibc/ibc/configs/particle/mlp_mse_best.gin \
    --task=GAZE \
    --tag=mse-50 \
    --add_time=True \
    --gin_bindings="GazeEnv.n_dim=2" \
    --gin_bindings="train_eval.dataset_path='/app/ibc/data/gaze/2d_allsessions_data.tfrecord'"  \
    --gin_bindings="train_eval.sequence_length=50"

python ./ibc/ibc/train_eval_custom.py\
    --alsologtostderr \
    --gin_file=ibc/ibc/configs/particle/mlp_ebm_langevin_best.gin \
    --task=GAZE \
    --tag=ebm-50 \
    --add_time=True \
    --gin_bindings="GazeEnv.n_dim=2" \
    --gin_bindings="train_eval.dataset_path='/app/ibc/data/gaze/2d_allsessions_data.tfrecord'"  \
    --gin_bindings="train_eval.sequence_length=50"