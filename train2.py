import sys
# stdout = sys.stdout
import argparse
import os
import time
from torch.utils.tensorboard import SummaryWriter
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from super_gradients.training.transforms.transforms import DetectionMosaic, DetectionRandomAffine, DetectionHSV, \
    DetectionHorizontalFlip, DetectionPaddedRescale, DetectionStandardize, DetectionTargetsFormatTransform
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed
from super_gradients.training.utils.detection_utils import CrowdDetectionCollateFN
from super_gradients.training.pre_launch_callbacks import modify_params_for_qat
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training import dataloaders
from super_gradients.training import Trainer
from super_gradients.training import models
import super_gradients
import argparse
import torch
import time
import yaml
import json
import os
# sys.stdout = stdout

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--data", type=str, required=True,
                    help="path to data.yaml")
    ap.add_argument("-n", "--name", type=str,
                    help="Checkpoint dir name")
    ap.add_argument("-b", "--batch", type=int, default=6,
                    help="Training batch size")
    ap.add_argument("-e", "--epoch", type=int, default=100,
                    help="Training number of epochs")
    ap.add_argument("-j", "--worker", type=int, default=2,
                    help="Training number of workers")
    ap.add_argument("-m", "--model", type=str, default='yolo_nas_s',
                choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'],
                help="Model type (eg: yolo_nas_s)")
    ap.add_argument("-w", "--weight", type=str, default='coco',
                    help="path to pre-trained model weight")
    ap.add_argument("-r", "--resume", action='store_true',
                    help="to resume model training")
    ap.add_argument("-s", "--size", type=int, default=640,
                    help="input image size")
    ap.add_argument("--gpus", action='store_true',
                help="Run on all gpus")
    ap.add_argument("--cpu", action='store_true',
                help="Run on CPU")
    ap.add_argument("--qat", action='store_true',
                help="Quantization Aware Training")
    
    
    # train_params
    ap.add_argument("--warmup_mode", type=str, default='linear_epoch_step',
                    help="Warmup Mode")
    ap.add_argument("--warmup_initial_lr", type=float, default=1e-7,
                    help="Warmup Initial LR")
    ap.add_argument("--lr_warmup_epochs", type=int, default=10,
                    help="LR Warmup Epochs")
    ap.add_argument("--initial_lr", type=float, default=1e-4,
                    help="Inital LR")
    ap.add_argument("--lr_mode", type=str, default='cosine',
                    help="LR Mode")
    ap.add_argument("--cosine_final_lr_ratio", type=float, default=0.1,
                    help="Cosine Final LR Ratio")
    ap.add_argument("--optimizer", type=str, default='AdamW',
                    help="Optimizer")
    ap.add_argument("--weight_decay", type=float, default=0.0001,
                    help="Weight Decay")
    args = vars(ap.parse_args())

    # 텐서보드 writer 초기화
    # writer = SummaryWriter(f'runs/{args["name"] if args["name"] else "experiment"}_{time.strftime("%Y%m%d-%H%M%S")}')
    writer = SummaryWriter(f'runs/{args["name"]}_{time.strftime("%Y%m%d-%H%M%S")}')

    # Start Time
    s_time = time.time()

    if args['name'] is None:
        name = 'train'
    else:
        name = args['name']
    
    if args['resume']:
        name = os.path.split(args['weight'])[0].split('/')[-1]
    else:
        n = 0
        while True:
            if not os.path.exists(os.path.join('runs', f'{name}{n}')):
                name = f'{name}{n}'
                os.makedirs(os.path.join('runs', name))
                break
            else:
                n += 1
    print(f"[INFO] Checkpoints saved in \033[1m{os.path.join('runs', name)}\033[0m")
    
    # Training on GPU or CPU
    if args['cpu']:
        print('[INFO] Training on \033[1mCPU\033[0m')
        # 디바이스 설정
        super_gradients.setup_device(device='cpu', multi_gpu=False, num_gpus=0)        
        trainer = Trainer(experiment_name=name, ckpt_root_dir='runs')
    elif args['gpus']:
        print(f'[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m')
        trainer = Trainer(experiment_name=name, ckpt_root_dir='runs', multi_gpu=args['gpus'])
    else:
        print(f'[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m')
        trainer = Trainer(experiment_name=name, ckpt_root_dir='runs')

    # Load Path Params
    yaml_params = yaml.safe_load(open(args['data'], 'r'))
    with open(os.path.join(yaml_params['Dir'], yaml_params['labels']['train'])) as f:
        no_class = len(json.load(f)['categories'])
        f.close()
    print(f"\033[1m[INFO] Number of Classes: {no_class}\033[0m")
    
    # Reain Dataset
    trainset = COCOFormatDetectionDataset(data_dir=yaml_params['Dir'],
                                      images_dir=yaml_params['images']['train'],
                                      json_annotation_file=yaml_params['labels']['train'],
                                      input_dim=(args['size'], args['size']),
                                      ignore_empty_annotations=False,
                                      transforms=[
                                          DetectionMosaic(prob=1., input_dim=(args['size'], args['size'])),
                                          DetectionRandomAffine(degrees=0., scales=(0.5, 1.5), shear=0.,
                                                                target_size=(args['size'], args['size']),
                                                                filter_box_candidates=False, border_value=128),
                                          DetectionHSV(prob=1., hgain=5, vgain=30, sgain=30),
                                          DetectionHorizontalFlip(prob=0.5),
                                          DetectionPaddedRescale(input_dim=(args['size'], args['size']), max_targets=300),
                                          DetectionStandardize(max_value=255),
                                          DetectionTargetsFormatTransform(max_targets=300, input_dim=(args['size'], args['size']),
                                                                          output_format="LABEL_CXCYWH")
                                      ])
    train_dataloader_params = {
                                "shuffle": True,
                                "batch_size": args['batch'],
                                "drop_last": False,
                                "pin_memory": True,
                                "collate_fn": CrowdDetectionCollateFN(),
                                "worker_init_fn": worker_init_reset_seed,
                                "min_samples": 512
                                }
    # Valid Data
    valset = COCOFormatDetectionDataset(data_dir=yaml_params['Dir'],
                                    images_dir=yaml_params['images']['val'],
                                    json_annotation_file=yaml_params['labels']['val'],
                                    input_dim=(args['size'], args['size']),
                                    ignore_empty_annotations=False,
                                    transforms=[
                                        DetectionPaddedRescale(input_dim=(args['size'], args['size']), max_targets=300),
                                        DetectionStandardize(max_value=255),
                                        DetectionTargetsFormatTransform(max_targets=300, input_dim=(args['size'], args['size']),
                                                                        output_format="LABEL_CXCYWH")
                                    ])
    val_dataloader_params = {
                            "shuffle": False,
                            "batch_size": int(args['batch']*2),
                            "num_workers": args['worker'],
                            "drop_last": False,
                            "pin_memory": True,
                            "collate_fn": CrowdDetectionCollateFN(),
                            "worker_init_fn": worker_init_reset_seed
                            }

    # Test Data
    if 'test' in (yaml_params['images'].keys() or yaml_params['labels'].keys()):
        testset = COCOFormatDetectionDataset(data_dir=yaml_params['Dir'],
                                    images_dir=yaml_params['images']['test'],
                                    json_annotation_file=yaml_params['labels']['test'],
                                    input_dim=(args['size'], args['size']),
                                    ignore_empty_annotations=False,
                                    transforms=[
                                        DetectionPaddedRescale(input_dim=(args['size'], args['size']), max_targets=300),
                                        DetectionStandardize(max_value=255),
                                        DetectionTargetsFormatTransform(max_targets=300, input_dim=(args['size'], args['size']),
                                                                        output_format="LABEL_CXCYWH")
                                    ])
        test_loader = dataloaders.get(dataset=testset, dataloader_params={
                                        "shuffle": False,
                                        "batch_size": int(args['batch']*2),
                                        "num_workers": args['worker'],
                                        "drop_last": False,
                                        "pin_memory": True,
                                        "collate_fn": CrowdDetectionCollateFN(),
                                        "worker_init_fn": worker_init_reset_seed
                                    })

    # To Resume Training or re-train
    if args['resume'] or args["weight"].endswith('.pth'):
        model = models.get(
            args['model'],
            num_classes=no_class,
            checkpoint_path=args["weight"]
        )
    else:
        model = models.get(
            args['model'],
            num_classes=no_class, 
            pretrained_weights=args["weight"]
        )
    if args['cpu']:
        mixed_precision = False
    else:
        mixed_precision = True

    train_params = {
        'silent_mode': False, # 훈련 중 출력 메세지를 표시할지 여부
        "average_best_models":True, # 최고 모델들의 평균을 사용하여 훈련
        "warmup_mode": args['warmup_mode'], # 초기 학습률을 선형으로 증가시키는 워밍업 모드
        "warmup_initial_lr": args['warmup_initial_lr'],  # 워밍업 초기 학습률
        "lr_warmup_epochs": args['lr_warmup_epochs'], # 워밍업 에폴 수
        "initial_lr": args['initial_lr'], # 초기 학습률
        "lr_mode": args['lr_mode'], # 학습률 감소 모드(코사인 감소)
        "cosine_final_lr_ratio": args['cosine_final_lr_ratio'], # 코사인 감소의 최종 합습률 비율
        "optimizer": args['optimizer'], # 옵티마이저
        "optimizer_params": {"weight_decay": args['weight_decay']}, # 옵티마이저 매개변수 (가중치 감쇠)
        "zero_weight_decay_on_bias_and_bn": True, # 바이어스 및 배치 정규화에 대한 가중치 감쇠 제외 여부
        "ema": True, # 지수 이동 평균 사용 여부
        "ema_params": {"decay": 0.9, "decay_type": "threshold"}, # 지수이동 평균 매개변수
        "max_epochs": args['epoch'], # 최대 에폭 수
        "mixed_precision": mixed_precision, # 혼합 정밀도 훈련 사용 여부
        "loss": PPYoloELoss( # 손실함수
            use_static_assigner=False,
            num_classes=no_class,
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=no_class,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7 #NMS 알고리즘의 임계값. 두 경계상자가 중복이 되면 신뢰도가 낮은 것을 제거하는 기준이 된다.
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }

    # to Resume Training
    if args['resume']:
        train_params['resume'] = True
    
    # Print Training Params
    # Print Training Params
    print('[INFO] Training Params:\n', train_params)

    train_loader = dataloaders.get(dataset=trainset,
                                dataloader_params=train_dataloader_params)
    valid_loader = dataloaders.get(dataset=valset,
                                dataloader_params=val_dataloader_params)
                                
    # Model Training...
    trainer.train(
        model=model, 
        training_params=train_params, 
        train_loader=train_loader, 
        valid_loader=valid_loader
    )

    # Load best model
    best_model = models.get(args['model'],
                            num_classes=no_class,
                            checkpoint_path=os.path.join('runs', name, 'ckpt_best.pth'))
    
    # Evaluating on Val Dataset
    eval_model = trainer.test(model=best_model,
                    test_loader=valid_loader,
                    test_metrics_list=DetectionMetrics_050(score_thres=0.1, 
                                                        top_k_predictions=300, 
                                                        num_cls=no_class, 
                                                        normalize_targets=True, 
                                                        post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                                nms_top_k=1000, 
                                                                                                                max_predictions=300,                                                                              
                                                                                                                nms_threshold=0.7)
                                                        ))
    print('\033[1m [INFO] Validating Model:\033[0m')
    for i in eval_model:
        print(f"{i}: {float(eval_model[i])}")

    # Evaluating on Test Dataset
    if 'test' in (yaml_params['images'].keys() or yaml_params['labels'].keys()):
        test_result = trainer.test(model=best_model,
                    test_loader=test_loader,
                    test_metrics_list=DetectionMetrics_050(score_thres=0.1, 
                                                        top_k_predictions=300, 
                                                        normalize_targets=True, 
                                                        post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                                nms_top_k=1000, 
                                                                                                                max_predictions=300,                                                                              
                                                                                                                nms_threshold=0.7)
                                                        ))
        print('\033[1m [INFO] Test Results:\033[0m')
        for i in test_result:
            print(f"{i}: {float(test_result[i])}")
    # 오류 도출
    # for epoch in range(args['epoch']):
    #     # Training step
    #     # Log training metrics
    #     writer.add_scalar('Loss/train', train_loss, epoch)
    #     writer.add_scalar('Accuracy/train', train_accuracy, epoch)

    #     # Validation step
    #     # Log validation metrics
    #     writer.add_scalar('Loss/valid', valid_loss, epoch)
    #     writer.add_scalar('Accuracy/valid', valid_accuracy, epoch)

    # Close the writer when done
    writer.close()
    print(f'[INFO] Training Completed in \033[1m{(time.time()-s_time)/3600} Hours\033[0m')
