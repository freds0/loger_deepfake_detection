python train_loger.py --config-name loger_fsm_combined \
ckpt_path=outputs_v0.1/loger_fsm_combined/2026-07-17/00-57-53/checkpoints/last.ckpt \
trainer.max_steps=50000 scheduler.t_max=50000 \
data.batch_size=200
