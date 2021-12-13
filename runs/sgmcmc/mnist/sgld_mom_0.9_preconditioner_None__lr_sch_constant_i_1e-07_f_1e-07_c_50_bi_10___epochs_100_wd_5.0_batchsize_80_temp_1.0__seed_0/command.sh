python run_sgmcmc.py --seed=0 --weight_decay=5. --dir=runs/sgmcmc/mnist --dataset_name=mnist --model_name=mlp_classification --init_step_size=1e-7 --final_step_size=1e-7 --num_epochs=100 --num_burnin_epochs=10 --momentum=0.9 --eval_freq=10 --batch_size=80 --save_freq=100
