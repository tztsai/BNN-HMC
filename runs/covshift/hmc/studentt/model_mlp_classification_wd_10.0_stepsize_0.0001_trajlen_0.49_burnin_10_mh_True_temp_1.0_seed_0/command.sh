covs/run_hmc.py --seed=0 --weight_decay=10. --dir=runs/hmc/mnist/studentt --dataset_name=mnist --model_name=mlp_classification --step_size=1.e-4 --trajectory_len=0.49 --num_iterations=100 --max_num_leapfrog_steps=5000 --num_burn_in_iterations=10 --prior_family=StudentT --studentt_degrees_of_freedom=5.
