# Gaussian prior
python3 covs/run_hmc.py --seed=2 --weight_decay=100  \
	  --dir=runs/hmc/mnist/gaussian \
	    --dataset_name=mnist --model_name=mlp_classification \
	      --step_size=1.e-05 --trajectory_len=0.15 \
	        --num_iterations=100 --max_num_leapfrog_steps=15500 \
		  --num_burn_in_iterations=10

# Laplace prior
python3 covs/run_hmc.py --seed=0 --weight_decay=3.0 \
	  --dir=runs/hmc/mnist/laplace --dataset_name=mnist \
	    --model_name=mlp_classification --step_size=6.e-05 \
	      --trajectory_len=0.9 --num_iterations=100 \
	        --max_num_leapfrog_steps=15500 \
		  --num_burn_in_iterations=10 --prior_family=Laplace

# Student-T prior
python3 covs/run_hmc.py --seed=0 --weight_decay=10. \
	  --dir=runs/hmc/mnist/studentt --dataset_name=mnist \
	    --model_name=mlp_classification --step_size=1.e-4 --trajectory_len=0.49 --num_iterations=100 --max_num_leapfrog_steps=5000 \
	        --num_burn_in_iterations=10 --prior_family=StudentT \
		  --studentt_degrees_of_freedom=5.

# Gaussian prior, T=0.1
# python3 covs/run_hmc.py --seed=11 --weight_decay=100 --temperature=0.01 --dir=runs/hmc/mnist/temp --dataset_name=mnist --model_name=mlp_classification --step_size=6.3e-07 --trajectory_len=0.015 --num_iterations=100 --max_num_leapfrog_steps=25500 --num_burn_in_iterations=10

# EmpCov prior
python3 covs/run_hmc.py --seed=0 --weight_decay=100 \
	  --dir=runs/hmc/mnist/empcov --dataset_name=mnist \
	    --model_name=mlp_classification --step_size=1.e-05 \
	      --trajectory_len=0.15 --num_iterations=100 \
	        --max_num_leapfrog_steps=15500 \
		  --num_burn_in_iterations=10 --prior_family=EmpCovMLP \
		    --empcov_invcov_ckpt=empcov_covs/mnist_mlp_pca_inv_cov.npy \
		      --empcov_wd=100  
