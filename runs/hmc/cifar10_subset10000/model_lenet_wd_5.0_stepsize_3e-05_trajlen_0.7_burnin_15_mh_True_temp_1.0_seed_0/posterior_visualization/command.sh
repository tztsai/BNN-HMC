python make_posterior_surface_plot.py --dataset_name=cifar10 --subset_train_to=10000 --model_name=lenet --weight_decay=5 --temperature=1 --dir runs/hmc/cifar10_subset10000/model_lenet_wd_5.0_stepsize_3e-05_trajlen_0.7_burnin_15_mh_True_temp_1.0_seed_0 --checkpoint1 model_step_16.pt --checkpoint2 model_step_56.pt --checkpoint3 model_step_96.pt --limit_bottom=-0.75 --limit_left=-0.75 --limit_right=1.75 --limit_top=1.75 --grid_size=50