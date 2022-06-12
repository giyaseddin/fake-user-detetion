train:
	echo "Starting training"
	python train.py --training_file data/fake_users.csv --cross_validate --n_k_fold 4 --model_folder trained_models/
	echo "Training done"


predict:
	echo "Starting prediction"
	python predict.py --test_file data/fake_users_test.csv --result_folder test_results/
	echo "Prediction done"


test:
	echo "Starting testing"
	python -m  unittest discover ./tests/
	echo "Testing testing"
