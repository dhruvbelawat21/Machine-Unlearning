
def prog1(): 
    import sys
    import os

    # ✅ Add project root to sys.path (so imports work correctly)
    project_root = r"C:\Users\dhruv\Downloads\MachineUnlearning-main\MachineUnlearning-main"
    sys.path.append(project_root)

    from Unlearner.RNNUnlearner import RNNUNlearner
    from Applications.CanaryRemoval.CanaryRemoval import get_params_by_model_name, load_data, unlearn_canary

    # Canary string setup
    CANARY_STR = "`my telephone number is {}!' said alice.\n\n  "
    CANARY_START = "`my telephone number is "
    canary_number = "0123456789"
    n_layers = 2
    n_units = 512

    # ✅ Use absolute paths instead of '../'
    data_path = os.path.join(project_root, "train_test_data", "Alice", "alice_in_wonderland.txt")
    weight_path = os.path.join(project_root, "models", "LSTM", 
        "checkpoint_lambda=0.0001-canary_number=0123456789-canary_reps=6-embedding_dim=64-seqlen=24-dropout=0.0.ckpt"
    )

    # Load params and canary
    lambda_, canary_number, canary_reps, embedding_dim, seq_length, p_dropout = get_params_by_model_name(weight_path)
    canary = CANARY_STR.format(canary_number)

    # Load training data
    x_train, y_train, idx2char = load_data(data_path, seq_length, canary, canary_reps)

    # Initialize unlearner
    unlearner = RNNUNlearner(
        x_train, y_train, embedding_dim, idx2char, lambda_, weight_path,
        CANARY_START, canary_number, canary_reps,
        n_layers=n_layers, n_units=n_units, p_dropout=p_dropout
    )

    # Generate data with canary start
    unlearner.generate_data(start_str=CANARY_START)

    # Evaluate perplexity
    canary_seq_perplexity = unlearner.calc_sequence_perplexity(canary_number)

    n_samples = 100000
    perplexity_distribution = unlearner.calc_perplexity_distribution(no_samples=n_samples, plot=True, only_digits=False)
    canary_exposure = unlearner.approx_exposure([canary_seq_perplexity], perplexity_distribution, only_digits=False)

    # Canary removal
    replacement_strs = ['not there ', 'dry enough']

    # First order params
    taus = [0.0003, 0.0009]
    batch_size_fo = 64

    # Second order params
    batch_size_so = 500
    damping = 0.1
    iterations = 30
    scales = [39000, 36000]

    # Run first-order unlearning
    res = unlearn_canary(unlearner, data_path, seq_length, canary_reps,
                        tau=taus[0], order=1, batch_size=batch_size_fo,
                        scale=1, damping=0.0, iterations=1, replace_char=replacement_strs[0])

    res = unlearn_canary(unlearner, data_path, seq_length, canary_reps,
                        tau=taus[1], order=1, batch_size=batch_size_fo,
                        scale=1, damping=0.0, iterations=1, replace_char=replacement_strs[1])

    # Run second-order unlearning
    res = unlearn_canary(unlearner, data_path, seq_length, canary_reps,
                        tau=1.0, order=2, batch_size=500,
                        scale=scales[0], damping=0.1, iterations=30, replace_char=replacement_strs[0])

    res = unlearn_canary(unlearner, data_path, seq_length, canary_reps,
                        tau=1.0, order=2, batch_size=batch_size_so,
                        scale=scales[1], damping=0.1, iterations=iterations, replace_char=replacement_strs[1])
    



def prog2():
    print("start")



prog1()