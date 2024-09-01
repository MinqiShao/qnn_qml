"""
generate mutants guided by coverage
"""

import random
import time
import numpy as np
import os
import argparse, pickle
import shutil

from image_queue import ImageInputCorpus, TensorInputCorpus
from fuzzone import build_fetch_function
from queue import Seed
from Fuzzer import Fuzzer
from tools.mutator import Mutator


def image_preprocessing(x):
    x = np.expand_dims(x.numpy(), axis=-1)
    return x

def metadata_function(meta_batches):
    return meta_batches

def image_mutation_function(batch_num):
    # Given a seed, randomly generate a batch of mutants
    def func(seed):
        return Mutator.image_random_mutate(seed, batch_num)

    return func

def objective_function(seed, names):
    metadata = seed.metadata
    ground_truth = seed.ground_truth
    assert (names is not None)
    results = []
    if len(metadata) == 1:
        # To check whether it is an adversarial sample
        if metadata[0] != ground_truth:
            results.append('')
    else:
        # To check whether it has different results between original model and quantized model
        # metadata[0] is the result of original model while metadata[1:] is the results of other models.
        # We use the string adv to represent different types;
        # adv = '' means the seed is not an adversarial sample in original model but has a different result in the
        # quantized version.  adv = 'a' means the seed is adversarial sample and has a different results in quan models.
        if metadata[0] == ground_truth:
            adv = ''
        else:
            adv = 'a'
        count = 1
        while count < len(metadata):
            if metadata[count] != metadata[0]:
                results.append(names[count] + adv)
            count += 1

    # results records the suffix for the name of the failed tests
    return results

def iterate_function(names):
    def func(queue, root_seed, parent, mutated_coverage_list, mutated_data_batches, mutated_metadata_list,
             objective_function):

        ref_batches, batches, cl_batches, l0_batches, linf_batches = mutated_data_batches

        successed = False
        bug_found = False
        # For each mutant in the batch, we will check the coverage and whether it is a failed test
        for idx in range(len(mutated_coverage_list)):

            input = Seed(cl_batches[idx], mutated_coverage_list[idx], root_seed, parent, mutated_metadata_list[:, idx],
                         parent.ground_truth, l0_batches[idx], linf_batches[idx])

            # The implementation for the isFailedTest() in Algorithm 1 of the paper
            results = objective_function(input, names)

            if len(results) > 0:
                # We have find the failed test and save it in the crash dir.
                for i in results:
                    queue.save_if_interesting(input, batches[idx], True, suffix=i)
                bug_found = True
            else:

                new_img = np.append(ref_batches[idx:idx + 1], batches[idx:idx + 1], axis=0)
                # If it is not a failed test, we will check whether it has a coverage gain
                result = queue.save_if_interesting(input, new_img, False)
                successed = successed or result
        return bug_found, successed

    return func


def dry_run(indir, fetch_function, coverage_function, queue, log):
    seed_lis = os.listdir(indir)
    # Read each initial seed and analyze the coverage
    for seed_name in seed_lis:
        log("Attempting dry run with '%s'...", seed_name)
        path = os.path.join(indir, seed_name)
        img = np.load(path)
        # Each seed will contain two images, i.e., the reference image and mutant (see the paper)
        input_batches = img[1:2]
        # Predict the mutant and obtain the outputs
        # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
        coverage_batches, metadata_batches = fetch_function((0, input_batches, 0, 0, 0))
        # Based on the output, compute the coverage information
        coverage_list = coverage_function(coverage_batches)
        metadata_list = metadata_function(metadata_batches)
        # Create a new seed
        input = Seed(0, coverage_list[0], seed_name, None, metadata_list[0][0], metadata_list[0][0])
        new_img = np.append(input_batches, input_batches, axis=0)
        # Put the seed in the queue and save the npy file in the queue dir
        queue.save_if_interesting(input, new_img, False, True, seed_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')

    parser.add_argument('-i', help='input seed directory')
    parser.add_argument('-o', help='output directory')

    parser.add_argument('-model', help="target model to fuzz", choices=['qcl', 'ccqc', 'qcnn', 'hier'],
                        default='qcl')
    parser.add_argument('-criteria', help="set the criteria to guide the fuzzing",
                        choices=['ksc', 'scc', 'tsc', 'kec'], default='ksc')
    parser.add_argument('-batch_num', help="the number of mutants generated for each seed", type=int, default=20)
    parser.add_argument('-max_iteration', help="maximum number of fuzz iterations", type=int, default=10000000)
    parser.add_argument('-metric_para', help="set the parameter for different metrics", type=float)
    parser.add_argument('-quantize_test', help="fuzzer for quantization", default=0, type=int)
    # parser.add_argument('-ann_threshold', help="Distance below which we consider something new coverage.", type=float,
    #                     default=1.0)
    parser.add_argument('-quan_model_dir', help="directory including the quantized models for testing")
    parser.add_argument('-random', help="whether to adopt random testing strategy", type=int, default=0)
    parser.add_argument('-select', help="test selection strategy",
                        choices=['uniform', 'tensorfuzz', 'deeptest', 'prob'], default='prob')

    args = parser.parse_args()

    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    input_tensor = Input(shape=input_shape)

    # Create the output directory including seed queue and crash dir, it is like AFL
    if os.path.exists(args.o):
        shutil.rmtree(args.o)
    os.makedirs(os.path.join(args.o, 'queue'))
    os.makedirs(os.path.join(args.o, 'crashes'))

    # Load model. For ImageNet, we use the default models from Keras framework.
    # For other models, we load the model from the h5 file.
    model = None
    model = load_model

    # Get the preprocess function based on different dataset
    preprocess = image_preprocessing

    # Load the profiling information which is needed by the metrics in DeepGauge
    profile_dict = pickle.load(open(model_profile_path[args.model], 'rb'))

    # Load the configuration for the selected metrics.
    if args.metric_para is None:
        cri = metrics_para[args.criteria]
    elif args.criteria == 'nc':
        cri = args.metric_para
    else:
        cri = int(args.metric_para)

    # The coverage computer
    coverage_handler = Coverage(model=model, criteria=args.criteria, k=cri,
                                profiling_dict=profile_dict)

    # The log file which records the plot data after each iteration of the fuzzing
    plot_file = open(os.path.join(args.o, 'plot.log'), 'a+')

    # fetch_function is to perform the prediction and obtain the outputs of each layers
    fetch_function = build_fetch_function(coverage_handler, preprocess)
    model_names = [args.model]

    # Like AFL, dry_run will run all initial seeds and keep all initial seeds in the seed queue
    dry_run_fetch = build_fetch_function(coverage_handler, preprocess)

    # The function to update coverage
    coverage_function = coverage_handler.update_coverage
    # The function to perform the mutation from one seed
    mutation_function = image_mutation_function(args.batch_num)

    # The seed queue
    if args.criteria == 'fann':
        queue = TensorInputCorpus(args.o, args.random, args.select, cri, "kdtree")
    else:
        queue = ImageInputCorpus(args.o, args.random, args.select, coverage_handler.total_size, args.criteria)

    # Perform the dry_run process from the initial seeds
    dry_run(args.i, dry_run_fetch, coverage_function, queue)

    # For each seed, compute the coverage and check whether it is a "bug", i.e., adversarial example
    image_iterate_function = iterate_function(model_names)

    # The main fuzzer class
    fuzzer = Fuzzer(queue, coverage_function, metadata_function, objective_function, mutation_function, fetch_function,
                    image_iterate_function, args.select)

    # The fuzzing process
    fuzzer.loop(args.max_iteration)

