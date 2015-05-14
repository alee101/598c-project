#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tasks
import sys
import run_model


def plot(input_data, expected_output, actual_output):
    """
    Given arrays for the input data, expected output and the actual output,
    Plot the them onto three subplots
    """
    plt.figure(1,figsize=(20,7))
    plt.subplot(311)
    plt.imshow(i.T,interpolation='nearest')
    plt.subplot(312)
    plt.imshow(o.T,interpolation='nearest')
    plt.subplot(313)
    plt.imshow(outputs.T,interpolation='nearest')
    plt.show()


def error_rate(expected_output, actual_output):
    """
    Given the expected output data and the actual output data by the NTM,
    compute the average error rate between the two.
    """
    abs_diff = abs(expected_output - actual_output)
    num_elements = expected_output.shape[0] * expected_output.shape[1]
    return sum(sum(abs_diff)) / num_elements


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python test.py <model_filename> <sequence_length>"
        sys.exit()

    # Make the NTM model and load the parameters
    P, do_task = run_model.make_model()
    P.load(sys.argv[1])
    
    # Randomly generate a copy task and perform the task
    input_data, expected_output = tasks.copy(8,seq_length)
    weights, actual_output = do_task(input_data)

    # Plot the outputs and compute the error_rate
    plot(int(sys.argv[2]))
    print "The average error rate was " + str(error_rate(expected_output, actual_output))

