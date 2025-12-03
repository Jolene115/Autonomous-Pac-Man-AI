import sys
import util
from pacman import Directions, parseAgentArgs
from q3Model import Q3Model
import samples
import numpy as np
import math
import pandas as pd
from featureExtraction import TRAINING_SET_SIZE, TEST_SET_SIZE


def default(str):
    return str + ' [Default: %default]'


USAGE_STRING = """
  USAGE:      python trainModel.py <options>
                 """


def readCommand(argv):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-t', '--training', help=default('The size of the training set'), default=TRAINING_SET_SIZE, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-v', '--use_validation', help=default("Boolean for where to keep the validation data set seperate or not"), action='store_true', default=False)
    parser.add_option('-m', '--model_save_path', help=default("Where to save your models weights to"), default="./models/q3.model", type="string")
    parser.add_option('-a','--model_args',dest='model_args',help='Comma separated values sent to the model. e.g. "opt1=val1,opt2,opt3=val3"')

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Training Model")
    print("--------------------")

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.test <= 0:
        print("Testing set size should be a positive integer (you provided: %d)" % options.test)
        print(USAGE_STRING)
        sys.exit(2)

    args['training_size'] = options.training
    args['testing_size'] = options.test
    args['use_validation'] = options.use_validation
    args['model_save_path'] = options.model_save_path
    args["model_args"] = options.model_args

    print(args)
    print(options)

    return args, options


def runTraining(args):

    numTraining = args['training_size']
    numTest = args['testing_size']
    useValidation = args['use_validation']
    model_save_path = args['model_save_path']
    model_args = args["model_args"]

    df_train = pd.read_csv("./pacmandata/q3_train.csv").head(numTraining)
    df_validation = pd.read_csv("./pacmandata/q3_validation.csv").head(numTest)
    df_test = pd.read_csv("./pacmandata/q3_test.csv").head(numTest)

    print(len(df_train))
    print(len(df_validation))
    print(len(df_test))

    trainingData = df_train.drop(columns=["label"]).to_numpy()
    trainingLabels = df_train["label"].to_numpy()

    validationData = df_validation.drop(columns=["label"]).to_numpy()
    validationLabels = df_validation["label"].to_numpy()

    # combine the training and validation data if the validation isn't needed
    if not useValidation:
        trainingData = np.vstack([trainingData, validationData])
        trainingLabels = np.hstack([trainingLabels, validationLabels])
        validationData = None
        validationLabels = None

    testData = df_test.drop(columns=["label"]).to_numpy()
    testLabels = df_test["label"].to_numpy()

    # create the model and give it any keyword arguments specified using -a 
    model = Q3Model(**parseAgentArgs(model_args))

    # Conduct training and testing
    print("Training...")
    model.train(trainingData, trainingLabels, validationData, validationLabels)

    print("Testing...")
    test_performance = model.evaluate(testData, testLabels)
    print(test_performance)

    # Save the model weights to file
    model.save(model_save_path)

    return model


if __name__ == '__main__':
    # Read input
    args, options = readCommand(sys.argv[1:])
    # Run classifier
    runTraining(args)